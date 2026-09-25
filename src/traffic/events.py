"""Rule-based traffic events from trajectories + scene layout + signal phase.

Every rule works in canonical normalised coordinates (see register.py) and
returns a list of (start_sec, end_sec) intervals for its class. Intervals of
one class are merged at the end (the metric treats simultaneous events of a
class as one segment).

What is learned vs hand-written:
  learned  : detector (COCO YOLO), road mask + flow field (from sample videos),
             per-video signal normalisation
  rules    : everything in this file; thresholds were tuned on our own dev
             labels of the four sample clips.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scene import Scene, points_in_poly
from .signal import GREEN, RED, phase_at
from .tracks import Track, rider_person_ids

ASPECT = np.array([1.0, 9 / 16])   # normalised y -> units of frame width (16:9 frames)
CAR_LIKE = {2, 5, 7}          # car, bus, truck
MOTOR = {2, 3, 5, 7}          # + motorcycle
ANY_VEHICLE = {1, 2, 3, 5, 7}  # + bicycle


@dataclass
class Params:
    # stationarity (box heights per second)
    still: float = 0.12
    moving: float = 0.35
    # jaywalking
    jay_min_dur: float = 1.2
    jay_cross_buffer: float = 0.03
    jay_kerb_margin: float = 0.008
    jay_min_path: float = 0.03
    jay_min_net: float = 0.02
    # failure to yield
    fty_ped_dist: float = 0.16
    fty_min_samples: int = 2
    fty_u_margin: float = 0.12
    fty_max_cross: float = 6.0
    fty_ped_moving: float = 0.25
    # red light
    red_grace: float = 0.8           # s after red onset
    red_tail: float = 3.0            # s before red ends (phase-change latency)
    red_platoon: int = 3             # >= this many crossings within +-2 s = queue discharge (green)
    # stop line
    stop_min_still: float = 2.0
    stop_margin: float = 0.012
    # stopped vehicle
    stopped_min: float = 10.0
    queue_gap: float = 0.035
    # congestion
    cong_min_veh: int = 6
    cong_speed: float = 0.10
    cong_min_dur: float = 12.0
    # wrong way
    ww_coh: float = 0.85
    ww_cnt: float = 25.0
    ww_cos: float = -0.6
    ww_min_dur: float = 1.5
    ww_min_dist: float = 0.05
    enabled: tuple = ("jaywalking", "failure_to_yield", "red_light", "stop_line",
                      "stopped_vehicle", "congestion", "wrong_way")


# ---------------------------------------------------------------- helpers ----
def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index runs [i, j] (inclusive) where mask is True."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def time_runs(t: np.ndarray, mask: np.ndarray, max_gap: float, min_len: float) -> list[tuple[float, float]]:
    """Runs of True in a sampled boolean series, bridging gaps < max_gap seconds."""
    segs = [(t[i], t[j]) for i, j in runs(mask)]
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(float(s), float(e)) for s, e in merged if e - s >= min_len]


def union(intervals: list[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    iv = sorted((float(s), float(e)) for s, e in intervals if e > s)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def dist_to_poly(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from each point to a polygon (0 inside)."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    inside = points_in_poly(pts, poly)
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    ap = pts[:, None, :] - a[None]
    tt = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-12)[None], 0, 1)
    proj = a[None] + tt[..., None] * ab[None]
    d = np.linalg.norm(pts[:, None, :] - proj, axis=-1).min(1)
    d[inside] = 0.0
    return d


def _box_gap(a: np.ndarray, b: np.ndarray) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


@dataclass
class Context:
    tracks: list[Track]
    scene: Scene
    sig_t: np.ndarray
    sig_phase: np.ndarray
    duration: float
    params: Params = field(default_factory=Params)

    def __post_init__(self):
        riders = rider_person_ids(self.tracks)
        moto_riders = rider_person_ids(self.tracks, classes=(3,))
        self.peds = [tr for tr in self.tracks if tr.is_person and tr.tid not in riders and len(tr.t) >= 4]
        # people on a crossing include those pushing (or riding) a bicycle; not motorcyclists
        self.crossers = [tr for tr in self.tracks if tr.is_person and tr.tid not in moto_riders and len(tr.t) >= 4]
        self.vehicles = [tr for tr in self.tracks if tr.cls in ANY_VEHICLE and len(tr.t) >= 4]
        self.has_signal = bool(len(self.sig_phase)) and bool(np.any(self.sig_phase != 0))
        # per-time index of pedestrian ground points for fast lookups
        self._ped_at: dict[float, np.ndarray] = {}
        for p in self.crossers:
            for t, g, sp in zip(p.t, p.gp_s, p.speed_rel):
                self._ped_at.setdefault(round(float(t), 3), []).append((g[0], g[1], sp))
        self._ped_at = {k: np.asarray(v) for k, v in self._ped_at.items()}
        self._veh_at: dict[float, list] = {}
        for v in self.vehicles:
            if v.cls not in CAR_LIKE:
                continue
            for i, t in enumerate(v.t):
                self._veh_at.setdefault(round(float(t), 3), []).append((v.tid, v.gp_s[i], v.speed_rel[i], v.box[i]))

        self.evidence: list[tuple[str, float, float, list[int]]] = []

    def ev(self, name: str, s: float, e: float, tids) -> tuple[float, float]:
        self.evidence.append((name, float(s), float(e), [int(x) for x in tids]))
        return (float(s), float(e))

    def phase(self, t):
        return phase_at(self.sig_t, self.sig_phase, t)

    def peds_at(self, t: float) -> np.ndarray:
        """(k, 3): x, y, speed_rel of (non-rider) pedestrians at time t."""
        return self._ped_at.get(round(float(t), 3), np.zeros((0, 3)))


# ------------------------------------------------------------------ rules ----
def jaywalking(ctx: Context) -> list[tuple[float, float]]:
    P, sc = ctx.params, ctx.scene
    out = []
    polys = list(sc.crosswalks.values())
    for p in ctx.peds:
        g = p.gp_s
        on_road = sc.on_road(g, margin=P.jay_kerb_margin)
        near_cross = np.zeros(len(g), bool)
        for poly in polys:
            near_cross |= dist_to_poly(g, poly) < P.jay_cross_buffer
        flag = on_road & ~near_cross & ~sc.in_zone(g, "bus_stop")
        for a, b in time_runs(p.t, flag, max_gap=0.8, min_len=P.jay_min_dur):
            i, j = p.at(a), p.at(b)
            path = np.linalg.norm(np.diff(g[i:j + 1], axis=0), axis=1).sum()
            net = np.linalg.norm(g[j] - g[i])
            if path < P.jay_min_path or net < P.jay_min_net:   # static "person" (painted trunk, sign)
                continue
            out.append(ctx.ev("jaywalking", a, b, [p.tid]))
    return out


def failure_to_yield(ctx: Context) -> list[tuple[float, float]]:
    P, sc = ctx.params, ctx.scene
    out = []
    for v in ctx.vehicles:
        if v.cls not in MOTOR:
            continue
        g = v.gp_s
        for cw_name, poly in sc.crosswalks.items():
            # a box cut by the bottom edge has no reliable ground point
            inside = points_in_poly(g, poly) & (v.box[:, 3] < 0.985)
            if inside.sum() < 2:
                continue
            for i, j in runs(inside):
                if v.t[j] - v.t[i] > P.fty_max_cross or np.median(v.speed_rel[i:j + 1]) < P.moving:
                    continue            # stood / crawled on the crossing -> not "drives through"
                hits = 0
                for k in range(i, j + 1):
                    peds = ctx.peds_at(v.t[k])
                    if len(peds) == 0:
                        continue
                    xy = peds[:, :2]
                    u = sc.crosswalk_u(xy, cw_name)
                    walking = peds[:, 2] > P.fty_ped_moving
                    mid = (u > 0.2) & (u < 0.8)          # standing mid-crossing (waiting for the car) counts too
                    on_cw = (points_in_poly(xy, poly) & (u > P.fty_u_margin) & (u < 1 - P.fty_u_margin)
                             & (walking | mid))
                    if not on_cw.any():
                        continue
                    d = np.linalg.norm((xy[on_cw] - g[k]) * ASPECT, axis=1)
                    if (d < P.fty_ped_dist).any():
                        hits += 1
                if hits >= P.fty_min_samples:
                    out.append(ctx.ev("failure_to_yield", v.t[i], v.t[j] + 0.15, [v.tid]))
    return out


def red_light(ctx: Context) -> list[tuple[float, float]]:
    P, sc = ctx.params, ctx.scene
    if not ctx.has_signal:
        return []
    out, cands, all_cross = [], [], []
    red_iv = [(s, e) for s, e in _phase_intervals(ctx, RED)]
    for v in ctx.vehicles:
        if v.cls not in MOTOR:
            continue
        side = sc.side_of_stop_line(v.gp_s)
        in_app = sc.in_zone(v.gp_s, "near_approach") | sc.in_zone(v.gp_s, "past_stop_line")
        cross = np.where((side[:-1] > 0) & (side[1:] <= 0) & in_app[:-1])[0]
        if len(cross) == 0:
            continue
        k = cross[0] + 1
        tc = float(v.t[k])
        all_cross.append((tc, v.tid))
        if not any(s + P.red_grace <= tc <= e - P.red_tail for s, e in red_iv):
            continue
        # must continue into the intersection (else it is a stop-line case)
        into = sc.in_zone(v.gp_s[k:], "intersection")
        if not into.any():
            continue
        # must not stop on the crossing for long before entering
        j_in = k + int(np.argmax(into))
        if v.speed_rel[k:j_in + 1].min() < P.still and (v.t[j_in] - tc) > 3.0:
            continue
        # end: leaves the intersection or the frame
        after = sc.in_zone(v.gp_s[j_in:], "intersection")
        j_out = j_in + (int(np.argmin(after)) if (~after).any() else len(after) - 1)
        cands.append((tc, float(min(v.t[j_out], tc + 12.0)), v.tid))
    # a platoon of vehicles crossing together means the approach actually has green
    times = np.array([c[0] for c in all_cross]) if all_cross else np.zeros(0)
    for tc, te, tid in cands:
        if (np.abs(times - tc) <= 2.0).sum() >= P.red_platoon:
            continue
        out.append(ctx.ev("red_light", tc, te, [tid]))
    return out


def stop_line(ctx: Context) -> list[tuple[float, float]]:
    P, sc = ctx.params, ctx.scene
    if not ctx.has_signal:
        return []
    out = []
    greens = [s for s, _ in _phase_intervals(ctx, GREEN)]
    for v in ctx.vehicles:
        if v.cls not in MOTOR:
            continue
        g = v.gp_s
        past = sc.in_zone(g, "past_stop_line") & (sc.side_of_stop_line(g) < -P.stop_margin)
        still = v.speed_rel < P.still
        red = ctx.phase(v.t) == RED
        flag = past & still & red
        for s, e in time_runs(v.t, flag, max_gap=1.0, min_len=P.stop_min_still):
            nxt = [gs for gs in greens if gs > s]
            end = nxt[0] if nxt else ctx.duration
            out.append(ctx.ev("stop_line", s, end, [v.tid]))
    return out


def stopped_vehicle(ctx: Context) -> list[tuple[float, float]]:
    P, sc = ctx.params, ctx.scene
    out = []
    for v in ctx.vehicles:
        if v.cls not in CAR_LIKE or v.duration < P.stopped_min:
            continue
        g = v.gp_s
        still = (v.speed_rel < P.still) & sc.on_road(g)
        for s, e in time_runs(v.t, still, max_gap=1.5, min_len=P.stopped_min):
            i, j = v.at(s), v.at(e)
            mid = (i + j) // 2
            pt = g[mid:mid + 1]
            if v.cls == 5 and sc.in_zone(pt, "bus_stop")[0]:
                continue
            if sc.in_zone(pt, "near_approach")[0] or sc.in_zone(pt, "past_stop_line")[0]:
                continue                        # signal queue
            if min(dist_to_poly(pt, poly)[0] for poly in sc.crosswalks.values()) < 0.03:
                continue                        # waiting at a crossing
            bx = v.box[mid]
            if bx[0] < 0.005 or bx[2] > 0.995 or bx[3] > 0.995:
                continue                        # cut by the frame edge
            # queue check: other stationary vehicles close by for most of the stop
            queued = 0
            for k in range(i, j + 1, 3):
                others = ctx._veh_at.get(round(float(v.t[k]), 3), [])
                near_still = sum(1 for tid, og, osp, ob in others
                                 if tid != v.tid and osp < P.moving and _box_gap(ob, v.box[k]) < P.queue_gap)
                queued += near_still >= 1
            if queued > 0.5 * len(range(i, j + 1, 3)):
                continue
            out.append(ctx.ev("stopped_vehicle", s, e, [v.tid]))
    return out


def congestion(ctx: Context) -> list[tuple[float, float]]:
    P, sc = ctx.params, ctx.scene
    times = np.array(sorted(ctx._veh_at))
    if len(times) == 0:
        return []
    out = []
    for zone in ("near_approach", "far_carriageway"):
        flag = np.zeros(len(times), bool)
        for n, t in enumerate(times):
            items = ctx._veh_at[t]
            pts = np.array([it[1] for it in items])
            spd = np.array([it[2] for it in items])
            inz = sc.in_zone(pts, zone)
            if inz.sum() < P.cong_min_veh:
                continue
            if np.median(spd[inz]) < P.cong_speed and np.mean(spd[inz] < P.moving) > 0.8:
                flag[n] = True
        if zone == "near_approach" and ctx.has_signal:
            flag &= ctx.phase(times) == GREEN      # a queue at red is not congestion
        out += time_runs(times, flag, max_gap=3.0, min_len=P.cong_min_dur)
    return out


def wrong_way(ctx: Context) -> list[tuple[float, float]]:
    P, sc = ctx.params, ctx.scene
    if sc.flow is None:
        return []
    out = []
    for v in ctx.vehicles:
        g, vel = v.gp_s, v.vel
        spd = np.linalg.norm(vel, axis=1)
        moving = v.speed_rel > P.moving
        if moving.sum() < 4:
            continue
        f, coh, cnt = sc.flow_at(g)
        u = vel / np.maximum(spd[:, None], 1e-9)
        cos = (u * f).sum(1)
        valid = moving & (coh > P.ww_coh) & (cnt > P.ww_cnt) & ~sc.in_zone(g, "intersection")
        against = valid & (cos < P.ww_cos)
        for s, e in time_runs(v.t, against, max_gap=1.0, min_len=P.ww_min_dur):
            i, j = v.at(s), v.at(e)
            if np.linalg.norm(g[j] - g[i]) < P.ww_min_dist:
                continue
            # most valid samples in the run must disagree with the flow
            vv = valid[i:j + 1]
            if vv.sum() and against[i:j + 1][vv].mean() < 0.7:
                continue
            out.append(ctx.ev("wrong_way", s, e, [v.tid]))
    return out


def _phase_intervals(ctx: Context, value: int):
    from .signal import phase_intervals
    return phase_intervals(ctx.sig_t, ctx.sig_phase, value)


MERGE_GAP = {"jaywalking": 2.0, "failure_to_yield": 0.5, "congestion": 5.0}

RULES = {
    "jaywalking": jaywalking,
    "failure_to_yield": failure_to_yield,
    "red_light": red_light,
    "stop_line": stop_line,
    "stopped_vehicle": stopped_vehicle,
    "congestion": congestion,
    "wrong_way": wrong_way,
}


def detect_all(ctx: Context) -> list[list]:
    events = []
    for name in ctx.params.enabled:
        try:
            raw = RULES[name](ctx)
        except Exception as exc:          # one faulty rule must not cost the other classes
            import sys
            print(f"[events] rule {name} failed: {exc!r}", file=sys.stderr)
            continue
        ivs = union(raw, gap=MERGE_GAP.get(name, 0.3))
        for s, e in ivs:
            s, e = max(0.0, s), min(ctx.duration, e)
            if e - s >= 0.2:
                events.append([round(s, 2), round(e, 2), name])
    events.sort()
    return events
