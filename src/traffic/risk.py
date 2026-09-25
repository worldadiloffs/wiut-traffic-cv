"""Part B - causal accident anticipation.

The estimator only sees frames in order. Every ~0.2 s it runs a light
detector + tracker, keeps short causal velocity estimates per track and
computes, for every pair of road users, the time and distance of closest
approach under constant velocity. A pair on a collision course (closest
approach < sum of half-sizes) within the 5 s horizon contributes

    risk_pair = closeness * exp(-t_closest / tau) * closing_speed_factor

The frame score is the max over pairs, smoothed with an exponential moving
average and mapped so that ~0.5 corresponds to a sustained, fast collision
course. Hard braking of a tracked vehicle adds a small bonus.
"""
from __future__ import annotations

import os
import time
from collections import deque

import cv2
import numpy as np

from .tracking import PERSON, VEHICLES, Tracker, TrackerConfig

HORIZON = 5.0


class _TrackState:
    __slots__ = ("hist",)

    def __init__(self):
        self.hist: deque = deque(maxlen=8)   # (t, cx, cy, w, h)


class RiskEstimator:
    def __init__(self):
        self.cfg = TrackerConfig(model=os.environ.get("TRAFFIC_RISK_MODEL", "yolo11n.pt"),
                                 imgsz=int(os.environ.get("TRAFFIC_RISK_IMGSZ", "960")),
                                 target_fps=float(os.environ.get("TRAFFIC_RISK_FPS", "5")))
        self.tracker = None
        self.tau = 2.0

    # ------------------------------------------------------------------
    def reset(self, meta: dict) -> None:
        self.meta = meta
        fps = float(meta.get("fps") or 25.0)
        self.stride = max(1, int(round(fps / self.cfg.target_fps)))
        self.n = 0
        self.score = 0.0
        self.ema = 0.0
        self.states: dict[int, _TrackState] = {}
        # fresh tracker per video (ByteTrack state must not leak between videos)
        self.tracker = Tracker(self.cfg)
        self.t_start = time.perf_counter()
        from .pipeline import RUN_CLOCK
        self.clock = dict(RUN_CLOCK)

    def _over_budget(self) -> bool:
        """Back off if the whole video is close to the harness time limit."""
        start, dur = self.clock.get("start"), self.clock.get("duration")
        if start is None or dur is None:
            return False
        return time.perf_counter() - start > 2.7 * dur

    # ------------------------------------------------------------------
    def step(self, frame: np.ndarray, t_sec: float) -> float:
        i = self.n
        self.n += 1
        if i % self.stride != 0:
            return self.score
        if self._over_budget():
            return self.score
        h, w = frame.shape[:2]
        if w > 1920:   # the detector resizes anyway; this keeps the copy cheap
            frame = cv2.resize(frame, (1920, int(h * 1920 / w)), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
        det = self.tracker.update(frame)
        raw = self._pair_risk(det, t_sec, w, h)
        # asymmetric smoothing: rise fast, decay slowly
        a = 0.6 if raw > self.ema else 0.15
        self.ema = (1 - a) * self.ema + a * raw
        self.score = float(np.clip(self.ema, 0.0, 1.0))
        return self.score

    # ------------------------------------------------------------------
    def _pair_risk(self, det: np.ndarray, t: float, w: int, h: int) -> float:
        alive = set()
        items = []
        for tid, cls, conf, x1, y1, x2, y2 in det:
            cls = int(cls)
            if cls not in VEHICLES and cls != PERSON:
                continue
            tid = int(tid)
            alive.add(tid)
            st = self.states.setdefault(tid, _TrackState())
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            st.hist.append((t, (x1 + x2) / 2 / w, y2 / h, bw, bh))
            if len(st.hist) >= 3:
                items.append((tid, cls, st))
        for tid in list(self.states):
            if tid not in alive and self.states[tid].hist and t - self.states[tid].hist[-1][0] > 2.0:
                del self.states[tid]
        if len(items) < 2:
            return 0.0
        P, V, R, C, B = [], [], [], [], []
        for tid, cls, st in items:
            hs = np.asarray(st.hist)
            tt, xy = hs[:, 0], hs[:, 1:3]
            dt = tt[-1] - tt[0]
            v = (xy[-1] - xy[0]) / max(dt, 1e-3)
            # braking: speed now vs speed a moment ago
            k = len(hs) // 2
            v_old = (xy[k] - xy[0]) / max(tt[k] - tt[0], 1e-3)
            v_new = (xy[-1] - xy[k]) / max(tt[-1] - tt[k], 1e-3)
            size = max(hs[-1, 4], 1e-3)
            brake = max(0.0, (np.linalg.norm(v_old) - np.linalg.norm(v_new)) / size)
            P.append(xy[-1]); V.append(v); R.append(0.5 * max(hs[-1, 3], 0.3 * hs[-1, 4]))
            C.append(cls); B.append(brake)
        P, V, R, C, B = map(np.asarray, (P, V, R, C, B))
        dp = P[:, None, :] - P[None, :, :]
        dv = V[:, None, :] - V[None, :, :]
        dv2 = (dv ** 2).sum(-1)
        tstar = -(dp * dv).sum(-1) / np.maximum(dv2, 1e-9)
        dmin = np.linalg.norm(dp + dv * tstar[..., None], axis=-1)
        reach = (R[:, None] + R[None, :])
        closing = np.sqrt(dv2)
        n = len(P)
        iu = np.triu_indices(n, 1)
        ts, dm, rc, cl = tstar[iu], dmin[iu], reach[iu], closing[iu]
        vehicle_pair = np.isin(C[iu[0]], list(VEHICLES)) | np.isin(C[iu[1]], list(VEHICLES))
        spd = np.linalg.norm(V, axis=1) / np.maximum(np.asarray(R) * 2, 1e-3)   # sizes per second
        both_known = (spd[iu[0]] > 0.3) | (spd[iu[1]] > 0.3)
        ok = (ts > 0) & (ts < 3.5) & (dm < 0.5 * rc) & vehicle_pair & both_known
        # ignore pairs that already overlap and move together (queues, occlusions)
        cur = np.linalg.norm(dp, axis=-1)[iu]
        ok &= cur > 0.6 * rc
        if not ok.any():
            self._pairs = {}
            return float(min(0.1, 0.03 * B.max())) if len(B) else 0.0
        closeness = 1.0 - dm[ok] / (0.5 * rc[ok])
        speed_f = np.clip(cl[ok] / (rc[ok] * 3.0 + 1e-6), 0.0, 1.0)
        r = closeness ** 2 * np.exp(-ts[ok] / 1.5) * speed_f
        # persistence: a pair must stay on a collision course for >= 2 evaluations
        keys = [(int(a), int(b)) for a, b in zip(np.asarray([it[0] for it in items])[iu[0]][ok],
                                                   np.asarray([it[0] for it in items])[iu[1]][ok])]
        prev = getattr(self, "_pairs", {})
        cur_pairs = {k: prev.get(k, 0) + 1 for k in keys}
        self._pairs = cur_pairs
        r = r * np.array([min(1.0, (cur_pairs[k] - 1) / 2.0) for k in keys])
        brake_bonus = 0.05 * np.clip(B.max(), 0, 2)
        return float(np.clip(r.max() + brake_bonus, 0.0, 1.0))
