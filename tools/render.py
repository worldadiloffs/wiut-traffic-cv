"""Render an annotated copy of a video + a JSON summary for the website.

    python tools/render.py VIDEO OUT_DIR [--width 960] [--risk risk.json]

Overlays: tracked boxes (colour by class, red when part of an event),
crossings / stop line, signal phase, and a banner for every active event.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.traffic.pipeline import analyze_extracted, extract          # noqa: E402
from src.traffic.register import apply_h                              # noqa: E402
from src.traffic.scene import get_scene                               # noqa: E402
from src.traffic.signal import GREEN, RED, phase_at                   # noqa: E402

COL = {0: (255, 200, 60), 1: (255, 120, 255), 2: (80, 220, 80), 3: (255, 120, 255),
       5: (60, 200, 255), 7: (60, 160, 255)}
EV_COL = {"jaywalking": (0, 170, 255), "failure_to_yield": (0, 80, 255), "red_light": (0, 0, 255),
          "stop_line": (0, 140, 255), "stopped_vehicle": (255, 0, 200), "congestion": (200, 120, 0),
          "wrong_way": (0, 0, 180), "near_miss": (0, 255, 255), "accident": (0, 0, 255)}
NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def operator_stats(an) -> dict:
    """Pedestrian compliance on crossing A and approach throughput during green."""
    from src.traffic.signal import GREEN, RED, phase_at
    ctx, sc = an.ctx, get_scene()
    poly = sc.crosswalks["A_near_arm"]
    red_s = green_s = 0.0
    for p in ctx.peds:
        u = sc.crosswalk_u(p.gp_s, "A_near_arm")
        on = sc.in_crosswalk(p.gp_s, "A_near_arm") & (u > 0.15) & (u < 0.85) & (p.speed_rel > 0.25)
        if on.sum() < 3:
            continue
        ph = phase_at(ctx.sig_t, ctx.sig_phase, p.t[on])
        dt = np.median(np.diff(p.t)) if len(p.t) > 1 else 0.13
        red_s += float((ph == RED).sum() * dt)
        green_s += float((ph == GREEN).sum() * dt)
    crossings = []
    for v in ctx.vehicles:
        side = sc.side_of_stop_line(v.gp_s)
        k = np.where((side[:-1] > 0) & (side[1:] <= 0))[0]
        if len(k):
            crossings.append(v.t[k[0] + 1])
    from src.traffic.signal import phase_intervals
    greens = phase_intervals(ctx.sig_t, ctx.sig_phase, GREEN)
    g_len = sum(b - a for a, b in greens)
    n_g = sum(1 for t in crossings if phase_at(ctx.sig_t, ctx.sig_phase, t) == GREEN)
    tot = red_s + green_s
    return {"ped_against_pct": (100 * green_s / tot) if tot > 0 else None,
            "veh_per_min_green": (60 * n_g / g_len) if g_len > 5 else None}


def summarise(an, name: str) -> dict:
    ctx = an.ctx
    dur = an.info["duration"]
    # per-second counts of tracked objects by class
    rows = an.rows
    secs = np.floor(rows[:, 1]).astype(int)
    counts = {}
    for c in (0, 2, 3, 5, 7, 1):
        m = rows[:, 3] == c
        n_frames = np.bincount(secs[m], minlength=int(dur) + 1)
        frames_per_sec = np.bincount(np.floor(np.unique(rows[:, 1])).astype(int), minlength=int(dur) + 1)
        counts[NAMES[c]] = np.round(n_frames / np.maximum(frames_per_sec, 1), 2).tolist()
    ph = [[round(float(t), 2), int(p)] for t, p in zip(an.lamp_t[::4], an.phase[::4])]
    return {
        "ops": operator_stats(an),
        "video": name, "duration": round(dur, 2), "fps": an.info["fps"],
        "size": [an.info["width"], an.info["height"]],
        "registration": an.reg_info,
        "events": an.events,
        "evidence": [[e[0], round(e[1], 2), round(e[2], 2), e[3]] for e in ctx.evidence],
        "counts_per_sec": counts,
        "phase": ph,
        "n_tracks": {NAMES.get(c, str(c)): int(sum(1 for t in ctx.tracks if t.cls == c)) for c in NAMES},
    }


def render(video: str, out_dir: str, width: int = 960, risk: list | None = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(video).stem
    ex = extract(video)
    an = analyze_extracted(ex)
    summ = summarise(an, Path(video).name)
    if risk is not None:
        summ["risk"] = risk
    (out / f"{name}.json").write_text(json.dumps(summ))

    sc = get_scene()
    Hi = np.linalg.inv(an.H)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = an.info["width"], an.info["height"]
    h_out = int(round(H * width / W / 2) * 2)
    s = np.array([width, h_out], float)
    ev_tids: dict[int, list] = {}
    for c, a, b, tids in an.ctx.evidence:
        for tid in tids:
            ev_tids.setdefault(tid, []).append((a, b, c))
    rows = an.rows
    by_frame: dict[int, np.ndarray] = {}
    for f in np.unique(rows[:, 0]).astype(int):
        by_frame[f] = rows[rows[:, 0] == f]
    frames_sorted = np.array(sorted(by_frame))
    polys = [(apply_h(Hi, p) * s).astype(np.int32) for p in sc.crosswalks.values()]
    stop = (apply_h(Hi, sc.stop_line) * s).astype(np.int32)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{width}x{h_out}", "-r", str(fps), "-i", "-",
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p",
                             "-movflags", "+faststart", str(out / f"{name}_annotated.mp4")],
                            stdin=subprocess.PIPE)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        img = cv2.resize(frame, (width, h_out), interpolation=cv2.INTER_AREA)
        ov = img.copy()
        for p in polys:
            cv2.polylines(ov, [p], True, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(ov, tuple(stop[0]), tuple(stop[1]), (0, 0, 255), 2, cv2.LINE_AA)
        img = cv2.addWeighted(ov, 0.6, img, 0.4, 0)
        # nearest analysed frame at or before idx
        k = np.searchsorted(frames_sorted, idx, side="right") - 1
        if k >= 0 and idx - frames_sorted[k] < 8:
            for r in by_frame[frames_sorted[k]]:
                tid, cls = int(r[2]), int(r[3])
                x1, y1, x2, y2 = (r[5:9] * width / W).astype(int)
                hot = [c for a, b, c in ev_tids.get(tid, []) if a - 0.2 <= t <= b + 0.2]
                col = EV_COL.get(hot[0], (0, 0, 255)) if hot else COL.get(cls, (200, 200, 200))
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if hot else 1, cv2.LINE_AA)
                if hot:
                    cv2.putText(img, hot[0], (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        # signal indicator
        ph = int(phase_at(an.lamp_t, an.phase, t)) if len(an.lamp_t) else 0
        colp = (0, 0, 230) if ph == RED else ((0, 200, 0) if ph == GREEN else (120, 120, 120))
        cv2.rectangle(img, (0, 0), (width, 26), (20, 20, 20), -1)
        cv2.circle(img, (14, 13), 8, colp, -1)
        txt = f"{name}  t={t:6.1f}s   signal: {'RED' if ph == RED else ('GREEN' if ph == GREEN else '?')}"
        cv2.putText(img, txt, (30, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        active = [e for e in an.events if e[0] <= t <= e[1]]
        for j, (a, b, c) in enumerate(active[:5]):
            y = 30 + 24 * j
            cv2.rectangle(img, (width - 250, y), (width - 4, y + 20), EV_COL.get(c, (0, 0, 255)), -1)
            cv2.putText(img, f"{c}  {a:.1f}-{b:.1f}s", (width - 244, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        proc.stdin.write(img.tobytes())
        idx += 1
    proc.stdin.close()
    proc.wait()
    cap.release()
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--risk", default=None)
    a = ap.parse_args()
    risk = json.loads(Path(a.risk).read_text()) if a.risk else None
    render(a.video, a.out, a.width, risk)
