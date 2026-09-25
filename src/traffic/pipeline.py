"""End-to-end Part A pipeline: video -> events.

    1. register the view to the canonical layout (SIFT + RANSAC homography)
    2. one sequential pass: YOLO11 + ByteTrack on ~7.5 fps, signal-lamp scores
    3. tracks -> canonical coordinates -> rules (events.py)

`analyze(video)` returns everything the website / renderer needs;
`detect_events(video)` returns only the event list for the harness.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibrate import to_canonical
from .events import Context, Params, detect_all
from .register import estimate_homography, sample_background
from .scene import get_scene
from .signal import lamp_rois, lamp_scores, phase_series
from .tracking import TrackerConfig, track_video, video_info
from .tracks import build_tracks

# Budget: the harness allows 3 x duration for Part A + Part B together.
PART_A_BUDGET = float(os.environ.get("TRAFFIC_PART_A_BUDGET", "1.8"))
CACHE_DIR = os.environ.get("TRAFFIC_CACHE")      # optional: reuse tracks across runs

# Shared with risk.py so Part B can back off if Part A used most of the budget.
RUN_CLOCK: dict[str, float] = {}


@dataclass
class Analysis:
    info: dict
    H: np.ndarray
    reg_info: dict
    rows: np.ndarray            # raw tracker rows (video pixels)
    lamp_t: np.ndarray
    lamp_s: np.ndarray          # (n, 2) red / green scores
    phase: np.ndarray
    ctx: Context
    events: list


def _cache_path(video: str) -> Path | None:
    if not CACHE_DIR:
        return None
    st = os.stat(video)
    key = f"{Path(video).stem}_{st.st_size}"
    return Path(CACHE_DIR) / f"{key}.npz"


def extract(video: str, cfg: TrackerConfig | None = None, deadline: float | None = None,
            progress: bool = False) -> dict:
    """Heavy part: registration + tracking + lamp scores (cached if TRAFFIC_CACHE is set)."""
    cp = _cache_path(video)
    if cp is not None and cp.exists():
        d = np.load(cp, allow_pickle=True)
        return {k: (json.loads(str(d[k])) if k in ("info", "reg_info") else d[k]) for k in d.files}
    try:
        bg = sample_background(video)
        H, reg_info = estimate_homography(bg)
    except Exception as exc:              # registration is a refinement, never a blocker
        H, reg_info = np.eye(3), {"status": f"identity (error: {exc!r})"}
    rois = lamp_rois(get_scene(), H)
    lamp_t, lamp_s = [], []

    def hook(idx, t, frame):
        lamp_t.append(t)
        lamp_s.append(lamp_scores(frame, rois))

    rows, info = track_video(video, cfg or TrackerConfig(), progress=progress,
                             deadline=deadline, frame_hook=hook)
    out = {"rows": rows, "info": info, "H": H, "reg_info": reg_info,
           "lamp_t": np.asarray(lamp_t, float), "lamp_s": np.asarray(lamp_s, float).reshape(-1, 2)}
    if cp is not None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cp, rows=rows, H=H, lamp_t=out["lamp_t"], lamp_s=out["lamp_s"],
                            info=json.dumps(info), reg_info=json.dumps(reg_info))
    return out


def analyze_extracted(ex: dict, params: Params | None = None) -> Analysis:
    info = ex["info"]
    w, h = info["width"], info["height"]
    rows = to_canonical(ex["rows"], w, h, ex["H"])
    tracks = build_tracks(rows, w, h)
    fps_s = info["fps"] / max(1, info.get("stride", 1))
    phase = phase_series(ex["lamp_t"], ex["lamp_s"], fps_s)
    duration = info["duration"]
    ctx = Context(tracks, get_scene(), ex["lamp_t"], phase, duration, params or Params())
    events = detect_all(ctx)
    return Analysis(info, ex["H"], ex["reg_info"], ex["rows"], ex["lamp_t"], ex["lamp_s"],
                    phase, ctx, events)


def analyze(video: str, params: Params | None = None, progress: bool = False) -> Analysis:
    t0 = time.perf_counter()
    duration = max(1.0, video_info(video)["duration"])
    ex = extract(video, deadline=t0 + PART_A_BUDGET * duration, progress=progress)
    return analyze_extracted(ex, params)


def detect_events(video: str) -> list[list]:
    RUN_CLOCK["start"] = time.perf_counter()
    RUN_CLOCK["duration"] = max(1.0, video_info(video)["duration"])
    return analyze(video).events
