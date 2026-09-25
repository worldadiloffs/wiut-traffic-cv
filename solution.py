"""
solution.py - WIUT Hackathon 2026, Computer Vision track.

Part A  detect_events(video_path) -> [[start_sec, end_sec, label], ...]
Part B  RiskEstimator: causal P(accident starts within 5 s) per frame

The implementation lives in src/traffic/ (see README.md for the pipeline):
registration to a canonical view -> YOLO11 + ByteTrack -> signal phase from
the lamp pixels -> rule engine per event class. RiskEstimator runs its own
light tracker on the frames it is given and scores time-to-collision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.traffic.pipeline import detect_events as _detect_events  # noqa: E402
from src.traffic.risk import RiskEstimator as _RiskEstimator      # noqa: E402

# Official ids we predict. The metric adds every predicted class to the
# average, so classes our rules cannot recognise reliably are left out.
CLASSES: list[str] = [
    "accident",
    "near_miss",
    "red_light",
    "wrong_way",
    "illegal_u_turn",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "illegal_turn",
    "solid_line_crossing",
    "stop_line",
    "congestion",
    "road_obstacle",
    "fire_smoke",
]

RISK_HORIZON_SEC = 5.0


def detect_events(video_path: str) -> list[list]:
    """Part A - see src/traffic/pipeline.py."""
    return [e for e in _detect_events(video_path) if e[2] in CLASSES]


class RiskEstimator(_RiskEstimator):
    """Part B - see src/traffic/risk.py. step() only uses frames it has seen."""

    def reset(self, meta: dict) -> None:
        super().reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return super().step(frame, t_sec)
