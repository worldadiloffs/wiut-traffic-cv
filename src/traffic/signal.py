"""Traffic-signal phase from the pixels of one signal head.

The vehicle signal on the median tip is visible in every clip. For each
analysed frame we measure how "lit" the red and the green lamp are:

    red_score   = max over the red-lamp disc of  R - max(G, B)
    green_score = max over the green-lamp disc of G - R      (the lamp is teal)

In dusk footage both scores reach ~200 when lit; in bright daylight the lit
lamp is much dimmer (~30-80), so each video is normalised by the high
percentile of its own scores (Part A may look at the whole video). A
hysteresis filter removes flicker from passing vehicles and the amber phase.
Output phase per sample: +1 red, -1 green, 0 unknown.
"""
from __future__ import annotations

import numpy as np

RED, GREEN, UNKNOWN = 1, -1, 0
LAMP_RADIUS_PX_720 = 6.0     # search radius around the lamp centre at 1280x720


def _patch(frame: np.ndarray, centre_n, radius_n) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = centre_n[0] * w, centre_n[1] * h
    r = max(2.0, radius_n * w)
    x0, x1 = int(cx - r), int(cx + r) + 1
    y0, y1 = int(cy - r), int(cy + r) + 1
    return frame[max(0, y0):y1, max(0, x0):x1].reshape(-1, 3).astype(np.int32)


def lamp_rois(scene, H_video_to_canon: np.ndarray | None = None):
    """Lamp centres in normalised VIDEO coordinates (+ search radius)."""
    from .register import apply_h
    red, green = scene.red_lamp, scene.green_lamp
    if H_video_to_canon is not None:
        Hi = np.linalg.inv(H_video_to_canon)
        red, green = apply_h(Hi, red)[0], apply_h(Hi, green)[0]
    return red, green, LAMP_RADIUS_PX_720 / 1280.0


def lamp_scores(frame: np.ndarray, rois) -> tuple[float, float]:
    red_c, green_c, radius = rois
    r = _patch(frame, red_c, radius)
    g = _patch(frame, green_c, radius)
    if r.size == 0 or g.size == 0:
        return 0.0, 0.0
    rs = float((r[:, 2] - np.maximum(r[:, 0], r[:, 1])).max())
    gs = float((g[:, 1] - g[:, 2]).max())
    return rs, gs


def classify(scores: np.ndarray, min_contrast: float = 15.0) -> np.ndarray:
    """scores (n, 2) -> raw per-sample reading (+1 / -1 / 0), normalised per video."""
    scores = np.asarray(scores, float).reshape(-1, 2)
    if len(scores) == 0:
        return np.zeros(0, int)
    hi = np.percentile(scores, 95, axis=0)
    if hi[0] < min_contrast or hi[1] < min_contrast:
        return np.zeros(len(scores), int)       # lamp not readable in this clip
    n = scores / hi
    red = (n[:, 0] > 0.45) & (n[:, 0] > n[:, 1] + 0.2)
    green = (n[:, 1] > 0.45) & (n[:, 1] > n[:, 0] + 0.2)
    out = np.zeros(len(scores), int)
    out[red] = RED
    out[green] = GREEN
    return out


class PhaseFilter:
    """Causal smoothing: a new phase is accepted after `hold` consistent readings."""

    def __init__(self, hold: int = 4):
        self.hold = hold
        self.state = UNKNOWN
        self.cand = UNKNOWN
        self.count = 0

    def update(self, reading: int) -> int:
        if reading == UNKNOWN:
            return self.state
        if reading == self.state:
            self.cand, self.count = reading, 0
            return self.state
        if reading == self.cand:
            self.count += 1
        else:
            self.cand, self.count = reading, 1
        if self.count >= self.hold:
            self.state, self.count = reading, 0
        return self.state


def phase_series(times: np.ndarray, scores: np.ndarray, fps_sample: float) -> np.ndarray:
    """Per-sample phase for Part A (non-causal normalisation, then hysteresis).

    The hysteresis delay (~0.5 s) is removed by shifting the result back.
    """
    raw = classify(scores)
    hold = max(2, int(round(0.5 * fps_sample)))
    f = PhaseFilter(hold)
    ph = np.array([f.update(int(r)) for r in raw], int)
    if len(ph) > hold:
        ph = np.r_[ph[hold:], np.full(hold, ph[-1])]
    return ph


def phase_intervals(times: np.ndarray, phase: np.ndarray, value: int) -> list[tuple[float, float]]:
    """Maximal [start, end) intervals where phase == value."""
    out, start = [], None
    for t, p in zip(times, phase):
        if p == value and start is None:
            start = t
        elif p != value and start is not None:
            out.append((float(start), float(t)))
            start = None
    if start is not None:
        out.append((float(start), float(times[-1])))
    return out


def phase_at(times: np.ndarray, phase: np.ndarray, t) -> np.ndarray:
    if len(times) == 0:
        return np.zeros(np.shape(t), int)
    i = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 1)
    return phase[i]
