"""Track table -> per-track trajectories with smoothed ground points and speeds.

All positions are normalised to the frame size. The ground point of a box is
its bottom-centre. Speeds are measured in *box heights per second* where a
perspective-robust quantity is needed (stationarity), and in normalised image
units per second for directions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tracking import PERSON, TWO_WHEEL, VEHICLES


@dataclass
class Track:
    tid: int
    cls: int
    t: np.ndarray           # (n,) seconds
    box: np.ndarray         # (n, 4) normalised x1 y1 x2 y2
    conf: np.ndarray        # (n,)
    gp: np.ndarray = field(init=False)      # ground point (n, 2)
    vel: np.ndarray = field(init=False)     # (n, 2) normalised units / s
    speed_rel: np.ndarray = field(init=False)  # |v| / box height  (1/s)
    size: np.ndarray = field(init=False)    # box height (n,)

    def __post_init__(self):
        b = self.box
        self.gp = np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]])
        self.size = np.maximum(b[:, 3] - b[:, 1], 1e-3)
        self.gp_s = _smooth(self.gp, self.t, 0.6)
        self.vel = _velocity(self.gp_s, self.t, 1.0)
        self.speed_rel = np.linalg.norm(self.vel, axis=1) / _smooth1(self.size, self.t, 1.0)

    @property
    def is_vehicle(self) -> bool:
        return self.cls in VEHICLES

    @property
    def is_person(self) -> bool:
        return self.cls == PERSON

    @property
    def is_two_wheeler(self) -> bool:
        return self.cls in TWO_WHEEL

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    def at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))


def _smooth(x: np.ndarray, t: np.ndarray, win: float) -> np.ndarray:
    """Centred moving average over a time window (seconds)."""
    if len(t) < 3:
        return x.copy()
    cs = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    lo = np.searchsorted(t, t - win / 2)
    hi = np.searchsorted(t, t + win / 2, side="right")
    return (cs[hi] - cs[lo]) / (hi - lo)[:, None]


def _smooth1(x: np.ndarray, t: np.ndarray, win: float) -> np.ndarray:
    return _smooth(x[:, None], t, win)[:, 0]


def _velocity(p: np.ndarray, t: np.ndarray, win: float) -> np.ndarray:
    if len(t) < 2:
        return np.zeros_like(p)
    lo = np.searchsorted(t, t - win / 2)
    hi = np.clip(np.searchsorted(t, t + win / 2, side="right") - 1, 0, len(t) - 1)
    dt = np.maximum(t[hi] - t[lo], 1e-3)
    v = (p[hi] - p[lo]) / dt[:, None]
    v[hi == lo] = 0
    return v


def build_tracks(rows: np.ndarray, width: int, height: int, min_len: int = 3) -> list[Track]:
    """rows: frame, t, tid, cls, conf, x1, y1, x2, y2 (pixels)."""
    tracks = []
    if len(rows) == 0:
        return tracks
    rows = rows[np.lexsort((rows[:, 1], rows[:, 2]))]
    tids, starts = np.unique(rows[:, 2], return_index=True)
    ends = np.r_[starts[1:], len(rows)]
    scale = np.array([width, height, width, height], float)
    for tid, s, e in zip(tids, starts, ends):
        r = rows[s:e]
        if len(r) < min_len:
            continue
        # majority class weighted by confidence
        cls_ids = r[:, 3].astype(int)
        w = np.bincount(cls_ids, weights=r[:, 4])
        cls = int(np.argmax(w))
        # drop duplicate timestamps
        _, keep = np.unique(r[:, 1], return_index=True)
        r = r[keep]
        tracks.append(Track(int(tid), cls, r[:, 1].copy(), r[:, 5:9] / scale, r[:, 4].copy()))
    return tracks


def rider_person_ids(tracks: list[Track], tol: float = 0.35, classes=(1, 3)) -> set[int]:
    """Persons that ride (or push) a two-wheeler of `classes` for most of their track."""
    by_time: dict[float, list[np.ndarray]] = {}
    for tr in tracks:
        if tr.cls in classes:
            for t, b in zip(tr.t, tr.box):
                by_time.setdefault(round(float(t), 3), []).append(b)
    riders = set()
    if not by_time:
        return riders
    for p in tracks:
        if not p.is_person:
            continue
        hits = 0
        for t, pb in zip(p.t, p.box):
            wb = by_time.get(round(float(t), 3))
            if not wb:
                continue
            wb = np.asarray(wb)
            ix = np.clip(np.minimum(pb[2], wb[:, 2]) - np.maximum(pb[0], wb[:, 0]), 0, None)
            iy = np.clip(np.minimum(pb[3], wb[:, 3]) - np.maximum(pb[1], wb[:, 1]), 0, None)
            if (ix * iy).max() > tol * (pb[2] - pb[0]) * (pb[3] - pb[1]):
                hits += 1
        if hits > 0.4 * len(p.t):
            riders.add(p.tid)
    return riders
