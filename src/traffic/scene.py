"""Scene layout for the fixed camera: polygons, stop line, signal ROI, learned masks.

Everything is stored in normalised coordinates (0..1) so the same layout
applies to the 4K originals and to any resized copy.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SCENE_JSON = HERE / "scene.json"
ROAD_MASK = HERE / "calib" / "road_mask.png"
FLOW_NPZ = HERE / "calib" / "flow.npz"


def points_in_poly(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd point-in-polygon. pts (N,2), poly (M,2) -> bool (N,)."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    xi, yi = poly[:, 0][None, :], poly[:, 1][None, :]
    xj, yj = np.roll(poly[:, 0], 1)[None, :], np.roll(poly[:, 1], 1)[None, :]
    cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
    return (cond.sum(1) % 2) == 1


def _edge_dist(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    ap = pts[:, None, :] - a[None]
    t = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-12)[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - proj, axis=-1).min(1)


class Scene:
    def __init__(self, path: Path = SCENE_JSON):
        cfg = json.loads(Path(path).read_text())
        w, h = cfg["reference_size"]
        norm = lambda pts: np.asarray(pts, float) / np.array([w, h], float)
        self.crosswalks = {k: norm(v) for k, v in cfg["crosswalks"].items()}
        self.stop_line = norm(cfg["stop_line_A"])
        self.zones = {k: norm(v) for k, v in cfg["zones"].items()}
        sig = cfg["signal"]
        self.red_lamp = norm([sig["red_lamp"]])[0]
        self.green_lamp = norm([sig["green_lamp"]])[0]
        self.lamp_radius = sig["radius"] / w
        self.road_polys = [norm(v) for k, v in cfg.get("road", {}).items() if not k.startswith("_")]
        self.islands = [norm(v) for k, v in cfg.get("islands", {}).items() if not k.startswith("_")]
        self.road = cv2.imread(str(ROAD_MASK), cv2.IMREAD_GRAYSCALE) if ROAD_MASK.exists() else None
        self.flow = dict(np.load(FLOW_NPZ)) if FLOW_NPZ.exists() else None

    # ---- geometry helpers -------------------------------------------------
    def in_zone(self, pts: np.ndarray, name: str) -> np.ndarray:
        return points_in_poly(pts, self.zones[name])

    def in_crosswalk(self, pts: np.ndarray, name: str | None = None, pad: float = 0.0) -> np.ndarray:
        names = [name] if name else list(self.crosswalks)
        out = np.zeros(len(np.asarray(pts).reshape(-1, 2)), bool)
        for n in names:
            poly = self.crosswalks[n]
            if pad:
                c = poly.mean(0)
                poly = c + (poly - c) * (1 + pad)
            out |= points_in_poly(pts, poly)
        return out

    def crosswalk_u(self, pts: np.ndarray, name: str) -> np.ndarray:
        """Position along the crossing (0 = one kerb, 1 = the other). Polygons are
        stored so that edges (p0,p1) and (p2,p3) are the kerb ends."""
        p = self.crosswalks[name]
        a, b = (p[0] + p[1]) / 2, (p[2] + p[3]) / 2
        d = b - a
        pts = np.asarray(pts, float).reshape(-1, 2)
        return ((pts - a) @ d) / (d @ d)

    def crosswalk_of(self, pts: np.ndarray) -> np.ndarray:
        """Index of the crosswalk containing each point, -1 if none."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        out = np.full(len(pts), -1)
        for i, poly in enumerate(self.crosswalks.values()):
            out[(out < 0) & points_in_poly(pts, poly)] = i
        return out

    def on_road(self, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """True on the hand-drawn carriageway (minus refuge islands).

        With margin > 0 a point must also be at least `margin` away from the
        edge of the road polygon containing it (people standing on the kerb)."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        if self.road_polys:
            on = np.zeros(len(pts), bool)
            for poly in self.road_polys:
                inside = points_in_poly(pts, poly)
                if margin > 0 and inside.any():
                    inside[inside] = _edge_dist(pts[inside], poly) > margin
                on |= inside
            for poly in self.islands:
                on &= ~points_in_poly(pts, poly)
            return on
        return self.on_learned_road(pts)

    def on_learned_road(self, pts: np.ndarray) -> np.ndarray:
        """True where the learned mask (moving-vehicle footprints) is set."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        if self.road is None:
            return np.ones(len(pts), bool)
        h, w = self.road.shape
        xi = np.clip((pts[:, 0] * w).astype(int), 0, w - 1)
        yi = np.clip((pts[:, 1] * h).astype(int), 0, h - 1)
        return self.road[yi, xi] > 127

    def side_of_stop_line(self, pts: np.ndarray) -> np.ndarray:
        """>0 upstream (approach side), <0 past the stop line."""
        (x1, y1), (x2, y2) = self.stop_line
        pts = np.asarray(pts, float).reshape(-1, 2)
        # approach side is up-left of the line (smaller y)
        s = (x2 - x1) * (pts[:, 1] - y1) - (y2 - y1) * (pts[:, 0] - x1)
        return -s

    def flow_at(self, pts: np.ndarray):
        """Dominant motion direction (unit vec), coherence and support at each point."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        if self.flow is None:
            n = len(pts)
            return np.zeros((n, 2)), np.zeros(n), np.zeros(n)
        vec, coh, cnt = self.flow["vec"], self.flow["coh"], self.flow["cnt"]
        gh, gw = coh.shape
        xi = np.clip((pts[:, 0] * gw).astype(int), 0, gw - 1)
        yi = np.clip((pts[:, 1] * gh).astype(int), 0, gh - 1)
        return vec[yi, xi], coh[yi, xi], cnt[yi, xi]


@lru_cache(maxsize=1)
def get_scene() -> Scene:
    return Scene()
