"""Learn scene priors from unlabeled sample videos (run once, offline).

* road_mask.png - where moving vehicles drive (the carriageway incl. crossings),
  built from ground points of moving vehicle tracks in canonical coordinates.
* flow.npz      - dominant direction of travel per grid cell, its coherence
  (mean resultant length) and support; used for wrong-way detection.

Usage: python -m src.traffic.calibrate cache/*.npz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .scene import FLOW_NPZ, ROAD_MASK
from .tracks import build_tracks

MASK_W, MASK_H = 320, 180
FLOW_W, FLOW_H = 64, 36


def accumulate(track_sets):
    occ = np.zeros((MASK_H, MASK_W), np.float64)
    vec = np.zeros((FLOW_H, FLOW_W, 2), np.float64)
    cnt = np.zeros((FLOW_H, FLOW_W), np.float64)
    for tracks in track_sets:
        for tr in tracks:
            if not (tr.is_vehicle or tr.cls == 1):
                continue
            moving = tr.speed_rel > 0.4
            if moving.sum() < 3:
                continue
            gp = tr.gp_s[moving]
            xi = np.clip((gp[:, 0] * MASK_W).astype(int), 0, MASK_W - 1)
            yi = np.clip((gp[:, 1] * MASK_H).astype(int), 0, MASK_H - 1)
            np.add.at(occ, (yi, xi), 1.0)
            v = tr.vel[moving]
            n = np.linalg.norm(v, axis=1, keepdims=True)
            u = v / np.maximum(n, 1e-9)
            fx = np.clip((gp[:, 0] * FLOW_W).astype(int), 0, FLOW_W - 1)
            fy = np.clip((gp[:, 1] * FLOW_H).astype(int), 0, FLOW_H - 1)
            np.add.at(vec, (fy, fx), u)
            np.add.at(cnt, (fy, fx), 1.0)
    return occ, vec, cnt


def build(track_sets, out_mask=ROAD_MASK, out_flow=FLOW_NPZ):
    occ, vec, cnt = accumulate(track_sets)
    occ = cv2.GaussianBlur(occ, (0, 0), 1.2)
    mask = (occ > 0.6).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    # fill small holes
    inv = cv2.bitwise_not(mask)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 60:
            mask[lab == i] = 255
    # vehicle ground points sit slightly outside the true kerb line -> erode a little
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_mask), mask)
    mean = vec / np.maximum(cnt[..., None], 1)
    coh = np.linalg.norm(mean, axis=2)
    unit = mean / np.maximum(coh[..., None], 1e-9)
    np.savez_compressed(out_flow, vec=unit.astype(np.float32), coh=coh.astype(np.float32),
                        cnt=cnt.astype(np.float32))
    return mask, unit, coh, cnt


def load_cached(npz_path: str):
    from .register import apply_h
    d = np.load(npz_path, allow_pickle=True)
    info = json.loads(str(d["info"]))
    rows = d["rows"].copy()
    hp = Path(npz_path.replace(".npz", "_H.npy"))
    H = np.load(hp) if hp.exists() else np.eye(3)
    rows = to_canonical(rows, info["width"], info["height"], H)
    return build_tracks(rows, info["width"], info["height"]), info


def to_canonical(rows: np.ndarray, width: int, height: int, H: np.ndarray) -> np.ndarray:
    """Map box corners (pixels, video) to canonical view (pixels of the same size)."""
    from .register import apply_h
    if len(rows) == 0 or np.allclose(H, np.eye(3)):
        return rows
    s = np.array([width, height], float)
    p1 = apply_h(H, rows[:, 5:7] / s) * s
    p2 = apply_h(H, rows[:, 7:9] / s) * s
    p3 = apply_h(H, np.column_stack([rows[:, 5], rows[:, 8]]) / s) * s
    p4 = apply_h(H, np.column_stack([rows[:, 7], rows[:, 6]]) / s) * s
    xs = np.column_stack([p1[:, 0], p2[:, 0], p3[:, 0], p4[:, 0]])
    ys = np.column_stack([p1[:, 1], p2[:, 1], p3[:, 1], p4[:, 1]])
    out = rows.copy()
    out[:, 5], out[:, 6] = xs.min(1), ys.min(1)
    out[:, 7], out[:, 8] = xs.max(1), ys.max(1)
    return out


if __name__ == "__main__":
    sets = [load_cached(p)[0] for p in sys.argv[1:]]
    m, u, c, n = build(sets)
    print("road pixels", int((m > 0).sum()), "flow cells with support", int((n > 20).sum()))
