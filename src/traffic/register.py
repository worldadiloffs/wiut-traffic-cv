"""Register a video's view to the canonical reference view.

The organisers' clips come from the same camera and place, but the framing
changes slightly between clips (tripod re-positioned, zoom). The scene layout
is drawn once on a canonical reference frame; for every new video we estimate
a homography H (video -> canonical) from SIFT matches between a background
image of the video (per-pixel median of a few frames) and the reference.
All ground points are mapped into canonical coordinates before the rules run,
and the signal-lamp ROI is mapped back into video pixels.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "calib" / "reference.jpg"
WORK_W, WORK_H = 1280, 720


def background_from_frames(frames: list[np.ndarray]) -> np.ndarray:
    small = [cv2.resize(f, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA) for f in frames]
    return np.median(np.stack(small), axis=0).astype(np.uint8)


def sample_background(path: str, n: int = 15, span_sec: float = 60.0) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last = min(total - 1, int(span_sec * fps))
    idxs = np.linspace(0, max(0, last), n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return background_from_frames(frames)


def _mask_static(h: int, w: int) -> np.ndarray:
    """Ignore the very top (sky/trees move) - keep the rest."""
    m = np.full((h, w), 255, np.uint8)
    return m


def estimate_homography(img: np.ndarray, ref: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Returns H mapping normalised video coords -> normalised canonical coords."""
    if ref is None:
        ref = cv2.imread(str(REFERENCE))
    a = cv2.cvtColor(cv2.resize(img, (WORK_W, WORK_H)), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(ref, (WORK_W, WORK_H)), cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    a, b = clahe.apply(a), clahe.apply(b)
    sift = cv2.SIFT_create(nfeatures=6000)
    ka, da = sift.detectAndCompute(a, _mask_static(*a.shape))
    kb, db = sift.detectAndCompute(b, _mask_static(*b.shape))
    info = {"kp_video": len(ka), "kp_ref": len(kb)}
    S = np.diag([1 / WORK_W, 1 / WORK_H, 1.0])
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        info["status"] = "identity (too few features)"
        return np.eye(3), info
    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
    knn = matcher.knnMatch(da, db, k=2)
    good = [m for m, n2 in (p for p in knn if len(p) == 2) if m.distance < 0.72 * n2.distance]
    info["matches"] = len(good)
    if len(good) < 25:
        info["status"] = "identity (few matches)"
        return np.eye(3), info
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(0)
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0, maxIters=5000, confidence=0.999)
    if H is None or inl is None or inl.sum() < 20:
        info["status"] = "identity (ransac failed)"
        return np.eye(3), info
    info["inliers"] = int(inl.sum())
    Hn = S @ H @ np.linalg.inv(S)
    # sanity: the frame corners must not move by more than 35% of the frame
    c = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    moved = np.abs(apply_h(Hn, c) - c).max()
    info["max_corner_shift"] = float(moved)
    if moved > 0.35:
        info["status"] = "identity (implausible warp)"
        return np.eye(3), info
    info["status"] = "ok"
    return Hn, info


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return p[:, :2] / p[:, 2:3]
