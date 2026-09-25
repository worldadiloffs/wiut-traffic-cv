"""Detection + multi-object tracking over a video.

Reads the video once, sequentially, keeping every `stride`-th frame, runs a
YOLO detector with ByteTrack association and returns one row per tracked box.

Rows: frame_idx, t_sec, track_id, cls, conf, x1, y1, x2, y2 (pixels of the
ORIGINAL frame).  Deterministic for a fixed model / stride / imgsz.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights"

# COCO ids we keep: road users + things that can be obstacles on the road.
KEEP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
        15: "cat", 16: "dog", 17: "horse", 19: "cow", 24: "backpack", 28: "suitcase"}
VEHICLES = {2, 3, 5, 7}
TWO_WHEEL = {1, 3}
PERSON = 0

COLS = ["frame", "t", "tid", "cls", "conf", "x1", "y1", "x2", "y2"]


@dataclass
class TrackerConfig:
    model: str = os.environ.get("TRAFFIC_MODEL", "yolo11s.pt")
    imgsz: int = int(os.environ.get("TRAFFIC_IMGSZ", "1280"))
    target_fps: float = float(os.environ.get("TRAFFIC_FPS", "7.5"))
    conf: float = 0.15
    device: str | None = None


def _device(cfg: TrackerConfig) -> str:
    if cfg.device:
        return cfg.device
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


def seed_everything(seed: int = 0) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:  # pragma: no cover
        pass


def video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": float(fps), "n_frames": n, "width": w, "height": h,
            "duration": n / float(fps) if fps else 0.0}


class Tracker:
    """Stateful detector+tracker; feed frames in order."""

    def __init__(self, cfg: TrackerConfig | None = None):
        from ultralytics import YOLO
        seed_everything(0)
        self.cfg = cfg or TrackerConfig()
        weights = WEIGHTS / self.cfg.model
        self.model = YOLO(str(weights if weights.exists() else self.cfg.model))
        self.device = _device(self.cfg)
        self.tracker_yaml = str(Path(__file__).with_name("bytetrack.yaml"))

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Returns array (k, 7): tid, cls, conf, x1, y1, x2, y2."""
        res = self.model.track(frame, imgsz=self.cfg.imgsz, conf=self.cfg.conf,
                               classes=list(KEEP), persist=True, verbose=False,
                               tracker=self.tracker_yaml, device=self.device)[0]
        b = res.boxes
        if b is None or b.id is None or len(b) == 0:
            return np.zeros((0, 7), np.float32)
        ids = b.id.cpu().numpy()
        out = np.column_stack([ids, b.cls.cpu().numpy(), b.conf.cpu().numpy(),
                               b.xyxy.cpu().numpy()]).astype(np.float32)
        return out


class FrameReader:
    """Decodes a video in a background thread and yields every `stride`-th frame.

    Skipped frames are only grabbed (no colour conversion), which is the
    cheapest way to advance an H.264 stream sequentially.
    """

    def __init__(self, path: str, stride: int, queue_size: int = 16):
        import queue
        import threading
        self.cap = cv2.VideoCapture(path)
        self.stride = stride
        self.q: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self.stop = False
        self.n_read = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        idx = 0
        while not self.stop:
            if idx % self.stride == 0:
                ok, frame = self.cap.read()
                if not ok:
                    break
                self.q.put((idx, frame))
            elif not self.cap.grab():
                break
            idx += 1
        self.n_read = idx
        self.q.put(None)
        self.cap.release()

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            yield item

    def close(self):
        self.stop = True
        try:
            while True:
                self.q.get_nowait()
        except Exception:
            pass


def track_video(path: str, cfg: TrackerConfig | None = None, progress: bool = False,
                deadline: float | None = None, frame_hook=None) -> tuple[np.ndarray, dict]:
    """Track a whole file.

    Returns (rows[N, 9], info). `frame_hook(idx, t, frame)` is called on every
    processed frame (used to read the traffic signal). If `deadline`
    (time.perf_counter() value) passes, processing stops early and
    info["truncated_at"] records the last timestamp analysed.
    """
    import time
    cfg = cfg or TrackerConfig()
    info = video_info(path)
    stride = max(1, int(round(info["fps"] / cfg.target_fps)))
    info["stride"] = stride
    tracker = Tracker(cfg)
    reader = FrameReader(path, stride)
    rows = []
    last_t = 0.0
    for idx, frame in reader:
        t = idx / info["fps"]
        last_t = t
        if frame_hook is not None:
            frame_hook(idx, t, frame)
        det = tracker.update(frame)
        if len(det):
            rows.append(np.column_stack([np.full(len(det), idx), np.full(len(det), t), det]))
        if progress and idx % (stride * 200) == 0:
            print(f"  frame {idx}/{info['n_frames']}", flush=True)
        if deadline is not None and time.perf_counter() > deadline:
            info["truncated_at"] = t
            reader.close()
            break
    info["last_t"] = last_t
    arr = np.concatenate(rows) if rows else np.zeros((0, 9), np.float32)
    return arr.astype(np.float64), info
