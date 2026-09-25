"""Dev helper: signal readings for a video -> npz (t, reading)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np
from src.traffic.scene import get_scene
from src.traffic.signal import read_lamps, lamp_rois
from src.traffic.register import sample_background, estimate_homography

video, out = sys.argv[1], sys.argv[2]
sc = get_scene(); H,_ = estimate_homography(sample_background(video)); rois = lamp_rois(sc, H); cap = cv2.VideoCapture(video); fps = cap.get(5); i = 0; T, R = [], []
stride = max(1, round(fps / 7.5))
while True:
    if i % stride == 0:
        ok, f = cap.read()
        if not ok: break
        T.append(i / fps); R.append(read_lamps(f, rois))
    elif not cap.grab(): break
    i += 1
np.savez(out, t=np.array(T), r=np.array(R))
print(out, len(T), np.bincount(np.array(R) + 1))
