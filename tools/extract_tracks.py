"""Cache tracks for a video: python tools/extract_tracks.py VIDEO OUT.npz"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.traffic.tracking import TrackerConfig, track_video

video, out = sys.argv[1], sys.argv[2]
t0 = time.time()
rows, info = track_video(video, TrackerConfig(), progress=True)
info["sec"] = time.time() - t0
np.savez_compressed(out, rows=rows, info=json.dumps(info))
print(out, rows.shape, info)
