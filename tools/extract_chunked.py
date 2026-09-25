"""Dev: restart-safe extraction for slow CPU machines.

Processes a video in 60 s windows (fresh tracker per window, track ids offset
per window), saving each window, then merges them into the same cache file
format as pipeline.extract(). Used only to build the website / sample outputs
in our CPU sandbox; the submission runs pipeline.extract() in one pass.
"""
import json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TRAFFIC_CACHE", str(ROOT / "cache" / "final"))
sys.path.insert(0, str(ROOT))
import numpy as np
from src.traffic.pipeline import _cache_path
from src.traffic.register import estimate_homography, sample_background
from src.traffic.scene import get_scene
from src.traffic.signal import lamp_rois, lamp_scores
from src.traffic.tracking import TrackerConfig, track_video, video_info

WIN = 60.0
for video in sys.argv[1:]:
    final = _cache_path(video)
    if final.exists():
        print("done", video); continue
    parts = ROOT / "cache" / "parts" / Path(video).stem
    parts.mkdir(parents=True, exist_ok=True)
    info = video_info(video)
    hp = parts / "H.json"
    if hp.exists():
        d = json.loads(hp.read_text()); H = np.array(d["H"]); reg = d["reg"]
    else:
        H, reg = estimate_homography(sample_background(video))
        hp.write_text(json.dumps({"H": H.tolist(), "reg": reg}))
    rois = lamp_rois(get_scene(), H)
    fps = info["fps"]; n = info["n_frames"]; win = int(WIN * fps)
    for k, s in enumerate(range(0, n, win)):
        pf = parts / f"{k:03d}.npz"
        if pf.exists():
            continue
        lt, ls = [], []
        t0 = time.time()
        rows, inf = track_video(video, TrackerConfig(), frame_hook=lambda i, t, f: (lt.append(t), ls.append(lamp_scores(f, rois))),
                                start_frame=s, end_frame=min(n, s + win))
        if len(rows):
            rows[:, 2] += (k + 1) * 100000
        np.savez_compressed(pf, rows=rows, lamp_t=np.array(lt), lamp_s=np.array(ls).reshape(-1, 2), stride=inf["stride"])
        print(video, "window", k, rows.shape, round(time.time() - t0), flush=True)
    ps = [np.load(p) for p in sorted(parts.glob("[0-9]*.npz"))]
    rows = np.concatenate([p["rows"] for p in ps if len(p["rows"])])
    lamp_t = np.concatenate([p["lamp_t"] for p in ps]); lamp_s = np.concatenate([p["lamp_s"] for p in ps])
    info["stride"] = int(ps[0]["stride"]); info["last_t"] = float(lamp_t[-1]); info["chunked_sec"] = WIN
    np.savez_compressed(final, rows=rows, H=H, lamp_t=lamp_t, lamp_s=lamp_s, info=json.dumps(info), reg_info=json.dumps(reg))
    print("merged", video, rows.shape, flush=True)
