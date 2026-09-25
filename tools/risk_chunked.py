"""Dev: Part B risk curves for sample videos, restart-safe (60 s windows, fresh
estimator per window; the first second of each window starts from zero state)."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import solution

for video in sys.argv[1:]:
    name = Path(video).stem
    final = ROOT / "cache" / f"risk_{name}.json"
    if final.exists():
        continue
    d = ROOT / "cache" / "risk_parts" / name; d.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video); fps = cap.get(5); n = int(cap.get(7)); W = int(cap.get(3)); H = int(cap.get(4)); cap.release()
    win = int(60 * fps)
    for k, s in enumerate(range(0, n, win)):
        pf = d / f"{k:03d}.json"
        if pf.exists():
            continue
        est = solution.RiskEstimator()
        est.reset({"video_id": name, "fps": fps, "width": W, "height": H, "n_frames": n})
        cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        cur = []
        for i in range(s, min(n, s + win)):
            ok, f = cap.read()
            if not ok:
                break
            cur.append([round(i / fps, 4), round(float(est.step(f, i / fps)), 4)])
        pf.write_text(json.dumps(cur)); print(name, k, len(cur), flush=True)
    curve = [x for p in sorted(d.glob("*.json")) for x in json.loads(p.read_text())]
    final.write_text(json.dumps(curve)); print("merged", name, len(curve))
