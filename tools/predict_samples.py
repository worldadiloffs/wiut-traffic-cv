"""Write predictions_samples.json from the cached sample-video runs.

On a GPU, `python run_submission.py --videos samples --out predictions_samples.json`
produces this file directly. Our development sandbox is CPU-only (2 cores), where
YOLO11s at 1280 px runs at ~1.5 analysed frames/s, so the harness time budget would
cut the analysis short. For the published file we therefore ran the same pipeline
restart-safely (tools/extract_chunked.py, 60 s windows) and assemble its output here,
using the harness's own clean_events() so the format is identical.
"""
import glob, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from run_submission import clean_events
from dev_events import load_ex
from src.traffic.pipeline import analyze_extracted
import solution

out = {"team": sys.argv[1] if len(sys.argv) > 1 else "team-105", "videos": {}, "log": {}}
for p in sorted(glob.glob(str(ROOT / "cache" / "final" / "*.npz"))):
    name = Path(p).stem.split("_")[0]
    an = analyze_extracted(load_ex(name))
    ev = [e for e in an.events if e[2] in solution.CLASSES]
    ev, problems = clean_events(ev, solution.CLASSES, an.info["duration"])
    rp = ROOT / "cache" / f"risk_{name}.json"
    risk = json.loads(rp.read_text()) if rp.exists() else []
    out["videos"][f"{name}.mp4"] = {"events": ev, "risk": risk}
    out["log"][f"{name}.mp4"] = {"duration": round(an.info["duration"], 2), "errors": problems,
                                 "note": "CPU sandbox run, see tools/predict_samples.py"}
    print(name, len(ev), "events,", len(risk), "risk samples")
(ROOT / "predictions_samples.json").write_text(json.dumps(out, indent=1))
