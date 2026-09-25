"""Rebuild the flow field / learned road mask from the final track caches: python tools/calibrate_final.py"""
import glob, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from src.traffic.calibrate import build
from dev_events import load_ex
from src.traffic.pipeline import analyze_extracted
sets = []
for p in sorted(glob.glob(str(ROOT / "cache" / "final" / "*.npz"))):
    name = Path(p).stem.split("_")[0]
    sets.append(analyze_extracted(load_ex(name)).ctx.tracks)
    print(name, len(sets[-1]), "tracks")
m, u, c, n = build(sets)
print("flow cells with support", int((n > 25).sum()))
