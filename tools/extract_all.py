"""Dev: run the heavy extraction (final config) on sample videos into cache/final/."""
import os, sys, time
from pathlib import Path
os.environ.setdefault("TRAFFIC_CACHE", str(Path(__file__).resolve().parents[1] / "cache" / "final"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.traffic.pipeline import extract
for v in sys.argv[1:]:
    t = time.time(); ex = extract(v, progress=True)
    print(v, ex["rows"].shape, ex["reg_info"], round(time.time() - t), flush=True)
