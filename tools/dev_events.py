"""Dev: run the rules on cached final tracks.  python tools/dev_events.py C3905 [C3896 ...]"""
import glob, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TRAFFIC_CACHE", str(ROOT / "cache" / "final"))
sys.path.insert(0, str(ROOT))
import numpy as np
from src.traffic.pipeline import analyze_extracted
from src.traffic.events import Params

def load_ex(name):
    p = glob.glob(str(ROOT / "cache" / "final" / f"{name}_*.npz"))[0]
    d = np.load(p, allow_pickle=True)
    return {k: (json.loads(str(d[k])) if k in ("info", "reg_info") else d[k]) for k in d.files}

def run(name, params=None):
    an = analyze_extracted(load_ex(name), params or Params())
    return an.events, an.ctx, an

if __name__ == '__main__':
    allp = {}
    for n in sys.argv[1:]:
        ev, ctx, an = run(n)
        allp[n + '.mp4'] = {"events": ev, "risk": []}
        from collections import Counter
        print(n, Counter(e[2] for e in ev))
        for e in ev: print('  ', e)
    (ROOT / 'cache' / 'dev_pred.json').write_text(json.dumps({"team": "dev", "videos": allp}, indent=1))
