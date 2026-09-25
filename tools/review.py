"""Dev: contact sheets for rule evidence.  python tools/review.py VIDEO CLASS [max] [--loose]"""
import sys, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from dev_events import run
from src.traffic.events import Params
from src.traffic.register import apply_h

LOOSE = Params(jay_min_dur=0.8, jay_cross_buffer=0.015, jay_kerb_margin=0.004, fty_ped_dist=0.14, fty_min_samples=1,
               fty_u_margin=0.05, fty_ped_moving=0.1, red_grace=0.0, red_tail=0.0, red_platoon=99, stop_margin=0.0,
               stop_min_still=1.0, queue_gap=0.0, cong_min_veh=5, cong_speed=0.15, cong_min_dur=8.0,
               ww_cos=-0.4, ww_coh=0.7, ww_cnt=10, ww_min_dur=1.0, ww_min_dist=0.03)

def sheets(name, cls, mx=40, loose=False, out='/tmp/claude-0/review'):
    ev, ctx, an = run(name, LOOSE if loose else None)
    Hi = np.linalg.inv(an.H)
    cap = cv2.VideoCapture(f'../data/{name}.mp4'); fps = cap.get(5)
    evs = sorted([e for e in ctx.evidence if e[0] == cls], key=lambda e: e[1])[:mx]
    Path(out).mkdir(exist_ok=True)
    for n, (c, s, e, tids) in enumerate(evs):
        tiles = []
        for t in (np.linspace(s, e, 4) if e - s > 0.5 else [s] * 4):
            cap.set(1, int(round(t * fps))); ok, f = cap.read()
            for tr in ctx.tracks:
                k = np.searchsorted(tr.t, t)
                if k >= len(tr.t) or abs(tr.t[k] - t) > 0.2: continue
                b = tr.box[k]; pts = apply_h(Hi, np.array([[b[0], b[1]], [b[2], b[3]]])) * [1280, 720]
                hot = tr.tid in tids
                col = (0, 0, 255) if hot else ((255, 255, 0) if tr.is_person else (0, 200, 0))
                cv2.rectangle(f, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)), col, 3 if hot else 1)
            cv2.putText(f, f'{name} {cls} #{n} t={t:.1f} [{s:.1f},{e:.1f}]', (10, 30), 0, 0.8, (0, 255, 255), 2)
            tiles.append(cv2.resize(f, (640, 360)))
        cv2.imwrite(f'{out}/{name}_{cls}_{n:02d}.jpg', np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    return evs

if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    evs = sheets(a[0], a[1], int(a[2]) if len(a) > 2 else 40, '--loose' in sys.argv)
    for n, e in enumerate(evs): print(n, e[0], round(e[1], 1), round(e[2], 1), e[3])
