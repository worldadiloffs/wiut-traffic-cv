"""Build everything the website needs from the sample videos.

    python tools/build_site.py --videos DIR [--skip-video] [--skip-risk]

Writes docs/data/*.json, docs/img/*.jpg, docs/video/*_annotated.mp4 and
docs/data/site.json. Heavy results (tracks) are cached in cache/final/.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TRAFFIC_CACHE", str(ROOT / "cache" / "final"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.traffic.pipeline import analyze_extracted, extract          # noqa: E402
from src.traffic.register import apply_h, sample_background          # noqa: E402
from src.traffic.scene import SCENE_JSON, FLOW_NPZ, get_scene         # noqa: E402
from src.traffic.signal import RED, phase_intervals                   # noqa: E402
import site_content as SC                                             # noqa: E402

DOCS = ROOT / "docs"
REF = ROOT / "src" / "traffic" / "calib" / "reference.jpg"
LIGHT = {"C3896": "sunny midday", "C3897": "sunny midday", "C3902": "late afternoon", "C3905": "dusk"}


def run_risk(video: str, cache: Path) -> list:
    if cache.exists():
        return json.loads(cache.read_text())
    import solution
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    est = solution.RiskEstimator()
    est.reset({"video_id": Path(video).name, "fps": fps, "width": int(cap.get(3)), "height": int(cap.get(4)),
               "n_frames": int(cap.get(7))})
    out, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        s = est.step(f, i / fps)
        if i % 6 == 0:
            out.append([round(i / fps, 2), round(float(s), 3)])
        i += 1
    cache.write_text(json.dumps(out))
    return out


def draw_layout(img: np.ndarray, H_inv=None) -> np.ndarray:
    sc = get_scene()
    h, w = img.shape[:2]
    s = np.array([w, h], float)
    mp = (lambda p: apply_h(H_inv, p)) if H_inv is not None else (lambda p: p)
    ov = img.copy()
    for poly in sc.road_polys:
        cv2.fillPoly(ov, [(mp(poly) * s).astype(np.int32)], (60, 90, 60))
    for poly in sc.islands:
        cv2.fillPoly(ov, [(mp(poly) * s).astype(np.int32)], (60, 60, 120))
    out = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
    for poly in sc.crosswalks.values():
        cv2.polylines(out, [(mp(poly) * s).astype(np.int32)], True, (0, 230, 255), 2, cv2.LINE_AA)
    for name, poly in sc.zones.items():
        pts = (mp(poly) * s).astype(np.int32)
        cv2.polylines(out, [pts], True, (255, 120, 255), 1, cv2.LINE_AA)
        cv2.putText(out, name, tuple(pts.mean(0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 160, 255), 1, cv2.LINE_AA)
    st = (mp(sc.stop_line) * s).astype(np.int32)
    cv2.line(out, tuple(st[0]), tuple(st[1]), (0, 0, 255), 3, cv2.LINE_AA)
    for c, col in ((sc.red_lamp, (0, 0, 255)), (sc.green_lamp, (0, 255, 0))):
        p = (mp(c[None]) * s)[0].astype(int)
        cv2.circle(out, tuple(p), 9, col, 2)
    return out


def heatmap(points: np.ndarray, base: np.ndarray, sigma: float = 6) -> np.ndarray:
    h, w = base.shape[:2]
    acc = np.zeros((h, w), np.float32)
    p = (points * [w, h]).astype(int)
    p = p[(p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)]
    np.add.at(acc, (p[:, 1], p[:, 0]), 1)
    acc = cv2.GaussianBlur(acc, (0, 0), sigma)
    acc = np.sqrt(acc / (acc.max() + 1e-9))
    col = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    a = np.clip(acc * 1.6, 0, 1)[..., None]
    return (base * (1 - a) * 0.8 + col * a).astype(np.uint8)


def examples(analyses: dict, videos: dict, per_class: int = 2) -> list:
    out = []
    (DOCS / "img" / "ex").mkdir(parents=True, exist_ok=True)
    for cls in [r["id"] for r in SC.RULES]:
        picks = []
        for name, an in analyses.items():
            for c, a, b, tids in an.ctx.evidence:
                if c == cls:
                    picks.append((b - a, name, a, b, tids))
        picks.sort(key=lambda x: -x[0])
        seen = set()
        for dur, name, a, b, tids in picks:
            if len([p for p in out if p["cls"] == cls]) >= per_class:
                break
            if (name, round(a)) in seen:
                continue
            seen.add((name, round(a)))
            an = analyses[name]
            t = a + min(1.5, (b - a) / 2)
            img = frame_with_boxes(videos[name], an, t, set(tids), cls)
            fn = f"img/ex/{Path(name).stem}_{cls}_{int(a * 10)}.jpg"
            cv2.imwrite(str(DOCS / fn), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
            out.append({"cls": cls, "video": name, "t": round(a, 1), "img": fn,
                        "note": f"{b - a:.1f} s segment"})
    return out


def frame_with_boxes(video: str, an, t: float, tids: set, cls: str, width: int = 640) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, f = cap.read()
    cap.release()
    W, H = an.info["width"], an.info["height"]
    rows = an.rows
    k = np.argmin(np.abs(rows[:, 1] - t))
    fr = rows[np.abs(rows[:, 1] - rows[k, 1]) < 1e-6]
    for r in fr:
        x1, y1, x2, y2 = r[5:9].astype(int)
        hot = int(r[2]) in tids
        cv2.rectangle(f, (x1, y1), (x2, y2), (0, 0, 255) if hot else (80, 200, 80), 4 if hot else 1, cv2.LINE_AA)
    # crop around the highlighted objects
    hot_rows = fr[np.isin(fr[:, 2].astype(int), list(tids))]
    if len(hot_rows):
        cx = (hot_rows[:, 5].min() + hot_rows[:, 7].max()) / 2
        cy = (hot_rows[:, 6].min() + hot_rows[:, 8].max()) / 2
        cw, ch = W * 0.5, H * 0.5
        x0 = int(np.clip(cx - cw / 2, 0, W - cw))
        y0 = int(np.clip(cy - ch / 2, 0, H - ch))
        f = f[y0:y0 + int(ch), x0:x0 + int(cw)]
    return cv2.resize(f, (width, int(width * f.shape[0] / f.shape[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--skip-risk", action="store_true")
    ap.add_argument("--dev", default=str(ROOT / "dev" / "dev_eval.json"))
    args = ap.parse_args()
    for d in ("data", "img", "video"):
        (DOCS / d).mkdir(parents=True, exist_ok=True)
    vids = {p.name: str(p) for p in sorted(Path(args.videos).glob("*.mp4"))}
    analyses, metas = {}, []
    import render
    for name, path in vids.items():
        print("==", name, flush=True)
        an = analyze_extracted(extract(path))
        analyses[name] = an
        risk = [] if args.skip_risk else run_risk(path, ROOT / "cache" / f"risk_{Path(name).stem}.json")
        summ = render.summarise(an, name)
        summ["risk"] = risk
        (DOCS / "data" / f"{Path(name).stem}.json").write_text(json.dumps(summ))
        if not args.skip_video:
            render.render(path, str(DOCS / "video"), 960, risk)
            (DOCS / "video" / f"{Path(name).stem}.json").unlink(missing_ok=True)
        reds = phase_intervals(an.lamp_t, an.phase, RED)
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        metas.append({
            "name": name, "stem": Path(name).stem, "duration": an.info["duration"], "fps": an.info["fps"],
            "source": "3840×2160", "lighting": LIGHT.get(Path(name).stem, "?"),
            "shift": float(np.abs(apply_h(an.H, corners) - corners).max()),
            "people": sum(1 for t in an.ctx.tracks if t.is_person),
            "vehicles": sum(1 for t in an.ctx.tracks if t.is_vehicle),
            "cycles": len(reds), "annotated": f"video/{Path(name).stem}_annotated.mp4",
        })

    # ---- images
    ref = cv2.imread(str(REF))
    cv2.imwrite(str(DOCS / "img" / "reference.jpg"), draw_layout(ref), [cv2.IMWRITE_JPEG_QUALITY, 85])
    day = [n for n in vids if "3896" in n] or list(vids)
    bg = sample_background(vids[day[0]])
    an = analyses[day[0]]
    S = np.diag([1280, 720, 1.0])
    warped = cv2.warpPerspective(cv2.resize(bg, (1280, 720)), S @ an.H @ np.linalg.inv(S), (1280, 720))
    cv2.imwrite(str(DOCS / "img" / "registration.jpg"), cv2.addWeighted(warped, 0.5, ref, 0.5, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    veh, ppl = [], []
    for an in analyses.values():
        for tr in an.ctx.tracks:
            if tr.is_vehicle:
                veh.append(tr.gp_s[tr.speed_rel > 0.35])
            elif tr.is_person:
                ppl.append(tr.gp_s)
    cv2.imwrite(str(DOCS / "img" / "heat_vehicles.jpg"), heatmap(np.concatenate(veh), ref), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(DOCS / "img" / "heat_people.jpg"), heatmap(np.concatenate(ppl), ref, 4), [cv2.IMWRITE_JPEG_QUALITY, 85])
    traj = (ref * 0.55).astype(np.uint8)
    an0 = analyses[list(analyses)[0]]
    for tr in an0.ctx.tracks:
        if not tr.is_vehicle or len(tr.t) < 8:
            continue
        p = (tr.gp_s * [1280, 720]).astype(np.int32)
        d = p[-1] - p[0]
        if np.hypot(*d) < 30:
            continue
        hue = (np.arctan2(d[1], d[0]) / (2 * np.pi)) % 1
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        cv2.polylines(traj, [p], False, (int(b * 255), int(g * 255), int(r * 255)), 1, cv2.LINE_AA)
    cv2.imwrite(str(DOCS / "img" / "trajectories.jpg"), traj, [cv2.IMWRITE_JPEG_QUALITY, 85])
    fl = np.load(FLOW_NPZ)
    fimg = (ref * 0.5).astype(np.uint8)
    gh, gw = fl["coh"].shape
    for y in range(gh):
        for x in range(gw):
            if fl["cnt"][y, x] > 25 and fl["coh"][y, x] > 0.85:
                c = np.array([(x + .5) * 1280 / gw, (y + .5) * 720 / gh])
                v = fl["vec"][y, x]
                hue = (np.arctan2(v[1], v[0]) / (2 * np.pi)) % 1
                r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
                cv2.arrowedLine(fimg, tuple(c.astype(int)), tuple((c + v * 14).astype(int)),
                                (int(b * 255), int(g * 255), int(r * 255)), 1, cv2.LINE_AA, tipLength=0.4)
    cv2.imwrite(str(DOCS / "img" / "flow.jpg"), fimg, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # ---- demo assets
    (DOCS / "data" / "scene.json").write_text(SCENE_JSON.read_text())
    (DOCS / "data" / "flow.json").write_text(json.dumps({
        "w": int(gw), "h": int(gh), "vec": fl["vec"].reshape(-1, 2).round(3).tolist(),
        "coh": fl["coh"].reshape(-1).round(3).tolist(), "cnt": fl["cnt"].reshape(-1).astype(int).tolist()}))

    # ---- site.json
    ex = examples(analyses, vids)
    dev = json.loads(Path(args.dev).read_text()) if Path(args.dev).exists() else {"intro": "", "header": [], "rows": []}
    failures = json.loads((ROOT / "dev" / "failures.json").read_text()) if (ROOT / "dev" / "failures.json").exists() else []
    team = json.loads((ROOT / "dev" / "team.json").read_text()) if (ROOT / "dev" / "team.json").exists() else []
    links = json.loads((ROOT / "dev" / "links.json").read_text()) if (ROOT / "dev" / "links.json").exists() else []
    tot = sum(m["duration"] for m in metas)
    n_ev = sum(len(a.events) for a in analyses.values())
    site = {
        "videos": metas,
        "stats": [[f"{len(metas)}", "sample clips, 4K"], [f"{tot / 60:.1f} min", "of footage analysed"],
                  [f"{sum(m['people'] + m['vehicles'] for m in metas):,}", "tracks"],
                  [f"{n_ev}", "events on the samples"], ["7", "event classes"], [f"{dev.get('score_a', 0):.3f}", "Score A on our own dev labels"]],
        "rules": SC.RULES, "findings": SC.FINDINGS, "report": SC.REPORT, "ablations": SC.ABLATIONS,
        "signal_note": "red = boulevard stopped, green = boulevard moving. The cycle is stable at ≈ 77–80 s.",
        "examples": ex, "failures": failures, "dev": dev, "team": team, "links": links,
        "demo_sample": "video/demo_sample.mp4",
    }
    (DOCS / "data" / "site.json").write_text(json.dumps(site, indent=1, ensure_ascii=False))
    print("site.json written:", n_ev, "events")


if __name__ == "__main__":
    main()
