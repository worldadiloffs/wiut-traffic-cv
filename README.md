# Junction Watch — traffic events and accident risk from a fixed road camera

WIUT Hackathon 2026 · Computer Vision track · elimination task.

**Website (approach, EDA, results, live demo, report):** https://worldadiloffs.github.io/wiut-traffic-cv/

Given a video from the organisers' fixed CCTV camera, `solution.py` returns

* **Part A** — every traffic event as `[start_sec, end_sec, label]`;
* **Part B** — a causal per-frame probability that an accident starts within 5 s.

---

## Run it

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

* No internet is needed at run time. The weights are in `weights/` (`yolo11s.pt` 19 MB for Part A,
  `yolo11n.pt` 5.6 MB for Part B, total ≈ 25 MB). `weights/download.sh` re-fetches them from the
  official Ultralytics release if the folder was stripped.
* GPU is used automatically when `torch.cuda.is_available()`; otherwise everything runs on CPU.
* `run_submission.py` and `evaluate.py` are the organisers' files, unchanged.
* `predictions_samples.json` is our output on the four sample clips.

Optional environment variables (defaults are what we submit):

| variable | default | meaning |
|---|---|---|
| `TRAFFIC_MODEL` | `yolo11s.pt` | Part A detector |
| `TRAFFIC_IMGSZ` | `1280` | Part A inference size |
| `TRAFFIC_FPS` | `7.5` | frames analysed per second in Part A |
| `TRAFFIC_RISK_MODEL` / `_IMGSZ` / `_FPS` | `yolo11n.pt` / `960` / `5` | Part B |
| `TRAFFIC_PART_A_BUDGET` | `1.8` | Part A stops analysing new frames after 1.8 × video duration |
| `TRAFFIC_CACHE` | unset | folder to cache tracks between runs (development only) |

## Approach

```
video ─► registration (SIFT + RANSAC homography to the canonical view)
      ─► one sequential pass at 7.5 fps: YOLO11s (1280 px) + ByteTrack │ signal-lamp colour scores
      ─► tracks in canonical coordinates (ground point = box bottom-centre, speed in box-heights / s)
      ─► 7 rule-based detectors ─► merged segments per class
```

**Learned:** the detector (YOLO11s / YOLO11n, COCO weights, not fine-tuned), the ByteTrack association,
a flow field of the dominant travel direction per cell (from the tracks of the sample clips, used for
wrong-way), and a per-clip normalisation of the lamp colour scores.

**Rule-based:** view registration, the scene layout (`src/traffic/scene.json`: three crossings, the stop
line, carriageway polygons, refuge islands, zones, signal-lamp position — drawn once on the canonical
frame), the signal phase, and all event rules (`src/traffic/events.py`).

| class | rule (short) |
|---|---|
| `jaywalking` | walking person on the carriageway, away from kerbs, crossings and the bus bay, ≥ 1.2 s |
| `failure_to_yield` | vehicle passes through a crossing while a moving pedestrian is on it nearby |
| `red_light` | vehicle crosses the near-arm stop line during red and enters the junction (platoons ignored) |
| `stop_line` | vehicle stationary past the stop line during red; ends when the signal turns green |
| `stopped_vehicle` | stationary ≥ 10 s on the carriageway, not in a signal queue / at a crossing / queued |
| `congestion` | ≥ 6 near-stationary vehicles in a direction for ≥ 12 s (approach: only during green) |
| `wrong_way` | heading opposite to the learned flow field for ≥ 1.5 s outside the junction box |

Classes we do not predict — `accident`, `near_miss`, `illegal_u_turn`, `illegal_turn`,
`solid_line_crossing`, `road_obstacle`, `fire_smoke` — are omitted on purpose: none occur in the samples,
and the metric adds every predicted class to the average, so an unverified class can only lower the score.
They stay in `CLASSES` (the list may only shrink) and are filtered at the end of `detect_events`.

**Part B** (`src/traffic/risk.py`) never opens the file. Every ~0.2 s it runs YOLO11n + ByteTrack on the
frame it was given, estimates short causal velocities, and for every pair of road users computes the time
and distance of closest approach under constant velocity. Pairs on a collision course within 3.5 s
contribute `closeness² · exp(−t/1.5) · closing-speed`, pairs must persist for ≥ 2 evaluations, and the
max is smoothed (fast rise, slow decay). On normal traffic in the samples the score stays below 0.13.

### Time budget

Measured per video (4K, 29.97 fps) on the evaluation profile: decoding runs in a background thread while
the detector runs; Part A stops taking new frames at 1.8 × duration (`TRAFFIC_PART_A_BUDGET`) and
returns events from what it has seen; Part B stops running the detector if the video as a whole passes
2.7 × duration. On a T4 both parts together take well under the 3 × budget.

### Determinism

`seed_everything(0)` fixes Python / NumPy / Torch seeds, cuDNN runs in deterministic mode, RANSAC uses
`cv2.setRNGSeed(0)`, frames are read sequentially (no random sampling). Two runs on the same machine give
the same `predictions.json`.

## Repository layout

```
solution.py              interface for the harness (CLASSES, detect_events, RiskEstimator)
run_submission.py        organisers' harness (unchanged)
evaluate.py              organisers' metric (unchanged)
requirements.txt
weights/                 yolo11s.pt, yolo11n.pt (+ download.sh)
src/traffic/
  tracking.py            threaded frame reader, YOLO + ByteTrack
  register.py            SIFT/RANSAC registration to the canonical view
  scene.py, scene.json   layout (normalised coordinates)
  signal.py              lamp colour scores -> red / green phase
  tracks.py              trajectories, smoothing, speeds, rider filter
  events.py              the rules
  risk.py                Part B
  pipeline.py            Part A orchestration + time budget
  calibrate.py           builds calib/flow.npz and road_mask.png from sample tracks
  calib/                 reference.jpg (canonical frame), flow.npz, road_mask.png
tools/                   extraction cache, rendering, website builder, dev evaluation
dev/                     our own labels of the sample clips + evaluation output
docs/                    the website (GitHub Pages), incl. the in-browser demo
predictions_samples.json our output on the sample clips
```

Reproduce the sample predictions and the website data:

```bash
python run_submission.py --videos samples --out predictions_samples.json --team junction-watch
python evaluate.py --pred predictions_samples.json --gt dev/dev_labels.json --per-video
python -m src.traffic.calibrate cache/final/*.npz      # rebuild the flow field (optional)
python tools/build_site.py --videos samples            # website data, annotated videos
```

## Data, models and licences

| item | source | licence |
|---|---|---|
| sample videos | organisers (WIUT Hackathon 2026) | organisers' terms, not redistributed |
| YOLO11s / YOLO11n weights | Ultralytics, trained on COCO | AGPL-3.0 |
| ByteTrack implementation | Ultralytics | AGPL-3.0 |
| ONNX Runtime Web (demo) | Microsoft | MIT |

No external dataset was used for training; nothing was fine-tuned. Our own labels of the sample clips
(`dev/dev_labels.json`) were made by us for evaluation only.

## Team

See the website's Team section for roles, contributions and links.
