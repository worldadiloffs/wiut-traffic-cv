"""Score predictions against our own dev labels with the official metric and
write the table shown on the website.

    python tools/dev_eval.py predictions_samples.json dev/dev_labels.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate  # noqa: E402  (the organisers' file, unchanged)


def main(pred_path: str, gt_path: str, out: str = str(ROOT / "dev" / "dev_eval.json")):
    pred = json.loads(Path(pred_path).read_text())
    gt = json.loads(Path(gt_path).read_text())
    rep = evaluate.evaluate(gt, pred)
    a = rep["part_a"]
    rows = []
    for c, pc in sorted(a["per_class"].items()):
        n_gt = sum(1 for v in gt.values() for e in v["events"] if e[2] == c)
        n_pr = sum(1 for v in pred["videos"].values() for e in v["events"] if e[2] == c)
        rows.append([c, n_gt, n_pr, f'{pc["0.3"]["f1"]:.2f}', f'{pc["0.5"]["f1"]:.2f}', f'{pc["0.7"]["f1"]:.2f}',
                     f'{pc["f1_mean"]:.2f}'])
    rows.append(["<b>Score A</b>", "", "", "", "", "", f'<b>{a["score_a"]:.3f}</b>'])
    table = {
        "intro": ("We labelled the four sample clips ourselves with the organisers' start/end conventions "
                  f"({sum(len(v['events']) for v in gt.values())} segments). Labels were made by watching the clips "
                  "and by reviewing candidate events from a deliberately loose version of our rules, so they share "
                  "some of our blind spots. Treat the numbers as a sanity check, not as a test-set estimate. "
                  "Scores below are the official <code>evaluate.py</code> run on <code>predictions_samples.json</code>."),
        "header": ["class", "labelled", "predicted", "F1@0.3", "F1@0.5", "F1@0.7", "mean F1"],
        "rows": rows,
        "score_a": a["score_a"],
    }
    Path(out).write_text(json.dumps(table, indent=1))
    evaluate.print_report(rep)
    return table


if __name__ == "__main__":
    main(*sys.argv[1:3])
