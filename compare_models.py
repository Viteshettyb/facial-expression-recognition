"""
Phase 6C: old vs improved, measured on ONE evaluation set.

Reads:  processed/fer_plus_v2.npz  (its val/test splits are byte-identical to
        processed/fer_clean.npz, so both models are scored on exactly the data
        the baseline was always scored on - and neither model trained on it)
        runs/<run>/best_model.pt for every run named on the command line

Writes: runs/comparison/{comparison.json, <run>_<split>_confusion.csv, report.txt}

Each checkpoint is rebuilt through ml.model.EmotionClassifier - the SAME class
the live camera and the upload pipeline use. If a checkpoint cannot be scored
here, it cannot be served either, so this doubles as the deployment check.
Every model is evaluated with the normalisation constants stored in its OWN
checkpoint, because that is what inference does.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from ml.model import EmotionClassifier                 # noqa: E402
from train_baseline import (compute_metrics, confusion_matrix,  # noqa: E402
                            write_confusion_csv)

NPZ = os.path.join(ROOT, "processed", "fer_plus_v2.npz")
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(RUNS, "comparison")


@torch.inference_mode()
def predict(clf: EmotionClassifier, images: np.ndarray,
            batch: int = 256) -> np.ndarray:
    """Run the checkpoint exactly the way ml/model.py runs it in production:
    48x48 uint8 in, resize + normalise from the checkpoint's own constants,
    probabilities out."""
    probs = clf.predict([images[i] for i in range(len(images))], batch_size=batch)
    return probs.argmax(1)


# ------------------------------------------------------------------ analysis
def neutral_happy_bias(y_true, y_pred, class_names):
    """The failure the live camera actually shows: an expressive face read as
    neutral or happiness. Measured as the share of truly EXPRESSIVE samples
    (anger, disgust, fear, sadness, surprise) that land in those two classes."""
    calm = [class_names.index("neutral"), class_names.index("happiness")]
    expressive = ~np.isin(y_true, calm)
    n = int(expressive.sum())
    if n == 0:
        return {"expressive_samples": 0, "misread_as_neutral_or_happiness": 0.0}
    leaked = int(np.isin(y_pred[expressive], calm).sum())
    per_class = {}
    for i, c in enumerate(class_names):
        if i in calm:
            continue
        mask = y_true == i
        if mask.sum():
            per_class[c] = float(np.isin(y_pred[mask], calm).mean())
    return {
        "expressive_samples": n,
        "misread_as_neutral_or_happiness": leaked / n,
        "per_class_leak_rate": per_class,
        "predicted_neutral_or_happiness_share": float(np.isin(y_pred, calm).mean()),
        "true_neutral_or_happiness_share": float(np.isin(y_true, calm).mean()),
    }


def paired_bootstrap(y_true, pred_a, pred_b, class_names, n_boot, seed=0):
    """Resample the evaluation set with replacement and recompute the metric
    difference each time. A difference whose interval straddles zero is not a
    difference, whatever the point estimate says - which matters here because
    disgust has 17 test samples and one image moves its recall by 5.9 points."""
    rng = np.random.default_rng(seed)
    k = len(class_names)
    n = len(y_true)
    diffs = {"macro_f1": [], "balanced_accuracy": [], "accuracy": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ma = compute_metrics(confusion_matrix(y_true[idx], pred_a[idx], k), class_names)
        mb = compute_metrics(confusion_matrix(y_true[idx], pred_b[idx], k), class_names)
        for key in diffs:
            diffs[key].append(mb[key] - ma[key])
    out = {}
    for key, vals in diffs.items():
        v = np.asarray(vals)
        out[key] = {"mean_difference": float(v.mean()),
                    "ci95_low": float(np.percentile(v, 2.5)),
                    "ci95_high": float(np.percentile(v, 97.5)),
                    "prob_improved": float((v > 0).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directory names under runs/")
    ap.add_argument("--dataset", default=NPZ,
                    help="artifact holding the evaluation splits. The merged "
                         "one also carries expw_holdout.")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out-name", default="comparison.json")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="paired bootstrap resamples for the held-out "
                         "difference intervals; 0 disables")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    d = np.load(args.dataset)
    images, labels, split = d["images"], d["labels"].astype(np.int64), d["split"]
    class_names = [str(c) for c in d["class_names"]]
    k = len(class_names)

    results = {}
    preds_by_run: dict = {}
    for run in args.runs:
        ckpt_path = os.path.join(RUNS, run, "best_model.pt")
        if not os.path.exists(ckpt_path):
            raise SystemExit(f"missing checkpoint: {ckpt_path}")
        clf = EmotionClassifier(ckpt_path)
        assert clf.class_names == class_names, \
            f"{run}: class order differs from the dataset"
        info = clf.info
        entry = {
            "checkpoint": os.path.relpath(ckpt_path, ROOT).replace("\\", "/"),
            "epoch": info.epoch,
            "img_size": info.img_size,
            "norm_mean": info.norm_mean,
            "norm_std": info.norm_std,
            "loads_in_production_pipeline": True,
            "splits": {},
        }
        for s in args.splits:
            m = split == s
            pred = predict(clf, images[m])
            preds_by_run.setdefault(run, {})[s] = pred
            cm = confusion_matrix(labels[m], pred, k)
            metrics = compute_metrics(cm, class_names)
            write_confusion_csv(os.path.join(OUT, f"{run}_{s}_confusion.csv"),
                                cm, class_names)
            entry["splits"][s] = {
                **metrics,
                "confusion_matrix": cm.astype(int).tolist(),
                "neutral_happiness_bias": neutral_happy_bias(
                    labels[m], pred, class_names)}
            print(f"{run:34s} {s:5s} acc {metrics['accuracy']:.4f} "
                  f"bal {metrics['balanced_accuracy']:.4f} "
                  f"macroF1 {metrics['macro_f1']:.4f}")
        results[run] = entry
        del clf
        torch.cuda.empty_cache()

    # ---- combined val+test: the widest held-out sample we legitimately have --
    do_combined = "val" in args.splits and "test" in args.splits
    combined = (split == "val") | (split == "test")
    for run in (args.runs if do_combined else []):
        pred = np.concatenate([preds_by_run[run][s] for s in ["val", "test"]])
        y = np.concatenate([labels[split == s] for s in ["val", "test"]])
        cm = confusion_matrix(y, pred, k)
        metrics = compute_metrics(cm, class_names)
        write_confusion_csv(os.path.join(OUT, f"{run}_valtest_confusion.csv"),
                            cm, class_names)
        results[run]["splits"]["val+test"] = {
            **metrics, "confusion_matrix": cm.astype(int).tolist(),
            "neutral_happiness_bias": neutral_happy_bias(y, pred, class_names)}
        print(f"{run:34s} {'v+t':5s} acc {metrics['accuracy']:.4f} "
              f"bal {metrics['balanced_accuracy']:.4f} "
              f"macroF1 {metrics['macro_f1']:.4f}  "
              f"expressive-read-as-calm "
              f"{results[run]['splits']['val+test']['neutral_happiness_bias']['misread_as_neutral_or_happiness']:.4f}")
    if do_combined:
        assert combined.sum() == len(np.concatenate(
            [labels[split == s] for s in ["val", "test"]])), "combined mask mismatch"

    # ---- is the difference real, or is it disgust's 17 samples? -------------
    comparisons = {}
    if do_combined and args.bootstrap > 0 and len(args.runs) >= 2:
        base = args.runs[0]
        y = np.concatenate([labels[split == s] for s in ["val", "test"]])
        pa = np.concatenate([preds_by_run[base][s] for s in ["val", "test"]])
        for run in args.runs[1:]:
            pb = np.concatenate([preds_by_run[run][s] for s in ["val", "test"]])
            comparisons[f"{run}_vs_{base}"] = paired_bootstrap(
                y, pa, pb, class_names, args.bootstrap)
            print()
            print(f"{run} vs {base} (val+test, {args.bootstrap} resamples)")
            for key, v in comparisons[f"{run}_vs_{base}"].items():
                print(f"  {key:20s} {v['mean_difference']:+.4f} "
                      f"[{v['ci95_low']:+.4f}, {v['ci95_high']:+.4f}] "
                      f"P(better)={v['prob_improved']:.3f}")

    with open(os.path.join(OUT, args.out_name), "w", encoding="utf-8") as f:
        json.dump({"dataset": os.path.relpath(args.dataset, ROOT).replace("\\", "/"),
                   "note": "val/test are byte-identical to processed/fer_clean.npz; "
                           "no model in this table trained on either split",
                   "class_names": class_names, "runs": results,
                   "paired_bootstrap_val_test": comparisons}, f, indent=2)

    # ---- readable report --------------------------------------------------
    lines = []
    for s in args.splits:
        lines.append(f"=== {s.upper()} ===")
        lines.append(f"{'run':<34}{'acc':>8}{'bal_acc':>9}{'macroF1':>9}")
        for run, e in results.items():
            m = e["splits"][s]
            lines.append(f"{run:<34}{m['accuracy']:>8.4f}"
                         f"{m['balanced_accuracy']:>9.4f}{m['macro_f1']:>9.4f}")
        lines.append("")
        lines.append(f"{'class':<12}" + "".join(f"{r[:16]:>18}" for r in results))
        for c in class_names:
            row = f"{c:<12}"
            for run, e in results.items():
                p = e["splits"][s]["per_class"][c]
                row += f"{'R ' + format(p['recall'], '.3f') + ' F ' + format(p['f1'], '.3f'):>18}"
            lines.append(row)
        lines.append("")
    report = "\n".join(lines)
    rname = args.out_name.replace(".json", "") + "_report.txt"
    with open(os.path.join(OUT, rname), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print()
    print(report)


if __name__ == "__main__":
    main()
