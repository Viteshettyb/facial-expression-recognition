"""
Phase 7B: merge the prepared ExpW artifact into the FER+ training set.

Reads:  processed/fer_plus_v2.npz   (FER+ with annotator vote distributions)
        processed/expw_v1.npz       (ExpW crops, same 48x48 pipeline)
Writes: processed/fer_expw_v1.npz
        processed/merge_stats.json

THE RULE THAT SHAPES THIS FILE
------------------------------
Taking only ExpW's disgust and fear - the tempting move, since those are the
classes we are short of - is the one that breaks the model. If nearly every
ExpW-sourced image carried a rare label, then "came from ExpW" would become a
near-perfect predictor of that label, and the network would learn the corpus
instead of the expression. It would score well offline and fail on a webcam.

So ExpW contributes to EVERY class, capped. The cap is what stops ExpW's own
34,883 neutral and 30,537 happy faces from re-importing exactly the
neutral/happiness dominance this whole effort exists to remove.

Two further rules, enforced by assertion rather than by intent:

  * FER+ val and test pass through untouched and unmixed. They stay the
    byte-identical splits every number in runs/comparison/ was measured on.
    ExpW's own holdout is carried alongside under its own split name.
  * ExpW rows carry one-hot targets. FER+ rows carry genuine 10-annotator
    distributions. The `source` column records which is which, so the trainer
    can smooth a single-annotator label harder than a unanimous one instead of
    pretending they are the same kind of evidence.

Selection order within a capped class is by descending detector score, so when
a cap bites it keeps the faces our own detector was most confident about.
"""

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "processed")
FER_NPZ = os.path.join(OUT, "fer_plus_v2.npz")
EXPW_NPZ = os.path.join(OUT, "expw_v1.npz")

# Selection policy. The plan assumed a flat per-class cap, but the measured
# ExpW yield (37,726 usable faces, not the ~22,000 projected) makes a flat cap
# actively harmful: it would hand neutral and happiness +3,500 each and make the
# dominance we are trying to remove WORSE in absolute terms.
#
# So each class is filled TOWARD a target instead:
#
#     take_c = min(available_c, max(FLOOR, TARGET - ferplus_train_c))
#
# TARGET levels the classes that are short. FLOOR guarantees every class still
# receives ExpW rows even when it needs none - without that, "came from ExpW"
# would again predict the rare classes, which is the failure mode this whole
# merge is built to avoid.
DEFAULT_TARGET = 7000
DEFAULT_FLOOR = 1200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET,
                    help="merged per-class train size to fill toward")
    ap.add_argument("--floor", type=int, default=DEFAULT_FLOOR,
                    help="ExpW rows every class receives regardless of need, "
                         "so the corpus cannot predict the label")
    ap.add_argument("--cap", type=int, default=0,
                    help="alternative policy: flat per-class cap, ignoring "
                         "--target/--floor. 0 disables.")
    ap.add_argument("--out-name", default="fer_expw_v1.npz")
    args = ap.parse_args()

    fer = np.load(FER_NPZ)
    expw = np.load(EXPW_NPZ)

    classes = [str(c) for c in fer["class_names"]]
    assert [str(c) for c in expw["class_names"]] == classes, \
        "class order differs between the two artifacts - refusing to merge"
    k = len(classes)

    f_split = fer["split"]
    e_split = expw["split"]

    # ------------------------------------------------------------- FER+ side
    f_images, f_hard, f_soft = fer["images"], fer["labels"], fer["soft"]
    f_source = np.array(["ferplus"] * len(f_hard))

    # ------------------------------------------------------------- ExpW side
    # Only ExpW's own train split may enter our training data. Its holdout is
    # carried through under a distinct split name and never trained on.
    e_images, e_hard = expw["images"], expw["labels"]
    e_score = expw["det_score"]

    f_train = f_split == "train"
    keep = np.zeros(len(e_hard), dtype=bool)
    capped = {}
    for i, c in enumerate(classes):
        idx = np.flatnonzero((e_hard == i) & (e_split == "train"))
        # descending detector score: when the quota bites, keep the faces our
        # own detector was most confident about
        idx = idx[np.argsort(-e_score[idx], kind="stable")]
        have = int(((f_hard == i) & f_train).sum())
        if args.cap > 0:
            want = args.cap
        else:
            want = max(args.floor, args.target - have)
        take = idx[:want]
        keep[take] = True
        capped[c] = {"ferplus_train": have, "expw_available": int(len(idx)),
                     "quota": int(want), "taken": int(len(take)),
                     "left_unused": int(max(0, len(idx) - want)),
                     "quota_unmet_by": int(max(0, want - len(idx)))}
    keep |= (e_split == "holdout")           # holdout always carried, never capped

    e_images, e_hard = e_images[keep], e_hard[keep]
    e_split_kept, e_score = e_split[keep], e_score[keep]
    # One-hot: ExpW gives a single label, and inventing a vote distribution it
    # does not have would be fabricating annotator agreement.
    e_soft = np.eye(k, dtype=np.float32)[e_hard.astype(np.int64)]
    e_source = np.array(["expw"] * len(e_hard))
    e_split_out = np.where(e_split_kept == "holdout", "expw_holdout", "train")

    # ---------------------------------------------------------------- concat
    images = np.concatenate([f_images, e_images])
    hard = np.concatenate([f_hard, e_hard]).astype(np.int8)
    soft = np.concatenate([f_soft, e_soft]).astype(np.float32)
    split = np.concatenate([f_split, e_split_out])
    source = np.concatenate([f_source, e_source])

    # ------------------------------------------------------------- guarantees
    f_eval = (f_split == "val") | (f_split == "test")
    assert int(f_eval.sum()) == int(((split == "val") | (split == "test")).sum()), \
        "the FER+ evaluation splits changed size"
    assert not (source[(split == "val") | (split == "test")] == "expw").any(), \
        "an ExpW row reached the FER+ evaluation splits"
    assert np.allclose(soft.sum(1), 1.0, atol=1e-5), "soft rows do not sum to 1"
    assert (soft.argmax(1) == hard).all(), "hard label is not the soft argmax"
    assert (split[source == "expw"] != "test").all(), "ExpW row in test"
    assert (split[source == "expw"] != "val").all(), "ExpW row in val"

    npz_path = os.path.join(OUT, args.out_name)
    np.savez_compressed(npz_path, images=images, labels=hard, soft=soft,
                        split=split, source=source,
                        class_names=np.array(classes))

    def dist(mask):
        return {c: int(((hard == i) & mask).sum()) for i, c in enumerate(classes)}

    tr = split == "train"
    stats = {
        "phase": "7B",
        "inputs": {"ferplus": "processed/fer_plus_v2.npz",
                   "expw": "processed/expw_v1.npz"},
        "selection_policy": (
            f"flat cap {args.cap}" if args.cap > 0 else
            f"fill toward target={args.target} with floor={args.floor}"),
        "policy_note": (
            "A flat cap was the plan's default. It was replaced once the real "
            "ExpW yield was measured: a flat cap would add thousands of rows to "
            "neutral and happiness, the two classes already dominating, making "
            "the imbalance worse in absolute terms. The floor keeps every class "
            "represented in both corpora so the source cannot predict the label."),
        "per_class_selection": capped,
        "train": {
            "total": int(tr.sum()),
            "per_class": dist(tr),
            "ferplus_only": dist(tr & (source == "ferplus")),
            "expw_only": dist(tr & (source == "expw")),
            "expw_share_per_class": {
                c: round(float(((hard == i) & tr & (source == "expw")).sum()) /
                         max(1, int(((hard == i) & tr).sum())), 4)
                for i, c in enumerate(classes)},
        },
        "ferplus_val": {"total": int((split == "val").sum()),
                        "per_class": dist(split == "val")},
        "ferplus_test": {"total": int((split == "test").sum()),
                         "per_class": dist(split == "test")},
        "expw_holdout": {"total": int((split == "expw_holdout").sum()),
                         "per_class": dist(split == "expw_holdout")},
        "imbalance": {
            "before_max_min_ratio": round(
                max(dist(tr & (source == "ferplus")).values()) /
                max(1, min(dist(tr & (source == "ferplus")).values())), 1),
            "after_max_min_ratio": round(
                max(dist(tr).values()) / max(1, min(dist(tr).values())), 1),
        },
        "targets": {
            "ferplus": "FER+ 10-annotator vote distribution (soft)",
            "expw": "one-hot; ExpW ships a single label per face and no vote "
                    "counts. Recorded via the `source` column so the trainer "
                    "can smooth it harder rather than treat it as consensus.",
        },
        "leakage_guarantees": {
            "ferplus_val_test_unchanged": True,
            "expw_rows_in_ferplus_eval": 0,
            "expw_holdout_kept_separate": True,
        },
        "outputs": {"npz": f"processed/{args.out_name}",
                    "stats": "processed/merge_stats.json"},
    }
    with open(os.path.join(OUT, "merge_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print("\nwrote:", npz_path)


if __name__ == "__main__":
    main()
