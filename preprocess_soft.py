"""
Phase 5A: attach FER+ annotator vote distributions to the frozen dataset.

Reads (all strictly read-only):
    processed/fer_clean.npz   authoritative ordering, labels, split, hashes
    data/fer2013new.csv       the 10 FER+ vote columns
    data/fer2013.csv          pixel strings, used ONLY to compute row hashes

Writes:
    processed/fer_soft.npz
    processed/fer_soft_stats.json

Design
------
Ordering, labels, split and class ordering are COPIED from fer_clean.npz, never
re-derived. The seeded shuffle in preprocess.py is not re-executed, so the
train/val/test assignment cannot move. Votes are attached by indexing the
stored `hash` array in its stored order.

The row filter from preprocess.py is replicated for one purpose only: to
restrict which CSV rows may enter the hash -> votes map. It never decides
membership of the output, which is fixed by fer_clean.npz.

Every assertion runs BEFORE anything is written.
"""

import csv
import hashlib
import json
import os
from collections import defaultdict

import numpy as np

csv.field_size_limit(10 ** 9)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "processed")

CLEAN_NPZ = os.path.join(OUT, "fer_clean.npz")
SOFT_NPZ = os.path.join(OUT, "fer_soft.npz")
SOFT_STATS = os.path.join(OUT, "fer_soft_stats.json")

# Identical to preprocess.py - not imported, so preprocess.py is never executed.
VOTE_COLS = ["neutral", "happiness", "surprise", "sadness", "anger",
             "disgust", "fear", "contempt", "unknown", "NF"]
CLASSES = ["neutral", "happiness", "surprise", "sadness", "anger",
           "disgust", "fear"]

MIN_CONSENSUS = 0.50
MAX_JUNK_VOTES = 2
IMG_TOKENS = 48 * 48

EXPECTED_N = 29259


def load_votes():
    """The 10 FER+ vote columns, one row per fer2013new.csv line."""
    rows = []
    with open(os.path.join(DATA, "fer2013new.csv"), newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert header[2:] == VOTE_COLS, f"unexpected FER+ header: {header}"
        for row in r:
            rows.append([int(c) for c in row[2:]])
    return np.array(rows, dtype=np.int64)


def row_passes_filter(votes, toks):
    """Replicates preprocess.py's per-row filter exactly."""
    if len(toks) != IMG_TOKENS:
        return False
    vals = [int(t) for t in toks]
    if min(vals) < 0 or max(vals) > 255:
        return False

    total = int(votes.sum())
    top = int(votes.max())
    winners = [k for k, v in enumerate(votes) if v == top]

    if len(winners) > 1:
        return False
    name = VOTE_COLS[winners[0]]
    if name in ("NF", "unknown"):
        return False
    if total == 0 or top / total < MIN_CONSENSUS:
        return False
    if votes[VOTE_COLS.index("unknown")] + votes[VOTE_COLS.index("NF")] > MAX_JUNK_VOTES:
        return False
    if name == "contempt":
        return False
    return True


def build_hash_to_votes(V):
    """hash -> vote vector, restricted to rows that pass the filter."""
    mapping = {}
    groups = defaultdict(list)
    n_kept = 0
    with open(os.path.join(DATA, "fer2013.csv"), newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert header == ["emotion", "pixels", "Usage"], f"unexpected header: {header}"
        for i, row in enumerate(r):
            toks = row[1].split()
            if not row_passes_filter(V[i], toks):
                continue
            h = hashlib.md5(" ".join(toks).encode()).hexdigest()
            groups[h].append(i)
            n_kept += 1
            if h not in mapping:
                mapping[h] = V[i]
    # a hash must never carry two different vote vectors
    conflicts = [h for h, g in groups.items()
                 if len({tuple(V[j]) for j in g}) > 1]
    return mapping, n_kept, len(groups), conflicts


def main():
    print("=" * 72)
    print("PHASE 5A - FER+ VOTE DISTRIBUTION ATTACHMENT")
    print("=" * 72)

    # ---- load the frozen artifact ---------------------------------------
    clean = np.load(CLEAN_NPZ)
    images = clean["images"]
    labels = clean["labels"]
    split = clean["split"]
    hsh = clean["hash"]
    class_names = clean["class_names"]
    n = len(labels)
    print(f"\nfer_clean.npz loaded: {n} samples")
    print(f"  splits: " + ", ".join(
        f"{s}={int((split == s).sum())}" for s in ("train", "val", "test")))

    # ---- read FER+ votes and build the join map -------------------------
    print("\nreading data/fer2013new.csv ...")
    V = load_votes()
    print(f"  vote rows: {len(V)}")

    print("reading data/fer2013.csv to compute pixel hashes (single pass) ...")
    mapping, n_kept_rows, n_kept_hashes, conflicts = build_hash_to_votes(V)
    print(f"  rows passing filter: {n_kept_rows}")
    print(f"  unique hashes among kept rows: {n_kept_hashes}")
    print(f"  hashes carrying conflicting vote vectors: {len(conflicts)}")

    # ---- assemble votes in the STORED order -----------------------------
    print("\nattaching votes in stored order (no reordering, no shuffle) ...")
    missing = [h for h in hsh if h not in mapping]
    votes = np.zeros((n, 10), dtype=np.int64)
    for i, h in enumerate(hsh):
        if h in mapping:
            votes[i] = mapping[h]

    sum7 = votes[:, :7].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        soft = (votes[:, :7].astype(np.float64)
                / np.maximum(sum7, 1)[:, None]).astype(np.float32)

    # =====================================================================
    # VALIDATION - all checks run BEFORE anything is written
    # =====================================================================
    print("\n" + "-" * 72)
    print("VALIDATION")
    print("-" * 72)

    checks = []

    def check(num, desc, ok, detail=""):
        checks.append({"check": num, "description": desc,
                       "passed": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {num:2d}. {desc}"
              + (f"  ({detail})" if detail else ""))
        return ok

    all_ok = True
    all_ok &= check(0, "no hash carries conflicting vote vectors",
                    len(conflicts) == 0, f"conflicts={len(conflicts)}")
    all_ok &= check(1, "every stored hash has a vote vector",
                    len(missing) == 0, f"missing={len(missing)}")
    all_ok &= check(2, "no unmatched hashes remain",
                    len(set(hsh) - set(mapping)) == 0,
                    f"unmatched={len(set(hsh) - set(mapping))}")
    agree = int((soft.argmax(axis=1) == labels.astype(np.int64)).sum())
    all_ok &= check(3, "soft.argmax(1) == labels for every sample",
                    agree == n, f"{agree}/{n}")
    all_ok &= check(4, "votes[:, :7].sum(1) > 0 for every sample",
                    bool((sum7 > 0).all()), f"min sum7={int(sum7.min())}")
    all_ok &= check(5, "soft.sum(1) approximately 1",
                    bool(np.allclose(soft.sum(axis=1), 1.0, atol=1e-5)),
                    f"max dev={float(np.abs(soft.sum(axis=1) - 1.0).max()):.2e}")
    all_ok &= check(6, "images array-equal to fer_clean.npz",
                    bool(np.array_equal(images, clean["images"])))
    all_ok &= check(7, "labels array-equal to fer_clean.npz",
                    bool(np.array_equal(labels, clean["labels"])))
    all_ok &= check(8, "split array-equal to fer_clean.npz",
                    bool(np.array_equal(split, clean["split"])))
    all_ok &= check(9, "hash array-equal to fer_clean.npz",
                    bool(np.array_equal(hsh, clean["hash"])))
    all_ok &= check(10, "class_names array-equal to fer_clean.npz",
                    bool(np.array_equal(class_names, clean["class_names"])))
    all_ok &= check(11, "kept-row unique hash count is exactly 29259",
                    n_kept_hashes == EXPECTED_N, f"got {n_kept_hashes}")
    all_ok &= check(12, "vote_cols[:7] matches class_names",
                    VOTE_COLS[:7] == [str(c) for c in class_names])

    if not all_ok:
        print("\n" + "=" * 72)
        print("ABORTED - one or more assertions failed. Nothing was written.")
        print("=" * 72)
        raise SystemExit(1)

    print("\nall assertions passed - proceeding to write")

    # =====================================================================
    # WRITE
    # =====================================================================
    votes_out = votes.astype(np.uint8)
    assert np.array_equal(votes_out.astype(np.int64), votes), \
        "uint8 downcast lost information"

    np.savez_compressed(
        SOFT_NPZ,
        images=images,
        labels=labels,
        split=split,
        hash=hsh,
        class_names=class_names,
        votes=votes_out,
        vote_cols=np.array(VOTE_COLS),
        soft=soft,
    )

    top = soft.max(axis=1)
    onehot = int((top >= 1.0 - 1e-6).sum())
    frac7 = sum7 / np.maximum(votes.sum(axis=1), 1)

    stats = {
        "phase": "5A",
        "description": "FER+ 7-class annotator vote distributions attached to "
                       "the frozen fer_clean.npz sample set",
        "source": {
            "frozen_artifact": "processed/fer_clean.npz",
            "votes": "data/fer2013new.csv",
            "hash_source": "data/fer2013.csv (pixel strings, hashing only)",
        },
        "join": {
            "key": "md5 of the space-joined pixel string",
            "method": "hash -> vote vector, restricted to rows passing the "
                      "preprocess.py filter; indexed in fer_clean.npz order",
            "split_recomputed": False,
            "shuffle_performed": False,
            "labels_regenerated": False,
            "seed_used": None,
            "rows_passing_filter": int(n_kept_rows),
            "unique_hashes_among_kept_rows": int(n_kept_hashes),
            "hashes_with_conflicting_votes": len(conflicts),
        },
        "arrays": {
            "images": "uint8 (N,48,48)", "labels": "int8 (N,)",
            "split": "str (N,)", "hash": "str (N,)",
            "class_names": "str (7,)", "votes": "uint8 (N,10)",
            "vote_cols": "str (10,)", "soft": "float32 (N,7)",
        },
        "class_names": [str(c) for c in class_names],
        "vote_cols": VOTE_COLS,
        "soft_definition": "votes[:, :7] / votes[:, :7].sum(axis=1)",
        "dropped_vote_columns": VOTE_COLS[7:],
        "samples": int(n),
        "splits": {s: int((split == s).sum()) for s in ("train", "val", "test")},
        "class_distribution": {c: int((labels == k).sum())
                               for k, c in enumerate(class_names)},
        "soft_labels": {
            "top_prob_min": float(top.min()),
            "top_prob_median": float(np.median(top)),
            "top_prob_mean": float(top.mean()),
            "top_prob_max": float(top.max()),
            "one_hot_samples": onehot,
            "one_hot_percent": round(100.0 * onehot / n, 4),
            "argmax_agreement_with_labels": f"{agree}/{n}",
            "sum_max_deviation_from_1":
                float(np.abs(soft.sum(axis=1) - 1.0).max()),
        },
        "renormalisation": {
            "rows_with_contempt_unknown_nf_votes": int((sum7 < votes.sum(axis=1)).sum()),
            "percent_with_discarded_votes":
                round(100.0 * float((sum7 < votes.sum(axis=1)).mean()), 4),
            "retained_mass_min": float(frac7.min()),
            "retained_mass_median": float(np.median(frac7)),
            "retained_mass_mean": float(frac7.mean()),
            "min_denominator_sum7": int(sum7.min()),
        },
        "validation": checks,
        "outputs": {"npz": "processed/fer_soft.npz",
                    "stats": "processed/fer_soft_stats.json"},
        "unmodified": ["processed/fer_clean.npz", "processed/dataset_stats.json",
                       "preprocess.py", "runs/baseline_resnet18/"],
    }

    with open(SOFT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nwrote: {SOFT_NPZ}")
    print(f"wrote: {SOFT_STATS}")
    print(f"\nsoft-label top prob: min {top.min():.4f}  median "
          f"{np.median(top):.4f}  mean {top.mean():.4f}")
    print(f"one-hot samples: {onehot}/{n} ({100.0 * onehot / n:.2f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()
