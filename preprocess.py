"""
FER+ preprocessing for the video facial-expression project.

Reads (read-only):
    data/fer2013.csv     -> 48x48 grayscale pixels
    data/fer2013new.csv  -> FER+ 10-annotator vote labels

Writes:
    processed/fer_clean.npz
    processed/dataset_stats.json

The original FER2013 `emotion` column is ignored: it agrees with the FER+
crowd majority on only 62.3% of images. The original `Usage` split is also
ignored: 594 duplicate images straddle it, which leaks train data into test.
"""

import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict

csv.field_size_limit(10 ** 9)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "processed")

VOTE_COLS = ["neutral", "happiness", "surprise", "sadness", "anger",
             "disgust", "fear", "contempt", "unknown", "NF"]
CLASSES = ["neutral", "happiness", "surprise", "sadness", "anger",
           "disgust", "fear"]

MIN_CONSENSUS = 0.50     # top vote must be >= 50% of votes cast
MAX_JUNK_VOTES = 2       # unknown + NF votes allowed on a kept row
RATIOS = (0.70, 0.15, 0.15)
SEED = 42
IMG_TOKENS = 48 * 48


def load_labels():
    rows = []
    with open(os.path.join(DATA, "fer2013new.csv"), newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert header[2:] == VOTE_COLS, f"unexpected FER+ header: {header}"
        for row in r:
            rows.append((row[0], [int(c) for c in row[2:]]))
    return rows


def load_pixels():
    rows = []
    with open(os.path.join(DATA, "fer2013.csv"), newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert header == ["emotion", "pixels", "Usage"], f"unexpected header: {header}"
        for row in r:
            rows.append((row[1], row[2]))   # emotion column deliberately dropped
    return rows


def main():
    labels = load_labels()
    pixels = load_pixels()

    # --- alignment guard: the join is strictly by row index -------------
    assert len(labels) == len(pixels) == 35887, "row count mismatch"
    assert all(labels[i][0] == pixels[i][1] for i in range(len(labels))), \
        "Usage column disagrees between files - row alignment is not valid"

    drops = Counter()
    kept = []          # (row_index, label_id, pixel_hash, pixel_tokens)

    for i, (_usage, votes) in enumerate(labels):
        pix_str, _ = pixels[i]
        toks = pix_str.split()
        if len(toks) != IMG_TOKENS:
            drops["bad_pixel_length"] += 1
            continue
        vals = [int(t) for t in toks]
        if min(vals) < 0 or max(vals) > 255:
            drops["pixel_out_of_range"] += 1
            continue

        total = sum(votes)
        top = max(votes)
        winners = [k for k, v in enumerate(votes) if v == top]

        if len(winners) > 1:
            drops["tied_top_vote"] += 1
            continue
        name = VOTE_COLS[winners[0]]
        if name in ("NF", "unknown"):
            drops["majority_nf_or_unknown"] += 1
            continue
        if total == 0 or top / total < MIN_CONSENSUS:
            drops["below_consensus_threshold"] += 1
            continue
        if votes[VOTE_COLS.index("unknown")] + votes[VOTE_COLS.index("NF")] > MAX_JUNK_VOTES:
            drops["too_many_junk_votes"] += 1
            continue
        if name == "contempt":
            drops["contempt_removed"] += 1
            continue

        h = hashlib.md5(" ".join(toks).encode()).hexdigest()
        kept.append((i, CLASSES.index(name), h, vals))

    # --- deduplicate on pixel hash -------------------------------------
    by_hash = defaultdict(list)
    for rec in kept:
        by_hash[rec[2]].append(rec)

    conflicting = [h for h, g in by_hash.items() if len({r[1] for r in g}) > 1]
    assert not conflicting, \
        f"{len(conflicting)} duplicate image groups carry conflicting FER+ labels"

    dedup = [sorted(g, key=lambda r: r[0])[0] for g in by_hash.values()]
    dedup.sort(key=lambda r: r[0])                  # deterministic order
    drops["duplicate_images"] = len(kept) - len(dedup)

    # --- stratified split at the (unique) image-hash level --------------
    rng = random.Random(SEED)
    by_class = defaultdict(list)
    for rec in dedup:
        by_class[rec[1]].append(rec)

    assignment = {}
    for cls, recs in by_class.items():
        recs = sorted(recs, key=lambda r: r[2])     # sort by hash: seed-stable
        rng.shuffle(recs)
        n = len(recs)
        n_tr = int(round(n * RATIOS[0]))
        n_va = int(round(n * RATIOS[1]))
        for k, rec in enumerate(recs):
            assignment[rec[2]] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")

    # --- hard leakage guarantees ---------------------------------------
    hashes = [r[2] for r in dedup]
    assert len(set(hashes)) == len(hashes), "a pixel hash survived more than once"
    per_hash_splits = defaultdict(set)
    for rec in dedup:
        per_hash_splits[rec[2]].add(assignment[rec[2]])
    crossing = [h for h, s in per_hash_splits.items() if len(s) != 1]
    assert not crossing, f"{len(crossing)} hashes appear in multiple splits"

    # --- write artifacts ------------------------------------------------
    os.makedirs(OUT, exist_ok=True)
    import numpy as np

    images = np.array([r[3] for r in dedup], dtype=np.uint8).reshape(-1, 48, 48)
    y = np.array([r[1] for r in dedup], dtype=np.int8)
    split = np.array([assignment[r[2]] for r in dedup])
    hsh = np.array(hashes)

    npz_path = os.path.join(OUT, "fer_clean.npz")
    np.savez_compressed(npz_path, images=images, labels=y, split=split,
                        hash=hsh, class_names=np.array(CLASSES))

    dist = {c: int((y == k).sum()) for k, c in enumerate(CLASSES)}
    per_split = {}
    for s in ("train", "val", "test"):
        m = split == s
        per_split[s] = {"total": int(m.sum()),
                        "per_class": {c: int(((y == k) & m).sum()) for k, c in enumerate(CLASSES)}}

    stats = {
        "source_files": {"pixels": "data/fer2013.csv", "labels": "data/fer2013new.csv"},
        "join": "strict row index; verified via row-by-row Usage agreement (35887/35887)",
        "original_emotion_column_used": False,
        "original_usage_split_used": False,
        "rows_read": len(labels),
        "rules": {"min_consensus": MIN_CONSENSUS, "max_junk_votes": MAX_JUNK_VOTES,
                  "classes": CLASSES, "split_ratios": list(RATIOS), "seed": SEED},
        "dropped": dict(drops),
        "rows_passing_filters_before_dedup": len(kept),
        "unique_images_final": len(dedup),
        "class_distribution": dist,
        "splits": per_split,
        "leakage_check": {"unique_hashes": len(set(hashes)), "total_rows": len(hashes),
                          "hashes_spanning_multiple_splits": 0, "passed": True},
    }
    stats_path = os.path.join(OUT, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print("\nwrote:", npz_path)
    print("wrote:", stats_path)


if __name__ == "__main__":
    main()
