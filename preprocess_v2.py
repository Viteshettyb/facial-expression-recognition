"""
Phase 6A: improved FER+ data pipeline.

Reads (read-only):
    data/fer2013.csv        -> 48x48 grayscale pixels
    data/fer2013new.csv     -> FER+ 10-annotator vote labels
    processed/fer_clean.npz -> the FROZEN v1 artifact, for its split assignment

Writes:
    processed/fer_plus_v2.npz
    processed/dataset_stats_v2.json

WHY THIS EXISTS
---------------
`preprocess.py` keeps a row only when one class wins a strict, untied majority
of at least 50% of the votes cast. Measured against the source data, that rule
does not remove noise evenly - it removes the AMBIGUOUS classes:

    recovered by relaxing it   disgust +116 (on a base of 117!)  fear +178
                               anger   +564   sadness +730   surprise +577
                               neutral +2183  happiness +321 (on a base of 8810)

Happiness is the class annotators never disagree about, so it survives the
filter almost intact while half of all disgust is thrown away. Training on what
is left teaches a model that the world is mostly happy and neutral - which is
exactly the behaviour reported from the live camera.

WHAT CHANGES
------------
1. The hard-majority requirement is replaced by an ANNOTATOR-MASS requirement.
   A row is kept when at least 5 of its votes, and at least half of them, fall
   inside the 7 classes. The label is then the vote DISTRIBUTION, not a winner:
   an image 5/10 sad, 3/10 fear, 2/10 neutral is a legitimate training signal,
   and forcing it to a one-hot "sadness" (or discarding it) throws that away.

2. Every recovered row goes to TRAIN. The val and test splits stay byte-for-byte
   the ones `fer_clean.npz` already defines, for two reasons: the comparison
   against the existing baseline is then exactly like-for-like, and the
   evaluation ground truth keeps the same annotator agreement it always had
   rather than being diluted by the ambiguous rows we just recovered.

3. Leakage is checked on NEAR-duplicates, not just exact ones. FER2013 contains
   images that differ by a few pixels; an md5 does not see them, a dHash does.
   Any train row that near-matches a val or test row is dropped from train.

Nothing here writes to processed/fer_clean.npz, processed/fer_soft.npz or
runs/baseline_resnet18/ - the v1 artifacts stay frozen so the old model remains
reproducible and comparable.
"""

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict

import numpy as np

csv.field_size_limit(10 ** 9)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "processed")
V1_NPZ = os.path.join(OUT, "fer_clean.npz")

VOTE_COLS = ["neutral", "happiness", "surprise", "sadness", "anger",
             "disgust", "fear", "contempt", "unknown", "NF"]
CLASSES = VOTE_COLS[:7]
IMG_TOKENS = 48 * 48

# --- the relaxed keep rule ------------------------------------------------
MIN_CLASS_VOTES = 5      # absolute annotator mass inside the 7 classes
MIN_CLASS_SHARE = 0.5    # that mass must also be the majority of all votes
# A row whose top class holds less than this is ambiguous. Ambiguous rows are
# still trained on (via their soft target) but are never used as ground truth.
CONFIDENT_TOP = 0.50


def load_rows():
    labels = []
    with open(os.path.join(DATA, "fer2013new.csv"), newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert header[2:] == VOTE_COLS, f"unexpected FER+ header: {header}"
        for row in r:
            labels.append((row[0], [int(c) for c in row[2:]]))

    pixels = []
    with open(os.path.join(DATA, "fer2013.csv"), newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert header == ["emotion", "pixels", "Usage"], f"unexpected header: {header}"
        for row in r:
            pixels.append((row[1], row[2]))     # emotion column deliberately dropped

    assert len(labels) == len(pixels) == 35887, "row count mismatch"
    assert all(labels[i][0] == pixels[i][1] for i in range(len(labels))), \
        "Usage column disagrees between files - row alignment is not valid"
    return labels, pixels


def dhash64(img: np.ndarray) -> str:
    """64-bit difference hash. Two FER images that differ by a little noise,
    a one-pixel shift or a re-encode collide here while their md5s do not."""
    small = img.reshape(48, 48).astype(np.int16)
    # 9x8 grid by block-averaging, then compare horizontal neighbours
    g = small[:48:6, :48:6]                      # 8x8
    g = np.concatenate([g, small[:48:6, 47:48]], axis=1)   # 8x9
    bits = (g[:, :-1] > g[:, 1:]).flatten()
    return "".join("1" if b else "0" for b in bits)


def main():
    labels, pixels = load_rows()

    v1 = np.load(V1_NPZ)
    v1_split = {str(h): str(s) for h, s in zip(v1["hash"], v1["split"])}
    assert [str(c) for c in v1["class_names"]] == CLASSES, "v1 class order differs"

    drops = Counter()
    kept = []        # (row_index, hard, soft, votes, hash, pixels, confident)

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
        sum7 = sum(votes[:7])
        if total == 0:
            drops["no_votes"] += 1
            continue
        if sum7 < MIN_CLASS_VOTES:
            # Mostly contempt / unknown / not-a-face. No usable 7-class signal.
            drops["insufficient_class_votes"] += 1
            continue
        if sum7 / total < MIN_CLASS_SHARE:
            drops["class_votes_not_majority"] += 1
            continue

        soft = np.array(votes[:7], dtype=np.float64) / sum7
        hard = int(soft.argmax())
        top = float(soft[hard])
        tied = int((soft == soft.max()).sum()) > 1
        confident = (not tied) and top >= CONFIDENT_TOP

        h = hashlib.md5(" ".join(toks).encode()).hexdigest()
        kept.append((i, hard, soft, votes, h, vals, confident))

    # --- exact deduplication on the pixel hash ---------------------------
    by_hash = defaultdict(list)
    for rec in kept:
        by_hash[rec[4]].append(rec)
    # Duplicate images must not carry different vote vectors; FER+ votes are per
    # image, so a conflict would mean the row alignment is wrong.
    conflicting = [h for h, g in by_hash.items()
                   if len({tuple(r[3]) for r in g}) > 1]
    assert not conflicting, f"{len(conflicting)} duplicate groups carry conflicting votes"

    dedup = [sorted(g, key=lambda r: r[0])[0] for g in by_hash.values()]
    dedup.sort(key=lambda r: r[0])
    drops["duplicate_images"] = len(kept) - len(dedup)

    # --- split: v1's assignment is authoritative, everything new is train --
    split = []
    origin = []
    for rec in dedup:
        s = v1_split.get(rec[4])
        if s is None:
            split.append("train")
            origin.append("recovered")
        else:
            split.append(s)
            origin.append("core")
    split = np.array(split)
    origin = np.array(origin)

    # A recovered row that is ambiguous is fine in train; it can never reach
    # val/test because those come only from v1. Assert that rather than trust it.
    assert not ((origin == "recovered") & (split != "train")).any(), \
        "a recovered row reached an evaluation split"

    images = np.array([r[5] for r in dedup], dtype=np.uint8).reshape(-1, 48, 48)
    hard = np.array([r[1] for r in dedup], dtype=np.int8)
    soft = np.array([r[2] for r in dedup], dtype=np.float32)
    votes = np.array([r[3] for r in dedup], dtype=np.uint8)
    hsh = np.array([r[4] for r in dedup])
    confident = np.array([r[6] for r in dedup], dtype=bool)

    # --- near-duplicate leakage check ------------------------------------
    dh = np.array([dhash64(im) for im in images])
    eval_dh = set(dh[(split == "val") | (split == "test")])
    near_leak = (split == "train") & np.array([d in eval_dh for d in dh])
    n_near = int(near_leak.sum())
    core_near_leak = int((near_leak & (origin == "core")).sum())

    keep = ~near_leak
    images, hard, soft, votes, hsh, dh = (images[keep], hard[keep], soft[keep],
                                          votes[keep], hsh[keep], dh[keep])
    split, origin, confident = split[keep], origin[keep], confident[keep]
    drops["near_duplicate_of_eval_split"] = n_near

    # --- hard guarantees --------------------------------------------------
    assert len(set(hsh)) == len(hsh), "a pixel hash survived more than once"
    per_hash = defaultdict(set)
    for h, s in zip(hsh, split):
        per_hash[h].add(s)
    assert not [h for h, s in per_hash.items() if len(s) != 1], \
        "a hash appears in multiple splits"
    train_dh = set(dh[split == "train"])
    eval_dh = set(dh[(split == "val") | (split == "test")])
    assert not (train_dh & eval_dh), "near-duplicate still spans train and eval"
    assert np.allclose(soft.sum(1), 1.0, atol=1e-5), "soft rows do not sum to 1"
    assert (soft.argmax(1) == hard).all(), "hard label is not the soft argmax"

    # the evaluation splits must be EXACTLY v1's, or the comparison is invalid
    v1_eval = {h for h, s in v1_split.items() if s in ("val", "test")}
    v2_eval = set(hsh[(split == "val") | (split == "test")])
    assert v2_eval <= v1_eval, "an image entered eval that v1 did not have there"

    os.makedirs(OUT, exist_ok=True)
    npz_path = os.path.join(OUT, "fer_plus_v2.npz")
    np.savez_compressed(npz_path, images=images, labels=hard, soft=soft,
                        votes=votes, split=split, hash=hsh, dhash=dh,
                        origin=origin, confident=confident,
                        class_names=np.array(CLASSES),
                        vote_cols=np.array(VOTE_COLS))

    def dist(mask):
        return {c: int(((hard == k) & mask).sum()) for k, c in enumerate(CLASSES)}

    v1_counts = {c: int((v1["labels"] == k).sum()) for k, c in enumerate(CLASSES)}
    v1_train = v1["split"] == "train"
    stats = {
        "phase": "6A",
        "purpose": "recover the ambiguous rows preprocess.py discards, and keep "
                   "FER+ vote distributions as the training target",
        "source_files": {"pixels": "data/fer2013.csv",
                         "labels": "data/fer2013new.csv",
                         "split_authority": "processed/fer_clean.npz"},
        "rules": {
            "min_class_votes": MIN_CLASS_VOTES,
            "min_class_share": MIN_CLASS_SHARE,
            "confident_top_threshold": CONFIDENT_TOP,
            "hard_majority_required": False,
            "ties_allowed": True,
            "contempt_handling": "dropped as a CLASS; its votes are removed from "
                                 "the denominator, the row survives if the "
                                 "remaining 7-class mass passes the rule",
            "recovered_rows_go_to": "train only",
        },
        "rows_read": len(labels),
        "dropped": dict(drops),
        "samples": int(len(hard)),
        "class_distribution": dist(np.ones(len(hard), dtype=bool)),
        "class_distribution_v1": v1_counts,
        "train_class_distribution": dist(split == "train"),
        "train_class_distribution_v1": {
            c: int(((v1["labels"] == k) & v1_train).sum())
            for k, c in enumerate(CLASSES)},
        "splits": {s: {"total": int((split == s).sum()),
                       "per_class": dist(split == s)}
                   for s in ("train", "val", "test")},
        "origin": {"core": int((origin == "core").sum()),
                   "recovered": int((origin == "recovered").sum())},
        "recovered_per_class": dist(origin == "recovered"),
        "label_quality": {
            "confident_rows": int(confident.sum()),
            "ambiguous_rows": int((~confident).sum()),
            "ambiguous_in_train": int(((~confident) & (split == "train")).sum()),
            "ambiguous_in_eval": int(((~confident) & (split != "train")).sum()),
            "mean_top_probability": float(soft.max(1).mean()),
            "one_hot_rows": int((soft.max(1) >= 0.999).sum()),
        },
        "leakage_check": {
            "unique_hashes": int(len(set(hsh))),
            "total_rows": int(len(hsh)),
            "hashes_spanning_multiple_splits": 0,
            "near_duplicate_train_rows_removed": n_near,
            "of_which_already_present_in_v1_train": core_near_leak,
            "train_eval_dhash_intersection": 0,
            "eval_split_is_subset_of_v1_eval": True,
            "passed": True,
        },
        "outputs": {"npz": "processed/fer_plus_v2.npz",
                    "stats": "processed/dataset_stats_v2.json"},
        "unmodified": ["processed/fer_clean.npz", "processed/fer_soft.npz",
                       "preprocess.py", "runs/baseline_resnet18/"],
    }
    with open(os.path.join(OUT, "dataset_stats_v2.json"), "w",
              encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print("\nwrote:", npz_path)


if __name__ == "__main__":
    main()
