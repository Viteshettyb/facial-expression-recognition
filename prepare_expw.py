"""
Phase 7A: turn the ExpW release into an artifact our training pipeline can read.

Reads (read-only):
    data/expw_raw/label.lst          91,793 labelled faces over 68,096 images
    data/expw_raw/images/            the extracted origin images
    processed/fer_plus_v2.npz        for the cross-dataset duplicate check

Writes:
    processed/expw_v1.npz
    processed/expw_stats.json

LICENCE
-------
ExpW ships with no licence text - only a citation request. It comes from CUHK
MMLab, whose identically-collected CelebA release states "available for
non-commercial research purposes only", "not property of MMLAB", and forbids
redistributing "any portion of the images and any portion of derived data".
This artifact is therefore treated as non-redistributable derived data: it stays
under processed/, which .gitignore already excludes. See
docs/expw_integration_plan.md section 1.

    Zhang, Luo, Loy & Tang, "From Facial Expression Recognition to Interpersonal
    Relation Prediction", arXiv:1609.06426.

WHAT THIS DOES, AND WHY IT IS NOT A REIMPLEMENTATION
----------------------------------------------------
Every ExpW face is routed through the SAME code the live camera uses:

    ml.detector.FaceDetector          BlazeFace short-range, min conf 0.70
    ml.preprocessing.square_face_crop margin 1.150, upward offset 0.079
    ml.preprocessing.to_fer48         BT.601 grey, INTER_AREA, 48x48 uint8

imported, not copied. The crop transform was calibrated in Phase 4A.1 against
BlazeFace boxes; ExpW's own 2016-detector boxes have different framing
conventions, so they are used ONLY to decide which detected face carries which
label (by IoU), never as the crop geometry. A face BlazeFace cannot find is
dropped rather than rescued: the live camera could never send us such a face, so
training on it teaches something unusable.

SPLIT AND LEAKAGE
-----------------
ExpW never enters the FER+ val/test splits - those stay byte-identical to
fer_clean.npz so every number already in runs/comparison/ stays comparable.
ExpW is split 90/10 into `train` and `holdout` BY SOURCE IMAGE NAME: two faces
cropped from one photo share lighting, camera and compression, so splitting by
face row would leak. Both FER2013 and ExpW were assembled from Google image
search, so every crop is also checked against every FER+ row by md5 and by
64-bit dHash within a Hamming radius.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from configuration.settings import SETTINGS               # noqa: E402
from ml.detector import FaceDetector                      # noqa: E402
from ml.preprocessing import square_face_crop, to_fer48   # noqa: E402
from preprocess_v2 import dhash64                         # noqa: E402

RAW = os.path.join(ROOT, "data", "expw_raw")
OUT = os.path.join(ROOT, "processed")
FER_NPZ = os.path.join(OUT, "fer_plus_v2.npz")

# Our class order, read back from the FER+ artifact and asserted, never assumed.
CLASSES = ["neutral", "happiness", "surprise", "sadness", "anger",
           "disgust", "fear"]
# ExpW label id -> our class name. Exact 1:1; nothing merged, split or invented.
EXPW_TO_OURS = {0: "anger", 1: "disgust", 2: "fear", 3: "happiness",
                4: "sadness", 5: "surprise", 6: "neutral"}

MIN_IOU = 0.40          # below this we cannot say which face the label describes
HOLDOUT_FRACTION = 0.10
NEAR_DUP_HAMMING = 6    # two detectors cropping one face differ by a few pixels
SEED = 42


def iou(a, b):
    """a, b = (x, y, w, h)."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def load_labels(path):
    """label.lst: name face_id top left right bottom confidence label."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) != 8:
                continue
            top, left, right, bottom = (int(p[2]), int(p[3]), int(p[4]), int(p[5]))
            rows.append({
                "image": p[0],
                "face_id": int(p[1]),
                "box": (float(left), float(top),
                        float(right - left), float(bottom - top)),
                "det_conf": float(p[6]),
                "expw_label": int(p[7]),
            })
    return rows


def find_images(root):
    """Map bare filename -> full path, wherever the archive nested them."""
    index = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                index.setdefault(fn, os.path.join(dirpath, fn))
    return index


_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def bits_to_u64(bitstrings) -> np.ndarray:
    """'0101...' (64 chars) -> uint64, so Hamming distance becomes XOR+popcount."""
    return np.array([int(b, 2) for b in bitstrings], dtype=np.uint64)


def hamming_to_all(query: np.uint64, corpus: np.ndarray) -> np.ndarray:
    """Hamming distance from one hash to every hash in `corpus`, vectorised.

    The obvious nested loop is 60k x 34k Python string comparisons - about two
    billion - which does not finish. XOR the packed integers, view the result as
    bytes and sum a 256-entry popcount table instead.
    """
    x = np.bitwise_xor(corpus, query).view(np.uint8).reshape(-1, 8)
    return _POPCOUNT8[x].sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default=os.path.join(RAW, "images"))
    ap.add_argument("--labels", default=os.path.join(RAW, "label.lst"))
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N source images (smoke test)")
    args = ap.parse_args()

    s = SETTINGS
    started = time.time()

    fer = np.load(FER_NPZ)
    assert [str(c) for c in fer["class_names"]] == CLASSES, \
        "FER+ class order differs from this script's - refusing to guess"
    class_index = {c: i for i, c in enumerate(CLASSES)}

    rows = load_labels(args.labels)
    print(f"label rows: {len(rows)}")

    index = find_images(args.images_dir)
    print(f"image files found: {len(index)}")
    if not index:
        raise SystemExit(f"no images under {args.images_dir}")

    by_image = defaultdict(list)
    for r in rows:
        by_image[r["image"]].append(r)
    image_names = sorted(by_image)
    if args.limit:
        image_names = image_names[:args.limit]

    detector = FaceDetector(s.detector_model, s.min_detection_confidence)
    drops = Counter()
    kept = []          # dicts with image, label, crop, dhash, md5

    try:
        for n, name in enumerate(image_names, 1):
            path = index.get(name)
            if path is None:
                drops["image_file_missing"] += len(by_image[name])
                continue
            frame = cv2.imread(path, cv2.IMREAD_COLOR)
            if frame is None:
                drops["image_unreadable"] += len(by_image[name])
                continue

            detections = detector.detect(frame)          # one pass per image
            for r in by_image[name]:
                our = EXPW_TO_OURS[r["expw_label"]]
                if min(r["box"][2], r["box"][3]) < s.min_face_px:
                    drops["labelled_box_too_small"] += 1
                    continue
                if not detections:
                    drops["blazeface_found_nothing"] += 1
                    continue
                best, best_iou = None, 0.0
                for d in detections:
                    v = iou((d.x, d.y, d.w, d.h), r["box"])
                    if v > best_iou:
                        best, best_iou = d, v
                if best is None or best_iou < MIN_IOU:
                    drops["no_detection_matched_label"] += 1
                    continue
                if best.short_side < s.min_face_px:
                    drops["detected_face_too_small"] += 1
                    continue

                crop = square_face_crop(frame, best.x, best.y, best.w, best.h,
                                        s.crop_margin, s.crop_upward_offset)
                if crop.status != "ok" or crop.image is None:
                    drops[f"crop_{crop.status}"] += 1
                    continue

                face48 = to_fer48(crop.image)
                kept.append({
                    "image": name,
                    "label": class_index[our],
                    "face48": face48,
                    "md5": hashlib.md5(face48.tobytes()).hexdigest(),
                    "dhash": dhash64(face48),
                    "det_score": round(float(best.score), 4),
                    "iou": round(float(best_iou), 4),
                })
            if n % 2000 == 0:
                print(f"  {n}/{len(image_names)} images | kept {len(kept)} faces "
                      f"| {time.time() - started:.0f}s", flush=True)
    finally:
        detector.close()

    print(f"detector pass done in {time.time() - started:.0f}s, kept {len(kept)}")

    # ---------------------------------------------------------- de-duplicate
    # 1. within ExpW itself (the same photo reappears under different queries)
    seen = {}
    unique = []
    for rec in kept:
        if rec["md5"] in seen:
            drops["duplicate_within_expw"] += 1
            continue
        seen[rec["md5"]] = True
        unique.append(rec)

    # 2. against FER+ - exact md5 of the 48x48 crop
    fer_md5 = {hashlib.md5(im.tobytes()).hexdigest(): str(sp)
               for im, sp in zip(fer["images"], fer["split"])}

    # 3. against FER+ - near duplicates by dHash within a Hamming radius.
    #    Exact dHash equality is too strict: two detectors cropping the same
    #    face produce slightly different pixels. Both FER2013 and ExpW were
    #    assembled from Google image search, so overlap is plausible, not
    #    hypothetical - this check is the reason we can claim no contamination.
    fer_split_arr = np.array([str(sp) for sp in fer["split"]])
    fer_bits = bits_to_u64([str(x) for x in fer["dhash"]])

    collisions = Counter()
    collision_detail = defaultdict(Counter)
    survivors = []
    for rec in unique:
        hit_split, how = fer_md5.get(rec["md5"]), None
        if hit_split is not None:
            how = "exact_md5"
        else:
            dist = hamming_to_all(np.uint64(int(rec["dhash"], 2)), fer_bits)
            near = np.flatnonzero(dist <= NEAR_DUP_HAMMING)
            if len(near):
                # If it matches several rows, the eval splits win: dropping a
                # usable train image costs little, contaminating val or test
                # would invalidate every comparison we have made.
                hits = set(fer_split_arr[near])
                hit_split = "val" if "val" in hits else (
                    "test" if "test" in hits else "train")
                how = "dhash_exact" if dist[near].min() == 0 else "dhash_near"
        if hit_split is not None:
            collisions[how] += 1
            collision_detail[hit_split][CLASSES[rec["label"]]] += 1
            drops[f"duplicate_of_ferplus_{hit_split}"] += 1
            continue
        survivors.append(rec)

    print(f"after de-duplication: {len(survivors)} faces")

    # ------------------------------------------------------- split by image
    # By SOURCE IMAGE, never by face row: two faces from one photo share
    # lighting, camera and compression.
    rng = np.random.default_rng(SEED)
    imgs = sorted({r["image"] for r in survivors})
    order = rng.permutation(len(imgs))
    n_hold = int(round(len(imgs) * HOLDOUT_FRACTION))
    holdout_images = {imgs[i] for i in order[:n_hold]}
    split = np.array(["holdout" if r["image"] in holdout_images else "train"
                      for r in survivors])

    tr_imgs = {r["image"] for r, sp in zip(survivors, split) if sp == "train"}
    ho_imgs = {r["image"] for r, sp in zip(survivors, split) if sp == "holdout"}
    assert not (tr_imgs & ho_imgs), "a source image straddles the ExpW split"

    images48 = np.stack([r["face48"] for r in survivors]).astype(np.uint8)
    labels = np.array([r["label"] for r in survivors], dtype=np.int8)
    src_image = np.array([r["image"] for r in survivors])
    md5 = np.array([r["md5"] for r in survivors])
    dh = np.array([r["dhash"] for r in survivors])
    det_score = np.array([r["det_score"] for r in survivors], dtype=np.float32)
    iou_arr = np.array([r["iou"] for r in survivors], dtype=np.float32)

    # final guarantees
    assert len(set(md5)) == len(md5), "a duplicate crop survived"
    assert not (set(md5) & set(fer_md5)), "an ExpW crop equals a FER+ crop"

    os.makedirs(OUT, exist_ok=True)
    npz_path = os.path.join(OUT, "expw_v1.npz")
    np.savez_compressed(npz_path, images=images48, labels=labels,
                        split=split, src_image=src_image, md5=md5, dhash=dh,
                        det_score=det_score, iou=iou_arr,
                        class_names=np.array(CLASSES))

    def dist(mask):
        return {c: int(((labels == i) & mask).sum()) for i, c in enumerate(CLASSES)}

    all_mask = np.ones(len(labels), dtype=bool)
    stats = {
        "phase": "7A",
        "source": {
            "dataset": "ExpW (Expression in-the-Wild)",
            "paper": "Zhang, Luo, Loy & Tang, arXiv:1609.06426",
            "origin": "CUHK MMLab, http://mmlab.ie.cuhk.edu.hk/projects/socialrelation/",
            "licence": "no licence text shipped; citation requested. Treated "
                       "under CUHK MMLab's CelebA terms: non-commercial "
                       "research only, no redistribution of images or derived "
                       "data. See docs/expw_integration_plan.md section 1.",
            "label_rows_in_file": len(rows),
            "source_images_processed": len(image_names),
        },
        "pipeline": {
            "detector": "ml.detector.FaceDetector (BlazeFace short-range)",
            "min_detection_confidence": s.min_detection_confidence,
            "min_face_px": s.min_face_px,
            "label_to_detection_match": f"IoU >= {MIN_IOU}",
            "crop": f"ml.preprocessing.square_face_crop(margin={s.crop_margin}, "
                    f"upward_offset={s.crop_upward_offset})",
            "encode": "ml.preprocessing.to_fer48 (BT.601 grey, INTER_AREA, 48x48)",
            "reimplemented_any_step": False,
        },
        "label_mapping": {str(k): v for k, v in EXPW_TO_OURS.items()},
        "faces_kept_after_detector": len(kept),
        "faces_after_dedup": len(survivors),
        "dropped": dict(drops),
        "class_distribution": dist(all_mask),
        "splits": {
            "train": {"total": int((split == "train").sum()),
                      "source_images": len(tr_imgs),
                      "per_class": dist(split == "train")},
            "holdout": {"total": int((split == "holdout").sum()),
                        "source_images": len(ho_imgs),
                        "per_class": dist(split == "holdout")},
        },
        "duplicate_check": {
            "near_duplicate_hamming_radius": NEAR_DUP_HAMMING,
            "method": "md5 of the 48x48 crop, plus 64-bit dHash exact and within "
                      f"Hamming {NEAR_DUP_HAMMING}, against every FER+ row",
            "within_expw_removed": int(drops["duplicate_within_expw"]),
            "vs_ferplus_removed": dict(collisions),
            "vs_ferplus_by_split_and_class": {k: dict(v) for k, v in
                                              collision_detail.items()},
            "ferplus_eval_rows_contaminated": 0,
            "expw_rows_dropped_for_matching_ferplus_eval": int(
                drops["duplicate_of_ferplus_val"] + drops["duplicate_of_ferplus_test"]),
        },
        "split_policy": {
            "ferplus_val_test": "untouched; no ExpW row enters either",
            "expw_split_unit": "source image name",
            "holdout_fraction": HOLDOUT_FRACTION,
            "seed": SEED,
        },
        "quality": {
            "mean_detector_score": float(det_score.mean()) if len(det_score) else 0.0,
            "mean_label_match_iou": float(iou_arr.mean()) if len(iou_arr) else 0.0,
        },
        "elapsed_seconds": round(time.time() - started, 1),
        "outputs": {"npz": "processed/expw_v1.npz",
                    "stats": "processed/expw_stats.json"},
    }
    with open(os.path.join(OUT, "expw_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps({k: stats[k] for k in
                      ("faces_after_dedup", "class_distribution", "splits",
                       "duplicate_check", "dropped")}, indent=2))
    print("\nwrote:", npz_path)


if __name__ == "__main__":
    main()
