"""
Phase 4A.1: calibrate the video face-crop transform against the FER+ training
framing distribution.

Method
------
The framing question is usually answered by eye. It does not have to be.
Running the SAME BlazeFace detector over FER training images measures the
framing directly: if a FER image of side S contains a detected face box of
side s, then that image was cropped at an effective margin of

    m = S / s

and the box centre's offset from the image centre says how the crop was
positioned vertically. Matching the median of that distribution reproduces
the training framing rather than approximating it.

Train split only, seed 42. No inference with the emotion model, no validation
or test data, no training.

Outputs (runs/phase4a_crop_calibration/):
    calibration_report.json
    contact_sheet.jpg
    crops/
"""

import json
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(ROOT, "processed", "fer_clean.npz")
MODEL = os.path.join(ROOT, "models", "blaze_face_short_range.tflite")
VIDEO = r"C:/Users/vites/Downloads/6382080-uhd_2160_3840_24fps.mp4"
OUT = os.path.join(ROOT, "runs", "phase4a_crop_calibration")
CROPS = os.path.join(OUT, "crops")

SEED = 42
N_FER_PROBE = 800          # FER train images to measure framing on
FER_UPSCALE = 384          # 48 -> 384 so BlazeFace has something to work with
FER_MIN_SCORE = 0.30       # lower gate: upscaled 48px faces are soft
VIDEO_MIN_SCORE = 0.50

MARGINS = [1.2, 1.3, 1.4]
OFFSETS = [0.0, 0.08, 0.10]     # upward centre offset, in units of box side
VIDEO_FRAMES = [0, 43, 86, 134, 221]


def square_box(cx, cy, side, W, H):
    """Square crop, clamped to frame. Returns (x, y, w, h), clipped flag."""
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    return (cx0, cy0, cx1 - cx0, cy1 - cy0), (x0, y0, x1, y1) != (cx0, cy0, cx1, cy1)


def fer_view(bgr, out_px=140):
    """Render a colour crop the way the model receives it: gray -> 48 -> upscaled."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, (48, 48), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(cv2.resize(s, (out_px, out_px), interpolation=cv2.INTER_NEAREST),
                        cv2.COLOR_GRAY2BGR)


def tag(img, text, y=16, scale=0.42):
    cv2.putText(img, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def make_detector(min_score):
    return vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=min_score))


def stat(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return None
    return {"n": int(a.size), "min": float(a.min()),
            "p10": float(np.percentile(a, 10)), "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "mean": float(a.mean()),
            "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max()), "std": float(a.std())}


def measure_fer_framing():
    """Empirical margin + centre offset distribution of FER train images."""
    d = np.load(NPZ)
    images, split = d["images"], d["split"]
    labels, class_names = d["labels"], [str(c) for c in d["class_names"]]
    train_idx = np.flatnonzero(split == "train")

    rng = np.random.default_rng(SEED)
    probe = rng.choice(train_idx, size=min(N_FER_PROBE, train_idx.size), replace=False)
    probe.sort()

    det = make_detector(FER_MIN_SCORE)
    margins, dy, dx, scores, hit_idx = [], [], [], [], []
    S = FER_UPSCALE
    for i in probe:
        big = cv2.resize(images[i], (S, S), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(big, cv2.COLOR_GRAY2RGB)
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=np.ascontiguousarray(rgb)))
        if not res.detections:
            continue
        b = max(res.detections,
                key=lambda x: x.bounding_box.width * x.bounding_box.height).bounding_box
        side = max(b.width, b.height)
        if side <= 0:
            continue
        bcx, bcy = b.origin_x + b.width / 2, b.origin_y + b.height / 2
        margins.append(S / side)
        # >0 means the box centre sits BELOW the image centre, i.e. the FER crop
        # is shifted upward relative to the box -> video needs an upward offset
        dy.append((bcy - S / 2) / side)
        dx.append((bcx - S / 2) / side)
        scores.append(float(max(res.detections,
                                key=lambda x: x.bounding_box.width * x.bounding_box.height)
                            .categories[0].score))
        hit_idx.append(int(i))
    det.close()
    return {"probe_n": int(probe.size), "detected_n": len(margins),
            "detect_rate": round(len(margins) / probe.size, 4),
            "margin": stat(margins), "offset_y_up": stat(dy), "offset_x": stat(dx),
            "detector_score": stat(scores)}, probe, hit_idx, images, labels, class_names


def main():
    os.makedirs(CROPS, exist_ok=True)
    fer, probe, hit_idx, images, labels, class_names = measure_fer_framing()
    print("FER framing measured:", json.dumps(fer["margin"], indent=2))

    m_med = fer["margin"]["median"]
    dy_med = fer["offset_y_up"]["median"]

    # ---------------- video candidate crops ----------------
    cap = cv2.VideoCapture(VIDEO)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    det = make_detector(VIDEO_MIN_SCORE)

    frames = {}
    idx = 0
    want = set(VIDEO_FRAMES)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx in want:
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                      data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            if res.detections:
                b = max(res.detections,
                        key=lambda x: x.bounding_box.width * x.bounding_box.height).bounding_box
                frames[idx] = (fr, b)
        idx += 1
    cap.release()
    det.close()

    variants = [("B_raw", 1.0, 0.0)]
    variants += [(f"C_m1.2", 1.2, 0.0), ("D_m1.3", 1.3, 0.0), ("E_m1.4", 1.4, 0.0)]
    variants += [("F_m1.3_up8", 1.3, 0.08), ("G_m1.3_up10", 1.3, 0.10)]
    # H: the framing actually measured from the FER train split
    variants += [(f"H_FERfit_m{m_med:.2f}_up{dy_med:.2f}", round(m_med, 3),
                  round(dy_med, 3))]

    per_variant = {}
    for name, m, off in variants:
        clipped_n, occ, top_cov, bot_cov = 0, [], [], []
        for i, (fr, b) in frames.items():
            side = max(b.width, b.height)
            cx = b.origin_x + b.width / 2
            cy = b.origin_y + b.height / 2 - off * side      # negative y = upward
            (x, y, w, h), clip = square_box(cx, cy, side * m, W, H)
            clipped_n += int(clip)
            occ.append((side * side) / (side * m) ** 2)
            # coverage beyond the detector box, in units of box side
            top_cov.append(m / 2 + off - 0.5)
            bot_cov.append(m / 2 - off - 0.5)
        per_variant[name] = {
            "margin": m, "offset_up": off,
            "aspect_ratio": 1.0,
            "face_occupancy_frac": round(float(np.mean(occ)), 4),
            "background_frac": round(1 - float(np.mean(occ)), 4),
            "forehead_coverage_boxunits": round(float(np.mean(top_cov)), 4),
            "chin_coverage_boxunits": round(float(np.mean(bot_cov)), 4),
            "frames_clipped": clipped_n,
            "frames_complete": len(frames) - clipped_n,
            "margin_delta_vs_fer_median": round(m - m_med, 4),
            "offset_delta_vs_fer_median": round(off - dy_med, 4),
        }

    # ---------------- contact sheet ----------------
    T = 140
    rng = np.random.default_rng(SEED)
    d = np.load(NPZ)
    tr = np.flatnonzero(d["split"] == "train")
    show = rng.choice(tr, size=16, replace=False)
    fer_tiles = []
    for i in show:
        t = cv2.cvtColor(cv2.resize(d["images"][i], (T, T),
                                    interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        fer_tiles.append(t)
    sec_a = np.vstack([np.hstack(fer_tiles[0:8]), np.hstack(fer_tiles[8:16])])
    hdr_a = np.zeros((28, sec_a.shape[1], 3), np.uint8)
    tag(hdr_a, "A) FER+ TRAIN images (seed 42) - the framing we must match", 20, 0.55)

    rows = []
    for i in sorted(frames):
        fr, b = frames[i]
        side = max(b.width, b.height)
        row = []
        for name, m, off in variants:
            cx = b.origin_x + b.width / 2
            cy = b.origin_y + b.height / 2 - off * side
            (x, y, w, h), clip = square_box(cx, cy, side * m, W, H)
            crop = fr[y:y + h, x:x + w]
            tile = fer_view(crop, T)
            tag(tile, name.split("_", 1)[1] + (" CLIP" if clip else ""))
            row.append(tile)
            if name == "D_m1.3":
                cv2.imwrite(os.path.join(CROPS, f"f{i:04d}_{name}_colour.png"),
                            cv2.resize(crop, (256, 256)))
        rows.append(np.hstack(row))
    sec_b = np.vstack(rows)
    hdr_b = np.zeros((28, sec_b.shape[1], 3), np.uint8)
    tag(hdr_b, "B-G) video crops rendered as the model sees them (gray, 48px)", 20, 0.55)

    wmax = max(sec_a.shape[1], sec_b.shape[1])

    def pad(x):
        return cv2.copyMakeBorder(x, 0, 0, 0, wmax - x.shape[1], cv2.BORDER_CONSTANT, value=0)

    sheet = np.vstack([pad(hdr_a), pad(sec_a), pad(hdr_b), pad(sec_b)])
    cv2.imwrite(os.path.join(OUT, "contact_sheet.jpg"), sheet,
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    fer_margins_all = fer["margin"]
    report = {
        "method": ("BlazeFace run over FER+ TRAIN images; implied margin = "
                   "image_side / detected_box_side; offset_y_up = "
                   "(box_centre_y - image_centre_y) / box_side"),
        "seed": SEED, "split_used": "train only",
        "fer_framing": fer,
        "video": {"path": VIDEO, "width": W, "height": H, "fps": fps,
                  "frames_used": sorted(frames)},
        "candidates": per_variant,
        "fer_median_margin": round(m_med, 4),
        "fer_median_offset_up": round(dy_med, 4),
    }
    with open(os.path.join(OUT, "calibration_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"fer_margin": fer["margin"], "fer_offset_y_up": fer["offset_y_up"],
                      "fer_detect_rate": fer["detect_rate"]}, indent=2))
    print(json.dumps(per_variant, indent=2))
    print("artifacts in", OUT)


if __name__ == "__main__":
    main()
