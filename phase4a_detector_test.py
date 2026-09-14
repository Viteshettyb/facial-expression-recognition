"""
Phase 4A: isolated validation of the MediaPipe BlazeFace short-range detector
and the crop strategy, on a real sample video.

This script does NOT run the emotion model and does NOT touch any dataset split.
It answers one question only: is this detector + crop approach sound enough to
build the Phase 4B pipeline on?

Outputs (all under runs/phase4a_detector_test/):
    detector_stats.json          aggregate statistics
    per_frame.csv                one row per sampled frame
    contact_sheet.jpg            9 annotated frames: early / middle / late
    margin_comparison.jpg        same face at several crop margins + model view
    crops/                       individual crops for close inspection
"""

import csv
import json
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO = r"C:/Users/vites/Downloads/6382080-uhd_2160_3840_24fps.mp4"
MODEL = os.path.join(ROOT, "models", "blaze_face_short_range.tflite")
OUT = os.path.join(ROOT, "runs", "phase4a_detector_test")
CROPS = os.path.join(OUT, "crops")

SAMPLE_FPS = 5.0          # the rate the real pipeline is planned to use
MIN_SCORE = 0.5           # detector confidence gate
MIN_FACE_PX = 48          # below the model's native 48x48, upsampling invents detail
MARGINS = [1.0, 1.2, 1.4, 1.6]   # candidates only - final choice deferred
PROBE_MARGIN = 1.3        # "reasonable initial" margin used for the saved crops


def square_box(cx, cy, side, W, H):
    """Square crop box, clamped to frame; returns (x, y, w, h) and a clipped flag."""
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x1), min(H, y1)
    clipped = (x0, y0, x1, y1) != (cx0, cy0, cx1, cy1)
    return (cx0, cy0, cx1 - cx0, cy1 - cy0), clipped


def model_view(crop_bgr, out_px=200):
    """What the trained ResNet-18 actually receives: gray -> 48x48 -> upscaled."""
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (48, 48), interpolation=cv2.INTER_AREA)
    big = cv2.resize(small, (out_px, out_px), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)


def label(img, text, y=22):
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    os.makedirs(CROPS, exist_ok=True)
    assert os.path.exists(MODEL), f"model bundle missing: {MODEL}"
    assert os.path.exists(VIDEO), f"video missing: {VIDEO}"

    cap = cv2.VideoCapture(VIDEO)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    print(f"video: {W}x{H} @ {fps:.3f} fps, {total} frames, {duration:.2f}s")

    step = fps / SAMPLE_FPS
    want = sorted({int(round(i * step)) for i in range(int(total / step) + 1)
                   if int(round(i * step)) < total})
    print(f"sampling {len(want)} frames at ~{SAMPLE_FPS} fps (every ~{step:.1f} frames)")

    # IMAGE mode on purpose: each frame judged independently, so raw detector
    # stability is visible instead of being hidden by temporal smoothing.
    opts = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=MIN_SCORE,
    )
    detector = vision.FaceDetector.create_from_options(opts)

    rows, keep_frames = [], {}
    want_set = set(want)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            dets = res.detections or []

            rec = {"frame_index": idx, "timestamp_s": round(idx / fps, 3),
                   "n_faces": len(dets)}
            if not dets:
                rec.update({"status": "no_face", "score": None, "x": None, "y": None,
                            "w": None, "h": None, "area_frac": None,
                            "short_side_px": None, "clipped": None})
            else:
                # one subject -> largest face by bounding-box area
                best = max(dets, key=lambda d: d.bounding_box.width * d.bounding_box.height)
                bb = best.bounding_box
                score = float(best.categories[0].score) if best.categories else None
                short = min(bb.width, bb.height)
                status = "ok"
                if score is not None and score < MIN_SCORE:
                    status = "low_detector_confidence"
                elif short < MIN_FACE_PX:
                    status = "face_too_small"
                side = max(bb.width, bb.height) * PROBE_MARGIN
                cx, cy = bb.origin_x + bb.width / 2, bb.origin_y + bb.height / 2
                _, clipped = square_box(cx, cy, side, W, H)
                rec.update({"status": status, "score": score,
                            "x": bb.origin_x, "y": bb.origin_y,
                            "w": bb.width, "h": bb.height,
                            "area_frac": (bb.width * bb.height) / (W * H),
                            "short_side_px": short, "clipped": clipped})
                keep_frames[idx] = (frame, bb, score)
            rows.append(rec)
        idx += 1
    cap.release()
    detector.close()

    # ---------------- statistics ----------------
    ok_rows = [r for r in rows if r["status"] == "ok"]
    miss = [r for r in rows if r["n_faces"] == 0]
    multi = [r for r in rows if r["n_faces"] > 1]
    scores = np.array([r["score"] for r in ok_rows], dtype=float)
    areas = np.array([r["area_frac"] for r in ok_rows], dtype=float)
    shorts = np.array([r["short_side_px"] for r in ok_rows], dtype=float)
    widths = np.array([r["w"] for r in ok_rows], dtype=float)
    heights = np.array([r["h"] for r in ok_rows], dtype=float)

    # jitter between consecutive detected samples, normalised by face size
    cjit, sjit = [], []
    prev = None
    for r in rows:
        if r["status"] != "ok":
            prev = None
            continue
        cx, cy = r["x"] + r["w"] / 2, r["y"] + r["h"] / 2
        side = max(r["w"], r["h"])
        if prev is not None:
            pcx, pcy, pside = prev
            cjit.append(float(np.hypot(cx - pcx, cy - pcy) / pside))
            sjit.append(abs(side - pside) / pside)
        prev = (cx, cy, side)

    def stat(a):
        if len(a) == 0:
            return None
        return {"min": float(np.min(a)), "p05": float(np.percentile(a, 5)),
                "median": float(np.median(a)), "mean": float(np.mean(a)),
                "p95": float(np.percentile(a, 95)), "max": float(np.max(a))}

    stats = {
        "video": {"path": VIDEO, "width": W, "height": H, "fps": fps,
                  "frame_count": total, "duration_s": round(duration, 3)},
        "detector": {"model": os.path.relpath(MODEL, ROOT),
                     "api": "mediapipe.tasks.python.vision.FaceDetector",
                     "running_mode": "IMAGE", "min_detection_confidence": MIN_SCORE,
                     "min_face_px": MIN_FACE_PX, "probe_margin": PROBE_MARGIN},
        "sampling": {"target_fps": SAMPLE_FPS, "n_sampled": len(rows)},
        "detection": {
            "frames_with_face": len(rows) - len(miss),
            "frames_usable_ok": len(ok_rows),
            "frames_no_face": len(miss),
            "frames_multi_face": len(multi),
            "detection_rate": round((len(rows) - len(miss)) / len(rows), 4),
            "usable_rate": round(len(ok_rows) / len(rows), 4),
            "missed_frame_indices": [r["frame_index"] for r in miss],
            "missed_timestamps_s": [r["timestamp_s"] for r in miss],
            "multi_face_frames": [(r["frame_index"], r["n_faces"]) for r in multi],
        },
        "confidence": stat(scores),
        "box_width_px": stat(widths),
        "box_height_px": stat(heights),
        "face_short_side_px": stat(shorts),
        "face_area_fraction": stat(areas),
        "jitter_center_over_facesize": stat(np.array(cjit)),
        "jitter_size_relative": stat(np.array(sjit)),
        "any_crop_clipped_at_margin_%.1f" % PROBE_MARGIN:
            int(sum(1 for r in ok_rows if r["clipped"])),
    }
    with open(os.path.join(OUT, "detector_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(OUT, "per_frame.csv"), "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    # ---------------- visual artifacts ----------------
    det_idx = sorted(keep_frames)
    picks = []
    if det_idx:
        thirds = np.array_split(np.array(det_idx), 3)
        for part in thirds:                       # 3 from early, middle, late
            if len(part):
                picks += [int(part[0]), int(part[len(part) // 2]), int(part[-1])]
    picks = sorted(dict.fromkeys(picks))

    # contact sheet: annotated full frames, 3x3
    tiles = []
    for i in picks[:9]:
        frame, bb, score = keep_frames[i]
        vis = frame.copy()
        cv2.rectangle(vis, (bb.origin_x, bb.origin_y),
                      (bb.origin_x + bb.width, bb.origin_y + bb.height),
                      (0, 255, 0), max(2, W // 400))
        side = max(bb.width, bb.height) * PROBE_MARGIN
        (mx, my, mw, mh), _ = square_box(bb.origin_x + bb.width / 2,
                                         bb.origin_y + bb.height / 2, side, W, H)
        cv2.rectangle(vis, (mx, my), (mx + mw, my + mh), (0, 165, 255),
                      max(2, W // 500))
        t = cv2.resize(vis, (360, int(360 * H / W)))
        label(t, f"f{i} t={i/fps:.1f}s s={score:.2f}")
        tiles.append(t)
    if tiles:
        th, tw = tiles[0].shape[:2]
        while len(tiles) < 9:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[r * 3:(r + 1) * 3]) for r in range(3)])
        cv2.imwrite(os.path.join(OUT, "contact_sheet.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

    # margin comparison: context + each margin + what the model sees
    TS = 220
    rows_img = []
    for i in picks[:3]:
        frame, bb, score = keep_frames[i]
        cx, cy = bb.origin_x + bb.width / 2, bb.origin_y + bb.height / 2
        base = max(bb.width, bb.height)
        row = []
        (gx, gy, gw, gh), _ = square_box(cx, cy, base * 2.4, W, H)
        ctx = frame[gy:gy + gh, gx:gx + gw].copy()
        cv2.rectangle(ctx, (bb.origin_x - gx, bb.origin_y - gy),
                      (bb.origin_x - gx + bb.width, bb.origin_y - gy + bb.height),
                      (0, 255, 0), 3)
        row.append(label(cv2.resize(ctx, (TS, TS)), "context+box"))
        for m in MARGINS:
            (x, y, w, h), clip = square_box(cx, cy, base * m, W, H)
            c = frame[y:y + h, x:x + w]
            row.append(label(cv2.resize(c, (TS, TS)),
                             f"m={m}{' CLIP' if clip else ''}"))
            if m == PROBE_MARGIN or (m == 1.2 and PROBE_MARGIN not in MARGINS):
                cv2.imwrite(os.path.join(CROPS, f"crop_f{i:04d}_m{m}.png"), c)
        (x, y, w, h), _ = square_box(cx, cy, base * PROBE_MARGIN, W, H)
        row.append(label(model_view(frame[y:y + h, x:x + w], TS),
                         f"model sees (48px) m={PROBE_MARGIN}"))
        rows_img.append(np.hstack(row))
    if rows_img:
        cv2.imwrite(os.path.join(OUT, "margin_comparison.jpg"), np.vstack(rows_img),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    # a few standalone crops at the probe margin, spread across the video
    for i in picks[:9]:
        frame, bb, _ = keep_frames[i]
        cx, cy = bb.origin_x + bb.width / 2, bb.origin_y + bb.height / 2
        (x, y, w, h), _ = square_box(cx, cy, max(bb.width, bb.height) * PROBE_MARGIN, W, H)
        cv2.imwrite(os.path.join(CROPS, f"probe_f{i:04d}.png"),
                    cv2.resize(frame[y:y + h, x:x + w], (256, 256)))

    print(json.dumps({k: v for k, v in stats.items() if k != "detection"}, indent=2))
    print(json.dumps(stats["detection"], indent=2))
    print("artifacts in", OUT)


if __name__ == "__main__":
    main()
