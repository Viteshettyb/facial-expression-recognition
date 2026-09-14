from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# Production configuration
# ============================================================

TARGET_FPS = 5.0
MIN_DETECTION_CONFIDENCE = 0.5
MIN_FACE_PX = 48

# Calibrated against FER+ training framing.
CROP_MARGIN = 1.15
CROP_UPWARD_OFFSET = 0.079

OUTPUT_SIZE = 48


# ============================================================
# MediaPipe Tasks API
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions


# ============================================================
# Utilities
# ============================================================

def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, p))


def safe_json_number(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def get_video_info(cap: cv2.VideoCapture) -> dict:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count / fps if fps > 0 else 0.0

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": duration,
    }


# ============================================================
# Largest-face selection
# ============================================================

def select_largest_face(detections):
    """
    Select the detection with the largest bounding-box area.

    Returns:
        (detection, area) or (None, 0)
    """
    if not detections:
        return None, 0.0

    best_detection = None
    best_area = -1.0

    for detection in detections:
        bbox = detection.bounding_box

        area = max(0, bbox.width) * max(0, bbox.height)

        if area > best_area:
            best_area = float(area)
            best_detection = detection

    return best_detection, best_area


# ============================================================
# Boundary-safe square crop
# ============================================================

def make_square_crop(
    image: np.ndarray,
    x0: float,
    y0: float,
    box_w: float,
    box_h: float,
    margin: float = CROP_MARGIN,
    upward_offset: float = CROP_UPWARD_OFFSET,
):
    """
    Create a square crop using:

        side = margin * max(box_w, box_h)

        center_x = bbox_center_x
        center_y = bbox_center_y - upward_offset * side

    Boundary policy:
        1. Translate square inside the image when possible.
        2. Edge-pad when translation is insufficient.
        3. Reject if the requested square is larger than
           the image's short side.

    Returns:
        crop, metadata
    """

    height, width = image.shape[:2]

    box_side = max(float(box_w), float(box_h))

    if box_side <= 0:
        return None, {
            "status": "invalid_face_box",
            "crop_shifted": False,
            "crop_padded": False,
        }

    side = margin * box_side

    # Impossible to fit as a square even with padding.
    if side > min(width, height):
        return None, {
            "status": "face_out_of_bounds",
            "crop_shifted": False,
            "crop_padded": False,
            "requested_side_px": side,
        }

    cx = float(x0) + float(box_w) / 2.0
    cy = float(y0) + float(box_h) / 2.0

    # Upward shift.
    cy -= upward_offset * side

    left = cx - side / 2.0
    top = cy - side / 2.0
    right = left + side
    bottom = top + side

    original_left = left
    original_top = top

    shifted = False

    # --------------------------------------------------------
    # First: translate while preserving exact crop size.
    # --------------------------------------------------------

    if left < 0:
        shift = -left
        left += shift
        right += shift
        shifted = True

    if right > width:
        shift = width - right
        left += shift
        right += shift
        shifted = True

    if top < 0:
        shift = -top
        top += shift
        bottom += shift
        shifted = True

    if bottom > height:
        shift = height - bottom
        top += shift
        bottom += shift
        shifted = True

    # Floating-point tolerance.
    if (
        left >= 0
        and top >= 0
        and right <= width
        and bottom <= height
    ):
        x1 = int(round(left))
        y1 = int(round(top))
        x2 = int(round(right))
        y2 = int(round(bottom))

        # Ensure valid integer square.
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = min(width, max(x2, x1 + 1))
        y2 = min(height, max(y2, y1 + 1))

        crop = image[y1:y2, x1:x2]

        return crop, {
            "status": "ok",
            "crop_shifted": bool(shifted),
            "crop_padded": False,
            "crop_x": x1,
            "crop_y": y1,
            "crop_w": x2 - x1,
            "crop_h": y2 - y1,
            "requested_side_px": side,
            "original_left": original_left,
            "original_top": original_top,
        }

    # --------------------------------------------------------
    # Second: edge-replicate padding.
    # --------------------------------------------------------

    # Work in integer coordinates around the requested square.
    x1 = int(np.floor(left))
    y1 = int(np.floor(top))
    x2 = int(np.ceil(right))
    y2 = int(np.ceil(bottom))

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - width)
    pad_bottom = max(0, y2 - height)

    ix1 = max(0, x1)
    iy1 = max(0, y1)
    ix2 = min(width, x2)
    iy2 = min(height, y2)

    if ix2 <= ix1 or iy2 <= iy1:
        return None, {
            "status": "invalid_crop",
            "crop_shifted": bool(shifted),
            "crop_padded": False,
        }

    crop = image[iy1:iy2, ix1:ix2]

    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )

        return crop, {
            "status": "ok",
            "crop_shifted": bool(shifted),
            "crop_padded": True,
            "crop_x": x1,
            "crop_y": y1,
            "crop_w": x2 - x1,
            "crop_h": y2 - y1,
            "requested_side_px": side,
            "original_left": original_left,
            "original_top": original_top,
        }

    return crop, {
        "status": "ok",
        "crop_shifted": bool(shifted),
        "crop_padded": False,
        "crop_x": x1,
        "crop_y": y1,
        "crop_w": x2 - x1,
        "crop_h": y2 - y1,
        "requested_side_px": side,
        "original_left": original_left,
        "original_top": original_top,
    }


# ============================================================
# FER preprocessing
# ============================================================

def preprocess_face(crop: np.ndarray) -> np.ndarray:
    """
    Convert production crop into the stored FER image format:

        BGR
          ↓
        BT.601 grayscale
          ↓
        48x48
          ↓
        uint8

    The normalization / tensor conversion remains outside this
    stage and will be reused from the existing training pipeline.
    """

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(
        gray,
        (OUTPUT_SIZE, OUTPUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    return resized.astype(np.uint8)


# ============================================================
# Main pipeline
# ============================================================

def process_video(
    video_path: Path,
    output_dir: Path,
    detector_model: Path,
    target_fps: float = TARGET_FPS,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    video_path = video_path.resolve()
    detector_model = detector_model.resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not detector_model.exists():
        raise FileNotFoundError(
            f"Face detector model not found: {detector_model}"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_info = get_video_info(cap)

    fps = video_info["fps"]

    if fps <= 0:
        raise RuntimeError("Invalid video FPS.")

    # --------------------------------------------------------
    # Sampling strategy
    #
    # We select frame indices based on target timestamps rather
    # than using a fixed integer stride. This handles phone
    # videos whose FPS is not exactly divisible by 5.
    # --------------------------------------------------------

    duration = video_info["duration_s"]

    timestamps = np.arange(
        0.0,
        duration,
        1.0 / target_fps,
        dtype=np.float64,
    )

    frame_indices = np.clip(
        np.round(timestamps * fps).astype(np.int64),
        0,
        video_info["frame_count"] - 1,
    )

    # Remove duplicate frame indices caused by rounding.
    frame_indices = np.unique(frame_indices)

    options = FaceDetectorOptions(
        base_options=BaseOptions(
            model_asset_path=str(detector_model)
        ),
        running_mode=VisionRunningMode.IMAGE,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )

    detector = FaceDetector.create_from_options(options)

    records = []

    n_faces = 0
    n_usable = 0
    n_no_face = 0
    n_multi_face = 0
    n_shifted = 0
    n_padded = 0
    n_rejected = 0

    confidences = []

    for sample_number, frame_index in enumerate(frame_indices):

        timestamp_s = float(frame_index / fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

        ok, frame = cap.read()

        record = {
            "sample": int(sample_number),
            "frame_index": int(frame_index),
            "timestamp_s": timestamp_s,
            "status": None,
            "face_count": 0,
            "face_confidence": None,
            "face_x": None,
            "face_y": None,
            "face_w": None,
            "face_h": None,
            "crop_x": None,
            "crop_y": None,
            "crop_w": None,
            "crop_h": None,
            "crop_shifted": False,
            "crop_padded": False,
            "output_path": None,
        }

        if not ok or frame is None:
            record["status"] = "frame_read_error"
            records.append(record)
            n_rejected += 1
            continue

        # MediaPipe expects RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        result = detector.detect(mp_image)

        detections = result.detections or []

        record["face_count"] = len(detections)

        if len(detections) == 0:
            record["status"] = "no_face"
            records.append(record)
            n_no_face += 1
            continue

        n_faces += 1

        if len(detections) > 1:
            n_multi_face += 1

        detection, _ = select_largest_face(detections)

        bbox = detection.bounding_box

        confidence = float(detection.categories[0].score)

        record["face_confidence"] = confidence
        record["face_x"] = float(bbox.origin_x)
        record["face_y"] = float(bbox.origin_y)
        record["face_w"] = float(bbox.width)
        record["face_h"] = float(bbox.height)

        confidences.append(confidence)

        if min(bbox.width, bbox.height) < MIN_FACE_PX:
            record["status"] = "face_too_small"
            records.append(record)
            n_rejected += 1
            continue

        crop, crop_meta = make_square_crop(
            frame,
            bbox.origin_x,
            bbox.origin_y,
            bbox.width,
            bbox.height,
        )

        record["status"] = crop_meta["status"]
        record["crop_shifted"] = crop_meta.get(
            "crop_shifted", False
        )
        record["crop_padded"] = crop_meta.get(
            "crop_padded", False
        )

        if crop_meta["status"] != "ok" or crop is None:
            records.append(record)
            n_rejected += 1
            continue

        if record["crop_shifted"]:
            n_shifted += 1

        if record["crop_padded"]:
            n_padded += 1

        record["crop_x"] = crop_meta.get("crop_x")
        record["crop_y"] = crop_meta.get("crop_y")
        record["crop_w"] = crop_meta.get("crop_w")
        record["crop_h"] = crop_meta.get("crop_h")

        processed = preprocess_face(crop)

        filename = (
            f"sample_{sample_number:04d}"
            f"_frame_{frame_index:06d}.png"
        )

        output_path = crops_dir / filename

        # Save the exact 48x48 grayscale model input.
        if not cv2.imwrite(str(output_path), processed):
            raise RuntimeError(
                f"Failed to write: {output_path}"
            )

        record["output_path"] = str(
            output_path.relative_to(output_dir)
        )

        n_usable += 1
        records.append(record)

    detector.close()
    cap.release()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total = len(frame_indices)

    stats = {
        "video": {
            "path": str(video_path),
            **{
                k: safe_json_number(v)
                if isinstance(v, (float, np.floating))
                else v
                for k, v in video_info.items()
            },
        },
        "pipeline": {
            "target_fps": target_fps,
            "sampled_frames": total,
            "detector": "MediaPipe Tasks FaceDetector",
            "model": str(detector_model),
            "running_mode": "IMAGE",
            "min_detection_confidence": MIN_DETECTION_CONFIDENCE,
            "min_face_px": MIN_FACE_PX,
            "crop_margin": CROP_MARGIN,
            "crop_upward_offset": CROP_UPWARD_OFFSET,
            "output_size": "48x48",
            "grayscale": "BT.601",
        },
        "results": {
            "frames_with_face": n_faces,
            "frames_usable": n_usable,
            "frames_no_face": n_no_face,
            "frames_multi_face": n_multi_face,
            "frames_rejected": n_rejected,
            "crop_shifted": n_shifted,
            "crop_padded": n_padded,
            "detection_rate": (
                n_faces / total if total else 0.0
            ),
            "usable_rate": (
                n_usable / total if total else 0.0
            ),
        },
        "confidence": {
            "min": percentile(confidences, 0),
            "p05": percentile(confidences, 5),
            "median": percentile(confidences, 50),
            "mean": (
                float(np.mean(confidences))
                if confidences else None
            ),
            "p95": percentile(confidences, 95),
            "max": percentile(confidences, 100),
        },
    }

    # --------------------------------------------------------
    # Write metadata
    # --------------------------------------------------------

    with open(
        output_dir / "frame_results.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        if records:
            writer = csv.DictWriter(
                f,
                fieldnames=list(records[0].keys()),
            )
            writer.writeheader()
            writer.writerows(records)

    with open(
        output_dir / "pipeline_stats.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
        )

    print(json.dumps(stats, indent=2))

    print()
    print(f"Artifacts written to: {output_dir.resolve()}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4B production phone-video "
            "face preprocessing pipeline."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Input phone video.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/phase4b_video_pipeline"),
        help="Output directory.",
    )

    parser.add_argument(
        "--detector-model",
        type=Path,
        default=Path("models/blaze_face_short_range.tflite"),
        help="BlazeFace short-range model.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=TARGET_FPS,
        help="Target sampling FPS.",
    )

    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be greater than zero.")

    process_video(
        video_path=args.video,
        output_dir=args.output,
        detector_model=args.detector_model,
        target_fps=args.fps,
    )


if __name__ == "__main__":
    main()