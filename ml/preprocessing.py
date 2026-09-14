"""Face crop and the exact training-time preprocessing.

The crop transform is the one calibrated in Phase 4A.1 against the FER+ TRAIN
framing distribution (median implied margin 1.150, median upward offset 0.079):

    s    = max(box_w, box_h)
    c_x  = box_x + box_w/2
    c_y  = box_y + box_h/2 - offset * s          (minus = upward)
    side = margin * s

Boundary policy, in order: translate to fit (preserves the calibrated scale),
edge-replicate pad, then reject. Never black-pad - FER+ contains no black bars.
Never naively clamp - that yields a non-square region which distorts on resize.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FER_SIZE = 48


@dataclass
class CropResult:
    image: np.ndarray | None       # BGR crop, square
    status: str                    # "ok" | "face_out_of_bounds" | "invalid_face_box"
    shifted: bool = False
    padded: bool = False
    x: int | None = None
    y: int | None = None
    side: int | None = None
    requested_side: float | None = None


def square_face_crop(frame: np.ndarray, box_x: float, box_y: float,
                     box_w: float, box_h: float, margin: float,
                     upward_offset: float) -> CropResult:
    h, w = frame.shape[:2]
    base = max(float(box_w), float(box_h))
    if base <= 0:
        return CropResult(None, "invalid_face_box")

    side = margin * base
    if side > min(w, h):
        return CropResult(None, "face_out_of_bounds", requested_side=side)

    cx = float(box_x) + float(box_w) / 2.0
    cy = float(box_y) + float(box_h) / 2.0 - upward_offset * side

    left, top = cx - side / 2.0, cy - side / 2.0
    right, bottom = left + side, top + side

    # 1. translate while preserving the exact calibrated size
    shifted = False
    if left < 0:
        right -= left; left = 0.0; shifted = True
    if right > w:
        left -= (right - w); right = float(w); shifted = True
    if top < 0:
        bottom -= top; top = 0.0; shifted = True
    if bottom > h:
        top -= (bottom - h); bottom = float(h); shifted = True

    x1, y1 = int(round(left)), int(round(top))
    x2, y2 = x1 + int(round(side)), y1 + int(round(side))

    if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
        return CropResult(frame[y1:y2, x1:x2], "ok", shifted=shifted,
                          x=x1, y=y1, side=x2 - x1, requested_side=side)

    # 2. edge-replicate pad
    px1, py1 = max(0, x1), max(0, y1)
    px2, py2 = min(w, x2), min(h, y2)
    if px2 <= px1 or py2 <= py1:
        return CropResult(None, "invalid_face_box", shifted=shifted)
    crop = frame[py1:py2, px1:px2]
    crop = cv2.copyMakeBorder(crop, max(0, -y1), max(0, y2 - h),
                              max(0, -x1), max(0, x2 - w),
                              borderType=cv2.BORDER_REPLICATE)
    return CropResult(crop, "ok", shifted=shifted, padded=True,
                      x=x1, y=y1, side=x2 - x1, requested_side=side)


def to_fer48(crop_bgr: np.ndarray) -> np.ndarray:
    """BGR crop -> the stored FER image format: BT.601 grayscale, 48x48 uint8.

    Routing through 48x48 is deliberate. Every image the model was trained and
    evaluated on passed through this bottleneck; feeding a sharp high-resolution
    crop straight to 112x112 would be out of distribution.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (FER_SIZE, FER_SIZE),
                      interpolation=cv2.INTER_AREA).astype(np.uint8)
