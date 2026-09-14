"""MediaPipe Tasks BlazeFace short-range detector, created once per process."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


@dataclass
class Detection:
    x: float
    y: float
    w: float
    h: float
    score: float

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def short_side(self) -> float:
        return min(self.w, self.h)


class FaceDetector:
    """Not thread-safe: callers must serialise access."""

    def __init__(self, model_path: str, min_confidence: float = 0.70):
        self.model_path = model_path
        self.min_confidence = min_confidence
        self._detector = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=min_confidence,
            )
        )

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=np.ascontiguousarray(rgb)))
        out = []
        for det in (result.detections or []):
            bb = det.bounding_box
            score = float(det.categories[0].score) if det.categories else 0.0
            out.append(Detection(float(bb.origin_x), float(bb.origin_y),
                                 float(bb.width), float(bb.height), score))
        return out

    @staticmethod
    def largest(detections: list[Detection]) -> Detection | None:
        """One subject per video: pick by area, not by confidence."""
        return max(detections, key=lambda d: d.area) if detections else None

    def close(self) -> None:
        try:
            self._detector.close()
        except Exception:
            pass
