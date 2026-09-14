"""Single source of truth for every production constant."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class Settings:
    # --- paths ---------------------------------------------------------
    root: str = ROOT
    # Phase 6 model. Trained by train_improved.py on processed/fer_plus_v2.npz,
    # selected on validation macro-F1, and measured against the Phase 3 baseline
    # in runs/comparison/: +2.8 points of balanced accuracy on 8,778 held-out
    # images (95% CI +0.3 to +5.5), with expressive faces misread as neutral or
    # happiness falling from 15.1% to 11.7%. The old checkpoint is still on disk
    # at runs/baseline_resnet18/best_model.pt; this line is the only switch.
    checkpoint: str = os.path.join(ROOT, "runs", "improved_resnet18", "best_model.pt")
    detector_model: str = os.path.join(ROOT, "models", "blaze_face_short_range.tflite")
    artifacts_dir: str = os.path.join(ROOT, "artifacts")

    # --- sampling ------------------------------------------------------
    target_fps: float = 5.0

    # --- detection -----------------------------------------------------
    # 0.70 rather than 0.50: the Phase 4A validation found a false positive at
    # 0.546 while the lowest true detection scored 0.856.
    min_detection_confidence: float = 0.70
    min_face_px: int = 48

    # --- crop (calibrated in Phase 4A.1 against FER+ train framing) -----
    crop_margin: float = 1.15
    crop_upward_offset: float = 0.079

    # --- temporal ------------------------------------------------------
    smoothing_window: int = 5          # samples; 5 @ 5 fps = 1.0 s
    min_run: int = 2                   # samples; drop 1-sample spikes
    max_gap_samples: int = 2           # never smooth across a longer gap

    # --- inference -----------------------------------------------------
    batch_size: int = 64

    # --- upload limits -------------------------------------------------
    max_upload_bytes: int = 500 * 1024 * 1024
    allowed_suffixes: tuple = field(
        default=(".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"))

    # --- job lifecycle -------------------------------------------------
    job_ttl_seconds: int = 3600

    @property
    def smoothing_window_seconds(self) -> float:
        """Window duration in SECONDS: samples / sampling fps (not video fps)."""
        return self.smoothing_window / self.target_fps


SETTINGS = Settings()
