"""Single source of truth for every production constant.

Every value that has to differ between a laptop and a serverless deployment is
readable from the environment. Nothing here is a secret, so nothing here is
read from a secret store: these are paths, limits and capability switches.

The defaults reproduce the original local behaviour exactly, so running the app
with no environment set behaves as it always has.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env_path(name: str, *default_parts: str) -> str:
    """An absolute path from the environment, or the repo-relative default.

    A relative value is resolved against ROOT rather than the process working
    directory: a serverless invocation does not start where you think it does.
    """
    raw = os.environ.get(name)
    if not raw:
        return os.path.join(ROOT, *default_parts)
    return raw if os.path.isabs(raw) else os.path.join(ROOT, raw)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def _env_list(name: str, default: tuple) -> tuple:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


# On a read-only serverless filesystem the only writable location is the
# system temp directory. Detected rather than configured, so a deployment
# cannot forget to set it and fail at startup.
_READ_ONLY_FS = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
_DEFAULT_ARTIFACTS = ("/tmp/fer_artifacts",) if _READ_ONLY_FS else ("artifacts",)


@dataclass(frozen=True)
class Settings:
    # --- paths ---------------------------------------------------------
    root: str = ROOT
    checkpoint: str = field(default_factory=lambda: _env_path(
        "FER_CHECKPOINT", "runs", "improved_resnet18", "best_model.pt"))
    detector_model: str = field(default_factory=lambda: _env_path(
        "FER_DETECTOR_MODEL", "models", "blaze_face_short_range.tflite"))
    artifacts_dir: str = field(default_factory=lambda: _env_path(
        "FER_ARTIFACTS_DIR", *_DEFAULT_ARTIFACTS))

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
    # A serverless platform caps the REQUEST BODY long before it caps the file.
    # Vercel's cap is 4.5 MB, so a deployment sets FER_MAX_UPLOAD_MB=4 and the
    # UI reads the real number back from /api/health instead of promising 200.
    max_upload_bytes: int = field(default_factory=lambda:
                                  _env_int("FER_MAX_UPLOAD_MB", 200) * 1024 * 1024)
    allowed_suffixes: tuple = field(
        default=(".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"))

    # --- job lifecycle -------------------------------------------------
    job_ttl_seconds: int = 3600
    # Background jobs keep state in this process's memory. That is correct on a
    # long-lived server and WRONG on serverless, where the next poll may reach a
    # different instance that has never heard of the job. Deployments that
    # cannot guarantee instance affinity set FER_ASYNC_JOBS=0, and the UI reads
    # the flag from /api/health and uses the synchronous endpoint instead.
    async_jobs_enabled: bool = field(default_factory=lambda:
                                     _env_bool("FER_ASYNC_JOBS", not _READ_ONLY_FS))

    # --- http ----------------------------------------------------------
    # Same-origin deployments need no entries at all; the regex below still
    # covers local development and LAN access from a phone.
    cors_origins: tuple = field(default_factory=lambda: _env_list(
        "FER_CORS_ORIGINS",
        ("http://localhost:5173", "http://127.0.0.1:5173",
         "http://localhost:4173")))

    @property
    def smoothing_window_seconds(self) -> float:
        """Window duration in SECONDS: samples / sampling fps (not video fps)."""
        return self.smoothing_window / self.target_fps

    @property
    def max_upload_mb(self) -> int:
        return self.max_upload_bytes // (1024 * 1024)


SETTINGS = Settings()
