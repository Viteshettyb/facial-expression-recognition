"""Response models. Kept permissive on nested analysis payloads so a valid
report is never rejected by its own serialiser."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    device: str
    gpu: str | None = None
    model_loaded: bool
    detector_loaded: bool
    architecture: str
    checkpoint_epoch: int | None
    validation_macro_f1: float | None
    class_names: list[str]


class JobCreated(BaseModel):
    job_id: str
    status: str
    filename: str


class JobStatus(BaseModel):
    job_id: str
    status: str = Field(description="queued | processing | completed | failed")
    progress: float = 0.0
    stage: str | None = None
    filename: str | None = None
    error: str | None = None
    created_at: float | None = None
    finished_at: float | None = None


class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str
    code: str | None = None


class AnalysisResponse(BaseModel):
    status: str
    job_id: str | None = None
    dominant_emotion: str | None
    emotion_distribution: dict[str, float]
    temporal_label_distribution: dict[str, float] = {}
    confidence: dict[str, Any]
    frame_count: int
    sampled_frames: int
    analyzed_frames: int
    quality: dict[str, Any]
    temporal_segments: list[dict[str, Any]]
    segment_count: int
    transitions: list[dict[str, Any]]
    transition_count: int
    video: dict[str, Any]
    model: dict[str, Any]
    analysis: dict[str, Any]
    warning: str | None = None


class LiveFrameResponse(BaseModel):
    """One live camera frame. Prediction fields are null when no usable face
    was found - the UI shows 'Face not detected' rather than a guess."""

    status: str = Field(description="ok | no_face | face_too_small | ...")
    face_count: int = 0
    detector_score: float | None = None
    box: list[int] | None = None
    frame_width: int
    frame_height: int
    face_area_fraction: float | None = None
    probabilities: dict[str, float] | None = None
    predicted_emotion: str | None = None
    confidence: float | None = None
