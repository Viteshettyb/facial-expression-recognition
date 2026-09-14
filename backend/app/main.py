"""FastAPI application for video-based facial expression analysis.

Model and detector are loaded once in the lifespan and reused. GPU work and the
MediaPipe detector (not thread-safe) are serialised behind one lock; the blocking
pipeline runs in a worker thread so the event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
import traceback
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from backend.app.jobs import JobStore                      # noqa: E402
from backend.app.schemas import (AnalysisResponse, HealthResponse,  # noqa: E402
                                 JobCreated, JobStatus, LiveFrameResponse)
from configuration.settings import SETTINGS                # noqa: E402
from ml.pipeline import VideoEmotionPipeline               # noqa: E402

STATE: dict = {}
CHUNK = 1024 * 1024

# A live camera frame is a single JPEG; anything larger is not a webcam frame.
MAX_LIVE_FRAME_BYTES = 4 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["pipeline"] = VideoEmotionPipeline(SETTINGS)
    STATE["lock"] = asyncio.Lock()
    STATE["jobs"] = JobStore(SETTINGS.job_ttl_seconds)
    os.makedirs(SETTINGS.artifacts_dir, exist_ok=True)
    yield
    try:
        STATE["pipeline"].detector.close()
    except Exception:
        pass
    STATE.clear()


app = FastAPI(
    title="Video Facial Expression Recognition API",
    description="Dominant facial expression and temporal emotion analysis "
                "from video, using a validated ResNet-18 FER+ classifier.",
    version="1.0.0",
    lifespan=lifespan,
)

# A same-origin deployment (frontend and API behind one domain) needs no CORS
# entries at all - the browser never makes a cross-origin request. The list and
# regex below exist for local development and for reaching the dev server from a
# phone on the LAN; both are configurable so production never has to ship a
# hardcoded localhost allowlist.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SETTINGS.cors_origins),
    # Live camera on a phone means the dev server is reached over the LAN, so
    # the origin is a private IP rather than localhost. Private ranges only.
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
                       r"|192\.168\.\d{1,3}\.\d{1,3}"
                       r"|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ helpers
def _save_upload(upload: UploadFile) -> tuple[str, str, int]:
    """Stream to disk. Never read a large upload fully into memory."""
    suffix = os.path.splitext(upload.filename or "")[1].lower()
    if suffix not in SETTINGS.allowed_suffixes:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix or 'unknown'}'. "
                   f"Allowed: {', '.join(SETTINGS.allowed_suffixes)}")

    workdir = tempfile.mkdtemp(prefix="fer_job_")
    path = os.path.join(workdir, f"input{suffix}")
    size = 0
    try:
        with open(path, "wb") as out:
            while True:
                chunk = upload.file.read(CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > SETTINGS.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the "
                               f"{SETTINGS.max_upload_bytes // (1024 * 1024)} MB limit.")
                out.write(chunk)
    except HTTPException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(status_code=500,
                            detail=f"Could not store upload: {exc}") from exc

    if size == 0:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return workdir, path, size


async def _run_analysis(path: str) -> dict:
    pipeline: VideoEmotionPipeline = STATE["pipeline"]
    async with STATE["lock"]:
        return await asyncio.to_thread(pipeline.analyze, path)


# ------------------------------------------------------------------- routes
@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health():
    pipeline: VideoEmotionPipeline | None = STATE.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    info = pipeline.classifier.info
    return HealthResponse(
        status="ok",
        device=info.device,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        model_loaded=True,
        detector_loaded=True,
        architecture="ResNet18",
        checkpoint_epoch=info.epoch,
        validation_macro_f1=info.val_macro_f1,
        class_names=info.class_names,
        # The UI reads these back rather than hardcoding them, so a deployment
        # with a 4.5 MB body cap advertises 4 MB instead of promising 200.
        max_upload_bytes=SETTINGS.max_upload_bytes,
        async_jobs=SETTINGS.async_jobs_enabled,
    )


@app.post("/analyze-video", response_model=AnalysisResponse, tags=["analysis"])
async def analyze_video(file: UploadFile = File(...)):
    """Synchronous analysis. Returns the complete report."""
    workdir, path, size = _save_upload(file)
    try:
        report = await _run_analysis(path)
        report["job_id"] = None
        report["analysis"]["upload_bytes"] = size
        report["analysis"]["source_filename"] = file.filename
        return JSONResponse(content=report)
    except ValueError as exc:
        raise HTTPException(status_code=415,
                            detail=f"Could not decode video: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500,
                            detail=f"Analysis failed: {exc}") from exc
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.post("/api/live/frame", response_model=LiveFrameResponse, tags=["analysis"])
async def live_frame(frame: UploadFile = File(...)):
    """Stateless single-frame inference for the live camera view.

    The client sends one JPEG at a time and waits for the reply before sending
    the next, so this endpoint can never be flooded by one browser tab.
    Temporal smoothing lives in the client, which owns the real frame clock;
    nothing here is remembered between calls.
    """
    pipeline: VideoEmotionPipeline | None = STATE.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    raw = await frame.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty frame.")
    if len(raw) > MAX_LIVE_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="Frame is too large.")

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=415, detail="Could not decode frame.")

    try:
        # Same lock as the video path: the detector is not thread-safe and the
        # GPU is shared with any upload analysis running at the same time.
        async with STATE["lock"]:
            result = await asyncio.to_thread(pipeline.analyze_frame, image)
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500,
                            detail=f"Live inference failed: {exc}") from exc
    return JSONResponse(content=result)


@app.post("/api/jobs", response_model=JobCreated, status_code=202,
          tags=["analysis"])
async def create_job(background: BackgroundTasks, file: UploadFile = File(...)):
    """Asynchronous analysis for long videos: poll /api/jobs/{id}."""
    if not SETTINGS.async_jobs_enabled:
        # Serverless: the instance that accepted this job may not be the one
        # that receives the poll, so a job id we hand out could be a promise we
        # cannot keep. Fail loudly and point at the endpoint that does work.
        raise HTTPException(
            status_code=501,
            detail="Background jobs are disabled in this deployment because "
                   "job state cannot outlive a single request. Use "
                   "POST /analyze-video instead.")
    workdir, path, _ = _save_upload(file)
    jobs: JobStore = STATE["jobs"]
    jobs.sweep()
    job = jobs.create(file.filename or "video", workdir)
    background.add_task(_process_job, job.job_id, path)
    return JobCreated(job_id=job.job_id, status="queued",
                      filename=job.filename)


async def _process_job(job_id: str, path: str) -> None:
    jobs: JobStore = STATE["jobs"]
    pipeline: VideoEmotionPipeline = STATE["pipeline"]
    jobs.update(job_id, status="processing", stage="analysing", progress=0.05)

    def progress(done: int, total: int) -> None:
        jobs.update(job_id, progress=round(0.05 + 0.85 * done / max(total, 1), 3))

    try:
        async with STATE["lock"]:
            report = await asyncio.to_thread(pipeline.analyze, path, None,
                                             progress)
        report["job_id"] = job_id
        jobs.update(job_id, status="completed", progress=1.0, stage="done",
                    result=report, finished_at=time.time())
    except Exception as exc:
        traceback.print_exc()
        jobs.update(job_id, status="failed", error=str(exc),
                    finished_at=time.time(), progress=1.0)
    finally:
        job = jobs.get(job_id)
        if job and job.workdir:
            # keep the report, discard the source video immediately
            try:
                os.remove(path)
            except OSError:
                pass


@app.get("/api/jobs/{job_id}", response_model=JobStatus, tags=["analysis"])
async def job_status(job_id: str):
    job = STATE["jobs"].get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return JobStatus(job_id=job.job_id, status=job.status, progress=job.progress,
                     stage=job.stage, filename=job.filename, error=job.error,
                     created_at=job.created_at, finished_at=job.finished_at)


@app.get("/api/jobs/{job_id}/result", tags=["analysis"])
async def job_result(job_id: str):
    job = STATE["jobs"].get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error or "Analysis failed.")
    if job.status != "completed":
        raise HTTPException(status_code=409,
                            detail=f"Job is {job.status}; result not ready.")
    return JSONResponse(content=job.result)


@app.delete("/api/jobs/{job_id}", tags=["analysis"])
async def delete_job(job_id: str):
    if not STATE["jobs"].delete(job_id):
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return {"status": "deleted", "job_id": job_id}
