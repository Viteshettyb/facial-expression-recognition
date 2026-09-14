"""In-memory job store with a TTL sweep.

No database: one user analysing one video at a time does not need durability,
and adding one would be unrequested scope.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Job:
    job_id: str
    filename: str
    status: str = "queued"           # queued | processing | completed | failed
    progress: float = 0.0
    stage: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    workdir: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None


class JobStore:
    def __init__(self, ttl_seconds: int = 3600):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.ttl = ttl_seconds

    def create(self, filename: str, workdir: str | None = None) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:16], filename=filename,
                  workdir=workdir)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for key, value in fields.items():
                    setattr(job, key, value)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job and job.workdir:
            shutil.rmtree(job.workdir, ignore_errors=True)
        return job is not None

    def sweep(self) -> int:
        """Drop jobs older than the TTL and remove their temp directories."""
        cutoff = time.time() - self.ttl
        with self._lock:
            stale = [j for j in self._jobs.values() if j.created_at < cutoff]
            for job in stale:
                self._jobs.pop(job.job_id, None)
        for job in stale:
            if job.workdir:
                shutil.rmtree(job.workdir, ignore_errors=True)
        return len(stale)
