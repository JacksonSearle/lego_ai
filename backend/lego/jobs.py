"""Background jobs (training, tuning, history demo) with an event log clients can stream."""
from __future__ import annotations

import secrets
import threading
import time
import traceback
from typing import Any, Callable


class Stopped(Exception):
    """Raised inside a job when the user asked it to stop."""


class Job:
    def __init__(self, kind: str, meta: dict | None = None):
        self.id = secrets.token_hex(6)
        self.kind = kind
        self.meta = meta or {}
        self.status = "running"
        self.error: str | None = None
        self.events: list[dict] = []
        self.created_at = time.time()
        self.stop_event = threading.Event()
        self._lock = threading.Lock()

    def emit(self, event: dict):
        with self._lock:
            event = {"seq": len(self.events), "t": time.time(), **event}
            self.events.append(event)

    def events_since(self, seq: int) -> list[dict]:
        with self._lock:
            return self.events[seq:]

    def check_stop(self):
        if self.stop_event.is_set():
            raise Stopped()

    @property
    def stopping(self) -> bool:
        return self.stop_event.is_set()

    def to_json(self):
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "error": self.error,
            "meta": self.meta,
            "created_at": self.created_at,
            "n_events": len(self.events),
        }


JOBS: dict[str, Job] = {}


def start(kind: str, fn: Callable[[Job], Any], meta: dict | None = None) -> Job:
    job = Job(kind, meta)
    JOBS[job.id] = job

    def runner():
        try:
            fn(job)
            job.status = "stopped" if job.stopping else "completed"
        except Stopped:
            job.status = "stopped"
        except Exception as e:  # noqa: BLE001
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
            job.emit({"type": "error", "message": job.error, "traceback": traceback.format_exc(limit=6)})
        job.emit({"type": "job_end", "status": job.status, "error": job.error})

    threading.Thread(target=runner, name=f"job-{kind}-{job.id}", daemon=True).start()
    return job


def get(job_id: str) -> Job:
    if job_id not in JOBS:
        raise KeyError(f"Job {job_id} not found")
    return JOBS[job_id]


def active(kind: str | None = None) -> list[Job]:
    return [j for j in JOBS.values() if j.status == "running" and (kind is None or j.kind == kind)]
