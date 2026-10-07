"""On-disk storage for training runs.

    data/runs/<run_id>/run.json        metadata, status, config, summary
    data/runs/<run_id>/metrics.json    per-step and per-epoch history
    data/runs/<run_id>/checkpoints/    last.pt (resumable), best.pt (best val)
"""
from __future__ import annotations

import json
import math
import secrets
import shutil
import threading
import time
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent.parent / "data" / "runs"
_lock = threading.Lock()


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def run_dir(run_id: str) -> Path:
    if not run_id or "/" in run_id or ".." in run_id:
        raise ValueError("bad run id")
    return RUNS_DIR / run_id


def sanitize(o):
    """Replace NaN/inf (not valid JSON) with None, recursively."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    return o


def _write_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(sanitize(data), indent=2))
    tmp.replace(path)


def create(config: dict, **extra) -> dict:
    run_id = new_id()
    d = run_dir(run_id)
    (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    now = time.time()
    run = {
        "id": run_id,
        "name": config.get("name"),
        "created_at": now,
        "updated_at": now,
        "status": "queued",
        "config": config,
        "summary": {},
        "error": None,
        **extra,
    }
    save(run)
    save_metrics(run_id, {"steps": [], "epochs": []})
    return run


def save(run: dict):
    with _lock:
        run["updated_at"] = time.time()
        _write_json(run_dir(run["id"]) / "run.json", run)


def get(run_id: str) -> dict:
    path = run_dir(run_id) / "run.json"
    if not path.exists():
        raise KeyError(f"Run {run_id} not found")
    return json.loads(path.read_text())


def update(run_id: str, **fields) -> dict:
    run = get(run_id)
    run.update(fields)
    save(run)
    return run


def metrics(run_id: str) -> dict:
    path = run_dir(run_id) / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {"steps": [], "epochs": []}


def save_metrics(run_id: str, m: dict):
    _write_json(run_dir(run_id) / "metrics.json", m)


def list_runs() -> list[dict]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in RUNS_DIR.glob("*/run.json"):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    out.sort(key=lambda r: r["created_at"], reverse=True)
    return out


def delete(run_id: str):
    shutil.rmtree(run_dir(run_id), ignore_errors=True)


def checkpoint_path(run_id: str, which: str = "last") -> Path:
    return run_dir(run_id) / "checkpoints" / f"{which}.pt"


def mark_interrupted():
    """Runs left 'running' by a previous server process can't still be running."""
    for r in list_runs():
        if r["status"] in ("running", "queued"):
            r["status"] = "stopped" if checkpoint_path(r["id"]).exists() else "failed"
            r["error"] = r.get("error") or "Server restarted while this run was in progress."
            save(r)
