"""Training on another computer: a "worker".

A worker is simply another copy of Lego AI, typically on a rented cloud GPU machine,
started with an access token:

    LEGO_TOKEN=some-long-secret HOST=0.0.0.0 ./run.sh

This copy sends the run's config there, relays the worker's live progress to the Train
page as if the run were local, and when it finishes brings the results and the trained
weights home, so the run sits on this leaderboard like any other.
"""
from __future__ import annotations

import base64
import copy
import json
import secrets
import time
import urllib.error
import urllib.request
from dataclasses import asdict

import torch

from . import datasets, runs, settings
from .datasets import CUSTOM_DIR
from .jobs import Job, Stopped


class WorkerError(ValueError):
    pass


def _open(worker: dict, method: str, path: str, body=None, timeout: float = 30):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Authorization": f"Bearer {worker.get('token') or ''}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(worker["url"].rstrip("/") + path, data=data, method=method, headers=headers)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise WorkerError(f"“{worker['name']}” refused the access token.") from None
        try:
            detail = json.loads(e.read()).get("detail")
        except Exception:  # noqa: BLE001
            detail = None
        raise WorkerError(f"“{worker['name']}” answered: {detail or e.reason}") from None
    except (urllib.error.URLError, OSError) as e:
        raise WorkerError(f"Could not reach “{worker['name']}” at {worker['url']}: {getattr(e, 'reason', e)}") from None


def call(worker: dict, method: str, path: str, body=None, timeout: float = 30):
    with _open(worker, method, path, body, timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def find(worker_id: str) -> dict:
    for w in settings.load()["workers"]:
        if w["id"] == worker_id:
            return w
    raise WorkerError("That worker is no longer in your list.")


def probe(worker: dict) -> dict:
    """Is the worker reachable, and what hardware does it have?"""
    try:
        hw = call(worker, "GET", "/api/hardware", timeout=8)
        meta = call(worker, "GET", "/api/version", timeout=8)
    except WorkerError as e:
        return {"ok": False, "error": str(e)}
    gpu = (hw.get("gpus") or [None])[0]
    return {
        "ok": True, "device": hw.get("best_device", "cpu"), "gpu": gpu and gpu.get("name"), "gpu_gb": gpu and gpu.get("memory_gb"),
        "cpu": hw.get("cpu"), "ram_gb": hw.get("ram_gb"), "version": meta.get("version"), "same_version": meta.get("version") == settings.version(),
    }


def add(name: str, url: str, token: str) -> dict:
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise WorkerError("The address must start with http:// or https://")
    w = {"id": secrets.token_hex(4), "name": name.strip()[:40] or url, "url": url, "token": token.strip()}
    state = probe(w)
    if not state["ok"]:
        raise WorkerError(state["error"])
    settings.save({"workers": [*settings.load()["workers"], w]})
    return state


def remove(worker_id: str):
    settings.save({"workers": [w for w in settings.load()["workers"] if w["id"] != worker_id]})


def _push_dataset(worker: dict, info: datasets.DatasetInfo, emit):
    """Copy a dataset the user imported to the worker, unless it already has it."""
    have = {d["id"] for d in call(worker, "GET", "/api/datasets")}
    if info.id in have:
        return
    emit({"type": "status", "message": f"Sending the dataset “{info.name}” to {worker['name']}…"})
    blob = (CUSTOM_DIR / info.id / "data.pt").read_bytes()
    call(worker, "POST", "/api/worker/dataset", {"meta": asdict(datasets.DATASETS[info.id]), "data": base64.b64encode(blob).decode()}, timeout=600)


def _events(worker: dict, job_id: str, since: int):
    """Yield the worker job's events from ``since`` (server-sent events), then return when it ends."""
    with _open(worker, "GET", f"/api/jobs/{job_id}/events?since={since}", timeout=60) as r:
        event = "message"
        for raw in r:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                if event == "end":
                    yield None
                    return
                yield json.loads(line[5:])
            elif not line:
                event = "message"


def run_remote(job: Job, run_id: str, worker_id: str):
    worker = find(worker_id)
    run = runs.get(run_id)
    cfg = run["config"]
    emit = lambda e: job.emit({**e, "run_id": run_id})  # noqa: E731
    try:
        if cfg.get("init_from"):
            raise WorkerError("Fine-tuning starts from weights stored on this computer, so it cannot be sent to a worker yet.")
        emit({"type": "status", "message": f"Contacting {worker['name']}…"})
        state = probe(worker)
        if not state["ok"]:
            raise WorkerError(state["error"])
        info = datasets.info_for(cfg["dataset"])
        if info.custom:
            _push_dataset(worker, info, emit)
        remote_cfg = copy.deepcopy(cfg)
        remote_cfg.pop("environment", None)
        remote_cfg["training"]["device"] = state["device"]
        started = call(worker, "POST", "/api/train", {"config": remote_cfg}, timeout=120)
        rjob, rrun = started["job"]["id"], started["run"]["id"]
        run["status"] = "running"
        run["remote"] = {"worker": worker["name"], "device": state["device"], "gpu": state.get("gpu")}
        run["config"]["training"]["device"] = state["device"]
        runs.save(run)

        seen, stopped, failures = 0, False, 0
        while True:
            try:
                for ev in _events(worker, rjob, seen):
                    if ev is None:
                        break
                    seen = ev["seq"] + 1
                    failures = 0
                    if ev.get("type") not in ("job_end",):
                        emit({k: v for k, v in ev.items() if k not in ("seq", "t", "run_id")})
                    if job.stopping and not stopped:
                        call(worker, "POST", f"/api/jobs/{rjob}/stop")
                        stopped = True
                else:
                    continue  # the stream dropped without an end marker: reconnect
                break
            except WorkerError:
                failures += 1
                if failures > 20:
                    raise
                time.sleep(3)

        emit({"type": "status", "message": "Bringing the results home…"})
        done = call(worker, "GET", f"/api/runs/{rrun}", timeout=60)
        for which in ("best", "last"):
            try:
                with _open(worker, "GET", f"/api/runs/{rrun}/checkpoint?which={which}", timeout=600) as r:
                    runs.checkpoint_path(run_id, which).write_bytes(r.read())
            except WorkerError:
                pass
        # A downloaded checkpoint is only ever tensors and plain numbers: make sure of that before keeping it.
        for which in ("best", "last"):
            p = runs.checkpoint_path(run_id, which)
            if p.exists():
                try:
                    torch.load(p, map_location="cpu", weights_only=True)
                except Exception:  # noqa: BLE001
                    if which == "best":
                        raise WorkerError("The worker returned weights in an unexpected form, so they were not kept.") from None
                    p.unlink()
        runs.save_metrics(run_id, done.get("metrics") or {"steps": [], "epochs": []})
        run = runs.get(run_id)
        run["summary"] = done.get("summary") or {}
        run["status"] = done.get("status", "completed")
        run["error"] = done.get("error")
        run["config"]["environment"] = (done.get("config") or {}).get("environment")
        runs.save(run)
        try:
            call(worker, "DELETE", f"/api/runs/{rrun}")
        except WorkerError:
            pass
        if run["status"] == "failed":
            raise WorkerError(run["error"] or "The run failed on the worker.")
        if stopped:
            raise Stopped()
        return run["summary"]
    except Stopped:
        raise
    except Exception as e:
        run = runs.get(run_id)
        if run["status"] in ("queued", "running"):
            run["status"] = "failed"
            run["error"] = str(e)
            runs.save(run)
        raise


def receive_dataset(meta: dict, data_b64: str) -> str:
    """Worker side: store a dataset sent by the computer that controls this worker."""
    import io

    ds_id = str(meta.get("id") or "")
    if not ds_id.startswith("u_") or not ds_id.replace("_", "").isalnum():
        raise WorkerError("Not a dataset id this worker accepts.")
    blob = base64.b64decode(data_b64)
    torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)  # tensors and plain values only
    d = CUSTOM_DIR / ds_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "data.pt").write_bytes(blob)
    fields = datasets.DatasetInfo.__dataclass_fields__
    info = datasets.DatasetInfo(**{k: v for k, v in meta.items() if k in fields})
    info.custom = True
    (d / "meta.json").write_text(json.dumps(asdict(info), indent=2))
    datasets.register(info)
    return ds_id
