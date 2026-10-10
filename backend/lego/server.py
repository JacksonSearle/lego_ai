"""HTTP API + static frontend.

Run with:  uv run uvicorn lego.server:app --port 8000
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import blocks  # noqa: F401  (registers built-ins)
from . import batchsize, config as cfgmod, custom, datasets, history, jobs, model as modelmod, predict, progress, runs, templates, tuner
from .registry import CATEGORY_ORDER, REGISTRY
from .train import train_run

app = FastAPI(title="Lego AI")
FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

custom.ensure_example()
CUSTOM_LOAD_ERRORS = custom.load_all()
runs.mark_interrupted()


def _bad(e: Exception, code=400):
    raise HTTPException(code, detail=str(e))


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
@app.get("/api/meta")
def meta():
    return {
        "environment": cfgmod.environment(),
        "hyperparams": cfgmod.HYPERPARAMS,
        "defaults": cfgmod.DEFAULT_TRAINING,
        "categories": CATEGORY_ORDER,
        "search_space": tuner.default_search_space(),
        "custom_errors": CUSTOM_LOAD_ERRORS,
    }


@app.get("/api/blocks")
def list_blocks():
    return [s.to_json() for s in REGISTRY.values()]


@app.get("/api/datasets")
def list_datasets():
    return [{**d.to_json(), "cached": datasets.is_cached(d.id)} for d in datasets.DATASETS.values()]


@app.get("/api/datasets/{dataset_id}/samples")
def dataset_samples(dataset_id: str, n: int = 24, split: str = "test", seed: int = 0):
    try:
        return predict.samples(dataset_id, min(n, 64), split, seed)
    except KeyError as e:
        _bad(e, 404)


@app.get("/api/templates/{dataset_id}")
def get_templates(dataset_id: str):
    return templates.for_dataset(dataset_id)


@app.post("/api/inspect")
def inspect_arch(body: dict = Body(...)):
    try:
        info = datasets.get_info(body["dataset"])
    except KeyError as e:
        _bad(e)
    return modelmod.inspect(body.get("architecture") or {"blocks": []}, info.input_shape, info.ctx()).to_json()


# ---------------------------------------------------------------------------
# Custom blocks
# ---------------------------------------------------------------------------
@app.get("/api/custom-blocks")
def list_custom():
    return {"files": custom.list_files(), "example": custom.EXAMPLE}


@app.post("/api/custom-blocks/validate")
def validate_custom(body: dict = Body(...)):
    try:
        specs = custom.exec_source(body["source"], body.get("filename") or "untitled.py")
        return {"ok": True, "blocks": [s.to_json() for s in specs]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.put("/api/custom-blocks/{filename}")
def save_custom(filename: str, body: dict = Body(...)):
    try:
        specs = custom.save(filename, body["source"])
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=400)
    CUSTOM_LOAD_ERRORS.pop(filename, None)
    return {"ok": True, "blocks": [s.to_json() for s in specs]}


@app.delete("/api/custom-blocks/{filename}")
def delete_custom(filename: str):
    custom.delete(filename)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _prepare_config(raw: dict) -> dict:
    """Normalize a config from the UI and embed the source of any custom blocks it uses."""
    cfg = cfgmod.normalize(raw)
    used = {b["type"] for b in cfg["architecture"]["blocks"]}
    embedded = cfg.get("custom_blocks") or {}
    current = custom.sources_for(used)
    cfg["custom_blocks"] = {**{k: v for k, v in embedded.items() if k not in current}, **current}
    info = datasets.get_info(cfg["dataset"]["id"])
    overlay = custom.build_registry_overlay(cfg["custom_blocks"])
    report = modelmod.inspect(cfg["architecture"], info.input_shape, info.ctx(), overlay)
    if not report.ok:
        raise ValueError(report.error)
    return cfg


def _start_training(run_id: str, *, resume=False) -> jobs.Job:
    busy = [j for j in jobs.active("train") if j.meta.get("run_id") == run_id]
    if busy:
        raise ValueError("This run is already training.")
    return jobs.start("train", lambda job: train_run(job, run_id, resume=resume), meta={"run_id": run_id})


@app.post("/api/train")
def train(body: dict = Body(...)):
    try:
        cfg = _prepare_config(body["config"])
        run = runs.create(cfg, reproduce_of=body.get("reproduce_of"))
        job = _start_training(run["id"])
    except (ValueError, KeyError) as e:
        _bad(e)
    return {"job": job.to_json(), "run": run}


@app.post("/api/runs/{run_id}/resume")
def resume(run_id: str, body: dict = Body(default={})):
    try:
        run = runs.get(run_id)
    except KeyError as e:
        _bad(e, 404)
    if not runs.checkpoint_path(run_id).exists():
        _bad(ValueError("This run has no checkpoint to resume from."))
    epochs = int(body.get("epochs") or run["config"]["training"]["epochs"])
    done = run["summary"].get("epochs_done", 0)
    if epochs <= done:
        _bad(ValueError(f"Run already has {done} epochs; choose a larger total."))
    run["config"]["training"]["epochs"] = epochs
    run["status"] = "queued"
    runs.save(run)
    try:
        job = _start_training(run_id, resume=True)
    except ValueError as e:
        _bad(e)
    return {"job": job.to_json(), "run": run}


@app.post("/api/runs/{run_id}/retrain")
def retrain(run_id: str):
    """Train the exact same config again from scratch (reproducibility check)."""
    try:
        src = runs.get(run_id)
        notes = custom.ensure_from_config(src["config"].get("custom_blocks"))
        cfg = cfgmod.normalize(src["config"])
        cfg.pop("environment", None)
        cfg["name"] = src["config"]["name"]
        run = runs.create(cfg, reproduce_of=run_id)
        job = _start_training(run["id"])
    except (ValueError, KeyError) as e:
        _bad(e)
    return {"job": job.to_json(), "run": run, "notes": notes, "env_diffs": cfgmod.env_differences(src["config"].get("environment"))}


@app.post("/api/batch-size")
def probe_batch_size(body: dict = Body(...)):
    try:
        cfg = _prepare_config(body["config"])
        return batchsize.probe(cfg, requested=cfg["training"]["batch_size"], only_requested=bool(body.get("only_requested")))
    except (ValueError, KeyError, RuntimeError) as e:
        _bad(e)


@app.post("/api/tune")
def tune(body: dict = Body(...)):
    try:
        cfg = _prepare_config(body["config"])
    except (ValueError, KeyError) as e:
        _bad(e)
    job = jobs.start("tune", lambda job: tuner.run_study(job, cfg, body.get("optuna") or {}), meta={"name": cfg["name"]})
    return {"job": job.to_json()}


@app.get("/api/studies")
def studies():
    return tuner.list_studies()


@app.post("/api/config/import")
def import_config(body: dict = Body(...)):
    """Validate an uploaded run config; installs embedded custom blocks if missing."""
    try:
        cfg = cfgmod.normalize(body)
        notes = custom.ensure_from_config(cfg.get("custom_blocks"))
        info = datasets.get_info(cfg["dataset"]["id"])
        overlay = custom.build_registry_overlay(cfg.get("custom_blocks"))
        report = modelmod.inspect(cfg["architecture"], info.input_shape, info.ctx(), overlay)
    except (ValueError, KeyError, TypeError) as e:
        _bad(e)
    return {"config": cfg, "notes": notes, "env_diffs": cfgmod.env_differences(body.get("environment")), "ok": report.ok, "error": report.error}


# ---------------------------------------------------------------------------
# Runs / leaderboard
# ---------------------------------------------------------------------------
@app.get("/api/runs")
def list_runs(include_demo: bool = False):
    out = []
    for r in runs.list_runs():
        if r.get("demo") and not include_demo:
            continue
        c = r["config"]
        out.append({
            "id": r["id"], "name": r["name"], "status": r["status"], "created_at": r["created_at"],
            "updated_at": r["updated_at"], "dataset": c["dataset"]["id"], "summary": r["summary"], "error": r.get("error"),
            "study": r.get("study"), "reproduce_of": r.get("reproduce_of"),
            "training": c["training"], "n_blocks": len(c["architecture"]["blocks"]),
            "block_types": [b["type"] for b in c["architecture"]["blocks"]],
            "has_checkpoint": runs.checkpoint_path(r["id"]).exists(),
        })
    return out


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    try:
        r = runs.get(run_id)
    except KeyError as e:
        _bad(e, 404)
    return {**r, "metrics": runs.metrics(run_id), "has_checkpoint": runs.checkpoint_path(run_id).exists()}


@app.post("/api/runs/{run_id}/predict")
def predict_run(run_id: str, body: dict = Body(default={})):
    """Run the trained model on a few test examples (for showing beginners what it guessed)."""
    try:
        return predict.predict(run_id, n=min(int(body.get("n", 12)), 48), seed=int(body.get("seed", 0)))
    except (KeyError, FileNotFoundError) as e:
        _bad(e, 404)


@app.patch("/api/runs/{run_id}")
def rename_run(run_id: str, body: dict = Body(...)):
    r = runs.get(run_id)
    if "name" in body:
        r["name"] = r["config"]["name"] = str(body["name"])[:200]
    runs.save(r)
    return {"ok": True}


@app.get("/api/runs/{run_id}/config")
def download_config(run_id: str):
    r = runs.get(run_id)
    data = json.dumps({**r["config"], "result": r["summary"]}, indent=2)
    return StreamingResponse(
        iter([data]), media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.lego.json"'},
    )


@app.get("/api/runs/{run_id}/checkpoint")
def download_checkpoint(run_id: str, which: str = "best"):
    p = runs.checkpoint_path(run_id, which if which in ("best", "last") else "best")
    if not p.exists():
        _bad(KeyError("No checkpoint"), 404)
    return FileResponse(p, filename=f"{run_id}-{which}.pt")


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    if any(j.meta.get("run_id") == run_id for j in jobs.active()):
        _bad(ValueError("Stop the run before deleting it."))
    runs.delete(run_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
@app.get("/api/jobs")
def list_jobs(active_only: bool = True):
    js = jobs.active() if active_only else list(jobs.JOBS.values())
    return [j.to_json() for j in js]


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str):
    try:
        jobs.get(job_id).stop_event.set()
    except KeyError as e:
        _bad(e, 404)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, since: int = 0):
    """Server-sent events: replays the job's log from ``since`` then streams new events."""
    try:
        job = jobs.get(job_id)
    except KeyError as e:
        _bad(e, 404)

    async def gen():
        seq = since
        idle = 0
        while True:
            evs = job.events_since(seq)
            for e in evs:
                yield f"id: {e['seq']}\ndata: {json.dumps(runs.sanitize(e), default=str)}\n\n"
            seq += len(evs)
            if job.status != "running" and not job.events_since(seq):
                yield "event: end\ndata: {}\n\n"
                return
            idle = 0 if evs else idle + 1
            if idle and idle % 75 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.2)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Learner progress (guided mode)
# ---------------------------------------------------------------------------
@app.get("/api/progress")
def get_progress():
    return {"progress": progress.load()}


@app.put("/api/progress")
def put_progress(body: dict = Body(...)):
    progress.save(body)
    return {"ok": True}


@app.delete("/api/progress")
def reset_progress():
    progress.reset()
    return {"ok": True}


# ---------------------------------------------------------------------------
# History tour
# ---------------------------------------------------------------------------
@app.get("/api/history")
def history_state():
    running = jobs.active("history")
    return {**history.state(), "job": running[0].to_json() if running else None}


@app.post("/api/history/run")
def history_run(body: dict = Body(default={})):
    running = jobs.active("history")
    if running:
        return {"job": running[0].to_json()}
    only = body.get("only")
    job = jobs.start("history", lambda job: history.run_all(job, only), meta={"only": only})
    return {"job": job.to_json()}


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
if FRONTEND.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = FRONTEND / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(FRONTEND / "index.html")
