"""HTTP API + static frontend.

Run with:  uv run uvicorn lego.server:app --port 8000
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import blocks  # noqa: F401  (registers built-ins)
from . import autopilot, batchsize, combine, config as cfgmod, custom, datasets, export, hardware, host, hub, lab, jobs, model as modelmod, predict, progress, remote, runs, scaling, seeds, settings, shifts, templates, tuner, update, userdata
from .registry import CATEGORY_ORDER, REGISTRY
from .train import train_run

app = FastAPI(title="Lego AI")
app.include_router(host.router)

# When this copy is reachable from other computers (a cloud GPU worker), LEGO_TOKEN protects it:
# every /api request must carry the token, as a header or as the cookie set by opening /?token=...
ACCESS_TOKEN = settings.ACCESS_TOKEN
if host.ENABLED:
    print(f"\n  Leaderboard host “{host.NAME}” is on.\n  Your own access (the app and the Host tab):  http://localhost:{os.environ.get('PORT', '8000')}/?token={ACCESS_TOKEN}\n  Players only need this computer's address.\n", flush=True)


@app.middleware("http")
async def require_token(request: Request, call_next):
    if ACCESS_TOKEN and request.url.path.startswith("/api"):
        given = request.headers.get("authorization", "").removeprefix("Bearer ").strip() or request.cookies.get("lego_token", "")
        if not hmac.compare_digest(given, ACCESS_TOKEN):
            return JSONResponse({"detail": "This copy of Lego AI needs its access token."}, status_code=401)
    response = await call_next(request)
    # Data must never be served from the browser's cache, and the page itself must be re-checked
    # on every load so that an update is picked up. (Files under /assets have unique names.)
    path = request.url.path
    if path.startswith(("/api", "/host")):
        response.headers["Cache-Control"] = "no-store"
    elif not path.startswith("/assets"):
        response.headers["Cache-Control"] = "no-cache"
    if ACCESS_TOKEN and hmac.compare_digest(request.query_params.get("token", ""), ACCESS_TOKEN):
        response.set_cookie("lego_token", ACCESS_TOKEN, httponly=True, samesite="lax", max_age=30 * 86400)
    return response


FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

custom.ensure_example()
CUSTOM_LOAD_ERRORS = custom.load_all()
userdata.load_all()
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
        "version": settings.version(),
        "host_mode": host.ENABLED,
        "host_name": host.NAME if host.ENABLED else None,
    }


@app.get("/api/version")
def version():
    return {"version": settings.version()}


@app.get("/api/blocks")
def list_blocks():
    return [s.to_json() for s in REGISTRY.values()]


@app.get("/api/datasets")
def list_datasets():
    return [{**d.to_json(), "cached": datasets.is_cached(d.id)} for d in datasets.DATASETS.values()]


# ---- importing datasets ----
def _user(fn):
    try:
        return fn()
    except (userdata.ImportError_, hub.HubError) as e:
        _bad(e)


@app.post("/api/datasets/upload")
def dataset_upload(body: dict = Body(...)):
    """Stage an uploaded file (base64) or the worked example, and describe what is in it."""
    if body.get("example"):
        return _user(lambda: userdata.preview(userdata.example()))
    return _user(lambda: userdata.preview(userdata.from_upload(str(body.get("filename") or ""), str(body.get("content") or ""))))


@app.post("/api/datasets/hub")
def dataset_from_hub(body: dict = Body(...)):
    """Stage (the first part of) a Hugging Face dataset."""
    def go():
        table = hub.dataset_table(str(body.get("repo") or ""), body.get("config"))
        return {**userdata.preview(userdata.stage(table)), "hub": table["hub"]}
    return _user(go)


@app.post("/api/datasets/import")
def dataset_import(body: dict = Body(...)):
    info = _user(lambda: userdata.build(str(body.get("token") or ""), body))
    return {**info.to_json(), "cached": True}


@app.delete("/api/datasets/{dataset_id}")
def dataset_delete(dataset_id: str):
    used = sum(1 for r in runs.list_runs() if r["config"]["dataset"]["id"] == dataset_id)
    _user(lambda: userdata.delete(dataset_id))
    return {"ok": True, "runs_left": used}


@app.post("/api/datasets/compatible")
def datasets_compatible(body: dict = Body(...)):
    """Which datasets a given architecture can be trained on as it stands."""
    arch = body.get("architecture") or {"blocks": []}
    overlay = custom.build_registry_overlay(body.get("custom_blocks"))
    out = []
    for info in datasets.DATASETS.values():
        report = modelmod.inspect(arch, info.input_shape, info.ctx(), overlay)
        out.append({"id": info.id, "ok": report.ok, "params": report.total_params if report.ok else None})
    return out


@app.get("/api/datasets/{dataset_id}/samples")
def dataset_samples(dataset_id: str, n: int = 24, split: str = "test", seed: int = 0, label: int | None = None, seq_len: int | None = None):
    try:
        return predict.samples(dataset_id, min(n, 96), split, seed, label, seq_len)
    except KeyError as e:
        _bad(e, 404)
    except ValueError as e:
        _bad(e)


@app.get("/api/datasets/{dataset_id}/shifts")
def dataset_shifts(dataset_id: str, example: int = 0):
    """The shifted test sets of a dataset, with one example image each."""
    if dataset_id not in datasets.DATASETS:
        _bad(KeyError(f"Unknown dataset '{dataset_id}'"), 404)
    return shifts.describe(dataset_id, example)


@app.post("/api/runs/{run_id}/shifts")
def measure_shifts(run_id: str):
    """Score a trained run on shifted test data (for runs made before this was automatic)."""
    def go():
        run = runs.get(run_id)
        net, _info, _data = predict._load(run_id)
        shifted = shifts.measure(net, run["config"]["dataset"])
        if not shifted:
            raise ValueError("Shifted test data exists only for the image datasets.")
        run = runs.get(run_id)
        run["summary"].update({"shift_acc": shifted["acc"], "shifts": shifted})
        runs.save(run)
        return shifted
    return _model_call(go)


@app.get("/api/templates/{dataset_id}")
def get_templates(dataset_id: str):
    return templates.for_dataset(dataset_id)


@app.post("/api/inspect")
def inspect_arch(body: dict = Body(...)):
    try:
        info = datasets.get_info(body["dataset"], body.get("seq_len"))
    except (KeyError, ValueError) as e:
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
    # A config that did not come from an existing run starts on the current initialization scheme.
    if "init_stream" not in cfg and "environment" not in raw:
        cfg["init_stream"] = 2
    used = {b["type"] for b in cfg["architecture"]["blocks"]}
    embedded = cfg.get("custom_blocks") or {}
    current = custom.sources_for(used)
    cfg["custom_blocks"] = {**{k: v for k, v in embedded.items() if k not in current}, **current}
    info = datasets.info_for(cfg["dataset"])
    overlay = custom.build_registry_overlay(cfg["custom_blocks"])
    report = modelmod.inspect(cfg["architecture"], info.input_shape, info.ctx(), overlay)
    if not report.ok:
        raise ValueError(report.error)
    return cfg


def _start_training(run_id: str, *, resume=False) -> jobs.Job:
    busy = [j for j in jobs.active() if j.meta.get("run_id") == run_id]
    for j in busy:
        # A run that was just paused needs a moment to save its checkpoint and wind down.
        if j.stopping and j.done_event.wait(timeout=30):
            continue
        raise ValueError("This run is already training.")
    return jobs.start("train", lambda job: train_run(job, run_id, resume=resume), meta={"run_id": run_id})


@app.post("/api/train")
def train(body: dict = Body(...)):
    try:
        cfg = _prepare_config(body["config"])
        run = runs.create(cfg, reproduce_of=body.get("reproduce_of"))
        if body.get("worker"):
            # Train on another computer (a cloud GPU worker) and bring the results back.
            worker_id = str(body["worker"])
            remote.find(worker_id)
            job = jobs.start("train", lambda job: remote.run_remote(job, run["id"], worker_id), meta={"run_id": run["id"]})
        else:
            job = _start_training(run["id"])
    except (ValueError, KeyError) as e:
        _bad(e)
    return {"job": job.to_json(), "run": run}


@app.post("/api/runs/{run_id}/resume")
def resume(run_id: str, body: dict = Body(default={})):
    # If the run was paused a moment ago, let it finish saving before reading its state.
    for j in jobs.active():
        if j.meta.get("run_id") == run_id and j.stopping:
            j.done_event.wait(timeout=30)
    try:
        run = runs.get(run_id)
    except KeyError as e:
        _bad(e, 404)
    if not runs.checkpoint_path(run_id).exists():
        _bad(ValueError("This run was stopped before its first training step, so there is nothing to resume. Train it again instead."))
    epochs = int(body.get("epochs") or run["config"]["training"]["epochs"])
    done = run["summary"].get("epochs_done") or 0
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


@app.post("/api/train-seeds")
def train_seeds(body: dict = Body(...)):
    """Train the same config with several consecutive seeds and report the mean and spread."""
    try:
        cfg = _prepare_config(body["config"])
        n = max(2, min(10, int(body.get("n", 3))))
    except (ValueError, KeyError) as e:
        _bad(e)
    job = jobs.start("seeds", lambda job: seeds.run_seeds(job, cfg, n), meta={"name": cfg["name"]})
    return {"job": job.to_json()}


@app.post("/api/train-suite")
def train_suite(body: dict = Body(...)):
    """Train one model on several datasets in turn."""
    try:
        cfg = _prepare_config(body["config"])
        ids = [str(d) for d in body.get("datasets") or [] if str(d) in datasets.DATASETS]
        if len(ids) < 2:
            raise ValueError("Choose at least two datasets.")
        overlay = custom.build_registry_overlay(cfg["custom_blocks"])
        for d in ids:
            info = datasets.get_info(d)
            report = modelmod.inspect(cfg["architecture"], info.input_shape, info.ctx(), overlay)
            if not report.ok:
                raise ValueError(f"This model does not fit {info.name}: {report.error}")
    except (ValueError, KeyError) as e:
        _bad(e)
    job = jobs.start("suite", lambda job: seeds.run_suite(job, cfg, ids), meta={"name": cfg["name"]})
    return {"job": job.to_json()}


# ---------------------------------------------------------------------------
# Frontier Lab: pretrain → reinforcement learning → safety fine-tune → benchmarks
# ---------------------------------------------------------------------------
@app.get("/api/lab/info")
def lab_info():
    return lab.info()


@app.get("/api/lab/pipelines")
def lab_pipelines():
    return lab.pipelines()


@app.post("/api/lab/start")
def lab_start(body: dict = Body(...)):
    try:
        raw = dict(body["config"])
        raw["dataset"] = {"id": "lab_world", "train_subset": None, "seq_len": (raw.get("dataset") or {}).get("seq_len") if (raw.get("dataset") or {}).get("id") == "lab_world" else None}
        cfg = _prepare_config(raw)
        if body.get("options", {}).get("start_from"):
            runs.get(str(body["options"]["start_from"]))
    except (ValueError, KeyError) as e:
        _bad(e)
    job = jobs.start("lab", lambda job: lab.run_pipeline(job, cfg, body.get("options") or {}), meta={"name": cfg["name"]})
    return {"job": job.to_json()}


@app.delete("/api/lab/pipelines/{pipeline_id}")
def lab_delete(pipeline_id: str):
    try:
        lab.delete(pipeline_id)
    except ValueError as e:
        _bad(e)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Settings, workers (training on another computer), updates
# ---------------------------------------------------------------------------
@app.get("/api/settings")
def get_settings():
    return settings.public()


@app.put("/api/settings")
def put_settings(body: dict = Body(...)):
    changes = {}
    if "server_url" in body:
        url = str(body["server_url"] or "").strip().rstrip("/")
        if url and not url.startswith(("http://", "https://")):
            _bad(ValueError("The address must start with http:// or https://"))
        changes["server_url"] = url
    if "player" in body:
        changes["player"] = " ".join(str(body["player"] or "").split())[:40]
    settings.save(changes)
    return settings.public()


@app.get("/api/workers")
def list_workers():
    return [{**{k: w[k] for k in ("id", "name", "url")}, **remote.probe(w)} for w in settings.load()["workers"]]


@app.post("/api/workers")
def add_worker(body: dict = Body(...)):
    try:
        remote.add(str(body.get("name") or ""), str(body.get("url") or ""), str(body.get("token") or ""))
    except remote.WorkerError as e:
        _bad(e)
    return list_workers()


@app.delete("/api/workers/{worker_id}")
def delete_worker(worker_id: str):
    remote.remove(worker_id)
    return list_workers()


@app.post("/api/worker/dataset")
def worker_dataset(body: dict = Body(...)):
    """Receive a dataset from the computer that controls this worker (only when an access token is set)."""
    if not ACCESS_TOKEN:
        _bad(ValueError("This copy is not running as a worker (no access token is set)."), 403)
    try:
        return {"id": remote.receive_dataset(body.get("meta") or {}, str(body.get("data") or ""))}
    except Exception as e:  # noqa: BLE001
        _bad(ValueError(f"Could not store the dataset: {e}"))


@app.get("/api/update")
def check_update(force: bool = False):
    return update.check(force)


# ---------------------------------------------------------------------------
# The shared leaderboard this copy submits to, and feedback
# ---------------------------------------------------------------------------
def _server(method: str, path: str, body=None, timeout: float = 20):
    base = settings.load()["server_url"]
    if not base:
        _bad(ValueError("No shared leaderboard is set up yet. Enter its address first."))
    data = None if body is None else json.dumps(runs.sanitize(body), default=str).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json", "User-Agent": "lego-ai"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail")
        except Exception:  # noqa: BLE001
            detail = None
        _bad(ValueError(detail or f"The leaderboard server answered with an error ({e.code})."), e.code if e.code in (404, 409, 413, 429) else 400)
    except (urllib.error.URLError, OSError, ValueError) as e:
        _bad(ValueError(f"Could not reach the leaderboard at {base}: {getattr(e, 'reason', e)}"), 502)


@app.get("/api/global/info")
def global_info():
    info = _server("GET", "/host/info", timeout=8)
    mine = {rid: s for rid, s in settings.load()["submissions"].items() if s.get("server") == settings.load()["server_url"]}
    return {**info, "player": settings.load()["player"], "mine": mine}


@app.get("/api/global/board")
def global_board(track: str):
    return _server("GET", f"/host/board?track={track}")


@app.post("/api/global/check")
def global_check(body: dict = Body(...)):
    try:
        run = runs.get(str(body.get("run")))
    except KeyError as e:
        _bad(e, 404)
    return _server("POST", "/host/check", {"track": body.get("track"), "config": run["config"]})


@app.post("/api/global/submit")
def global_submit(body: dict = Body(...)):
    try:
        run = runs.get(str(body.get("run")))
    except KeyError as e:
        _bad(e, 404)
    if run["status"] != "completed":
        _bad(ValueError("Only a finished run can be submitted."))
    player = " ".join(str(body.get("player") or settings.load()["player"]).split())[:40]
    settings.save({"player": player})
    out = _server("POST", "/host/submit", {"player": player, "track": body.get("track"), "config": run["config"], "summary": run["summary"], "version": settings.version()})
    subs = settings.load()["submissions"]
    subs[run["id"]] = {"id": out["id"], "track": body.get("track"), "server": settings.load()["server_url"], "at": time.time()}
    settings.save({"submissions": subs})
    return out


@app.get("/api/global/submissions/{sub_id}")
def global_submission(sub_id: str):
    return _server("GET", f"/host/submissions/{sub_id}")


@app.post("/api/global/open")
def global_open(body: dict = Body(...)):
    """Fetch the recipe of an entry on the shared leaderboard."""
    cfg = _server("GET", f"/host/submissions/{body.get('id')}/config")
    code = cfg.get("custom_blocks") or {}
    if code and not body.get("allow_code"):
        return {"needs_confirm": True, "custom_blocks": {k: v.get("source", "") for k, v in code.items()}}
    cfg = cfgmod.normalize(cfg)
    notes = custom.ensure_from_config(cfg.get("custom_blocks"))
    return {"config": cfg, "notes": notes}


@app.post("/api/feedback")
def send_feedback(body: dict = Body(...)):
    """Keep a copy of the message here, and pass it on to the shared server when one is set."""
    text = str(body.get("text") or "").strip()
    if not text:
        _bad(ValueError("Write a message first."))
    s = settings.load()
    entry = {
        "text": text[:8000], "kind": str(body.get("kind") or "idea")[:20], "contact": str(body.get("contact") or "")[:120],
        "player": s["player"], "version": settings.version(), "at": time.time(),
        "context": {"tab": str(body.get("tab") or "")[:30], "platform": cfgmod.environment().get("platform"), "torch": cfgmod.environment().get("torch")} if body.get("include_context", True) else {},
    }
    path = settings.PATH.parent / "feedback.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    sent = False
    if s["server_url"]:
        try:
            _server("POST", "/host/feedback", entry, timeout=10)
            sent = True
        except HTTPException:
            sent = False
    return {"sent": sent, "saved": True, "email": s["feedback_email"] or None}


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------
@app.get("/api/hub/status")
def hub_status():
    return hub.status()


@app.post("/api/hub/login")
def hub_login(body: dict = Body(...)):
    return _user(lambda: hub.login(str(body.get("token") or "")))


@app.post("/api/hub/logout")
def hub_logout():
    return hub.logout()


@app.get("/api/hub/models")
def hub_models(q: str = ""):
    return _user(lambda: hub.search(q))


@app.post("/api/hub/import")
def hub_import(body: dict = Body(...)):
    return _user(lambda: hub.import_model(str(body.get("repo") or ""), bool(body.get("allow_code"))))


@app.post("/api/runs/{run_id}/publish")
def hub_publish(run_id: str, body: dict = Body(...)):
    try:
        runs.get(run_id)
    except KeyError as e:
        _bad(e, 404)
    return _user(lambda: hub.publish(run_id, str(body.get("name") or ""), bool(body.get("private"))))


@app.get("/api/hardware")
def hardware_report():
    return hardware.report()


@app.post("/api/estimate")
def estimate_model(body: dict = Body(...)):
    """Memory and time needed to train a model here and on reference GPUs (pure arithmetic, no model is built)."""
    try:
        info = datasets.get_info(body["dataset"], body.get("seq_len"))
    except (KeyError, ValueError) as e:
        _bad(e)
    n = info.train_size or 0
    if body.get("train_subset"):
        n = min(n, int(body["train_subset"])) if n else int(body["train_subset"])
    return hardware.estimate(
        int(body.get("params", 0)), int(body.get("act_bytes", 0)), int(body.get("flops", 0)),
        int(body.get("batch_size") or 128), str(body.get("optimizer") or "adamw"), n,
        int(body.get("epochs") or 1), str(body.get("device") or "cpu"),
    )


@app.post("/api/custom-blocks/fuse")
def fuse_blocks(body: dict = Body(...)):
    """Turn a run of blocks into one new custom block."""
    try:
        spec = custom.fuse(body.get("name", ""), body.get("color", ""), body.get("blocks") or [])
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=400)
    return {"ok": True, "block": spec.to_json()}


def _zip(raw: dict) -> Response:
    try:
        name, data = export.export_zip(raw)
    except Exception as e:  # noqa: BLE001
        _bad(ValueError(f"Could not export this model: {e}"))
    return Response(data, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.post("/api/export")
def export_draft(body: dict = Body(...)):
    """A model as standalone Python (train.py plus generate.py or predict.py), zipped."""
    try:
        cfg = _prepare_config(body)
    except (ValueError, KeyError, TypeError) as e:
        _bad(e)
    return _zip(cfg)


@app.get("/api/runs/{run_id}/export")
def export_run(run_id: str):
    return _zip(runs.get(run_id)["config"])


# ---------------------------------------------------------------------------
# Advanced: combining models, autopilot, scaling curves
# ---------------------------------------------------------------------------
@app.post("/api/combine")
def combine_runs(body: dict = Body(...)):
    """Make a soup, an ensemble, a gated mixture or a MonoSoup from trained runs."""
    kind, ids, name = body.get("kind"), [str(i) for i in body.get("runs") or []], body.get("name") or None
    try:
        for i in ids:
            runs.get(i)
    except KeyError as e:
        _bad(e, 404)
    fns = {
        "soup": lambda job: combine.soup(job, ids, "uniform", name),
        "greedy": lambda job: combine.soup(job, ids, "greedy", name),
        "ensemble": lambda job: combine.ensemble(job, ids, "average", name),
        "gate": lambda job: combine.ensemble(job, ids, "gate", name),
        "monosoup": lambda job: combine.monosoup(job, ids[0], name),
    }
    if kind not in fns or not ids:
        _bad(ValueError("Choose what to make and which models to make it from."))
    job = jobs.start("combine", fns[kind], meta={"name": name or kind, "combine": kind})
    return {"job": job.to_json()}


@app.post("/api/runs/{run_id}/cook")
def cook_soup(run_id: str, body: dict = Body(default={})):
    """Fine-tune several variations of a trained run, then average them."""
    try:
        run = runs.get(run_id)
        combine.best_state(run_id)
    except KeyError as e:
        _bad(e, 404)
    except ValueError as e:
        _bad(e)
    n, epochs, scale = int(body.get("n") or 4), int(body.get("epochs") or 2), float(body.get("lr_scale") or 0.3)
    job = jobs.start("cook", lambda job: combine.cook(job, run_id, n, epochs, scale), meta={"name": f"Soup from {run['name']}"})
    return {"job": job.to_json()}


@app.post("/api/runs/{run_id}/autopilot")
def start_autopilot(run_id: str, body: dict = Body(default={})):
    try:
        run = runs.get(run_id)
    except KeyError as e:
        _bad(e, 404)
    job = jobs.start("autopilot", lambda job: autopilot.run_autopilot(job, run_id, body), meta={"name": run["name"]})
    return {"job": job.to_json()}


@app.post("/api/scaling/plan")
def scaling_plan(body: dict = Body(...)):
    """The sizes a scaling study would train, with their weight counts (nothing is trained)."""
    try:
        cfg = _prepare_config(body["config"])
        pts = scaling.plan(cfg, body.get("axis") or "width", body.get("factors") or [])
    except (ValueError, KeyError, TypeError) as e:
        _bad(e)
    return [{k: p[k] for k in ("factor", "label", "params", "n_train", "error")} for p in pts]


@app.post("/api/scaling")
def start_scaling(body: dict = Body(...)):
    try:
        cfg = _prepare_config(body["config"])
        axis, factors = body.get("axis") or "width", [float(f) for f in body.get("factors") or []]
        if len([p for p in scaling.plan(cfg, axis, factors) if not p["error"]]) < 2:
            raise ValueError("A scaling curve needs at least two sizes that build correctly.")
    except (ValueError, KeyError, TypeError) as e:
        _bad(e)
    job = jobs.start("scale", lambda job: scaling.run_scaling(job, cfg, axis, factors), meta={"name": cfg["name"]})
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
        info = datasets.info_for(cfg["dataset"])
        overlay = custom.build_registry_overlay(cfg.get("custom_blocks"))
        report = modelmod.inspect(cfg["architecture"], info.input_shape, info.ctx(), overlay)
    except (ValueError, KeyError, TypeError) as e:
        _bad(e)
    return {"config": cfg, "notes": notes, "env_diffs": cfgmod.env_differences(body.get("environment")), "ok": report.ok, "error": report.error}


# ---------------------------------------------------------------------------
# Runs / leaderboard
# ---------------------------------------------------------------------------
def _has_weights(run_id: str) -> bool:
    return runs.checkpoint_path(run_id).exists() or runs.checkpoint_path(run_id, "best").exists()


def _arch_key(c: dict) -> str:
    """Identifies a model design: the same blocks with the same settings, whatever the dataset."""
    import hashlib

    blocks = [[b["type"], b.get("repeat", 1), sorted((b.get("params") or {}).items(), key=lambda kv: kv[0])] for b in c["architecture"]["blocks"]]
    return hashlib.sha1(json.dumps(blocks, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _brief(c: dict | None) -> dict | None:
    """A combine record without its bulky per-layer details, for the run list."""
    return None if not c else {"kind": c["kind"], "method": c.get("method"), "n": len(c.get("members") or [])}


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
            "study": r.get("study"), "reproduce_of": r.get("reproduce_of"), "group": r.get("group"),
            "combine": _brief(r.get("combine")), "scale": r.get("scale"), "variation": r.get("variation"),
            "suite": r.get("suite"), "hub": r.get("hub"), "arch": _arch_key(c), "remote": r.get("remote"), "pipeline": r.get("pipeline"),
            "init_from": c.get("init_from"),
            "training": c["training"], "n_blocks": len(c["architecture"]["blocks"]),
            "block_types": [b["type"] for b in c["architecture"]["blocks"]],
            "has_checkpoint": _has_weights(r["id"]),
        })
    return out


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    try:
        r = runs.get(run_id)
    except KeyError as e:
        _bad(e, 404)
    return {**r, "metrics": runs.metrics(run_id), "has_checkpoint": _has_weights(run_id), "resumable": runs.checkpoint_path(run_id).exists()}


def _model_call(fn):
    """Run a model-inspection function, turning its expected failures into clean HTTP errors."""
    try:
        return fn()
    except (KeyError, FileNotFoundError) as e:
        _bad(e, 404)
    except ValueError as e:
        _bad(e)


@app.post("/api/runs/{run_id}/predict")
def predict_run(run_id: str, body: dict = Body(default={})):
    """Run the trained model on a few random examples."""
    return _model_call(lambda: predict.predict(run_id, n=min(int(body.get("n", 12)), 96), seed=int(body.get("seed", 0)), split=body.get("split", "test")))


@app.post("/api/runs/{run_id}/evaluate")
def evaluate_run(run_id: str, body: dict = Body(default={})):
    """Score the model over (part of) a split: confusion matrix and its most confident mistakes."""
    return _model_call(lambda: predict.evaluate(
        run_id, split=body.get("split", "test"), limit=int(body.get("limit", 2000) or 0),
        true=body.get("true"), pred=body.get("pred"), show=min(int(body.get("show", 48)), 96),
    ))


@app.post("/api/runs/{run_id}/predict-input")
def predict_input(run_id: str, body: dict = Body(...)):
    """The model's answer for something the user drew, uploaded or typed."""
    return _model_call(lambda: predict.predict_input(run_id, image=body.get("image"), text=body.get("text"), vector=body.get("vector")))


@app.post("/api/runs/{run_id}/generate")
def generate_text(run_id: str, body: dict = Body(...)):
    """Continue a prompt with a language model."""
    return _model_call(lambda: predict.generate(
        run_id, str(body.get("prompt", "")), max_new=int(body.get("max_new", 200)),
        temperature=float(body.get("temperature", 0.8)), seed=int(body.get("seed", 0)), top_k=body.get("top_k"),
    ))


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
    pid, state = progress.load()
    return {"profile": pid, "progress": state}


@app.put("/api/progress")
def put_progress(body: dict = Body(...), profile: str | None = None):
    progress.save(body, profile)
    return {"ok": True}


@app.delete("/api/progress")
def reset_progress():
    progress.reset()
    return {"ok": True}


@app.get("/api/profiles")
def list_profiles():
    return progress.list_profiles()


@app.post("/api/profiles")
def create_profile(body: dict = Body(default={})):
    pid = progress.create(body.get("name", ""), demo=bool(body.get("demo")), progress=body.get("progress"))
    return {"id": pid, **progress.list_profiles()}


@app.post("/api/profiles/{pid}/activate")
def activate_profile(pid: str):
    try:
        progress.activate(pid)
    except KeyError as e:
        _bad(e, 404)
    return progress.list_profiles()


@app.patch("/api/profiles/{pid}")
def rename_profile(pid: str, body: dict = Body(...)):
    try:
        progress.rename(pid, body.get("name", ""))
    except KeyError as e:
        _bad(e, 404)
    return progress.list_profiles()


@app.delete("/api/profiles/{pid}")
def delete_profile(pid: str):
    try:
        progress.delete(pid)
    except KeyError as e:
        _bad(e, 404)
    except ValueError as e:
        _bad(e)
    return progress.list_profiles()


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
if FRONTEND.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        # An unknown API address is an error, not a page (an older server would otherwise answer
        # a newer page's requests with HTML).
        if path.startswith(("api/", "host/")):
            raise HTTPException(404, detail="This server does not know that address. It may be older than the page: restart it.")
        f = FRONTEND / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(FRONTEND / "index.html")
