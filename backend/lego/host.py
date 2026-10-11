"""Leaderboard host: a shared leaderboard that other copies of Lego AI submit to.

Enabled with ``LEGO_HOST=1``. The design rests on one rule: **the host never trusts a
score it is sent.** A submission is a run *config* (the complete recipe). The host checks
the recipe against the track's limits, trains it itself from scratch on its own hardware,
and lists the score it measured. Reproducibility comes from the config; fairness from
everyone being measured on the same machine.

    tracks        leaderboards with limits (dataset, weights, epochs, training examples, seconds)
    submissions   a queue the host owner works through: pending → verified / rejected
    feedback      messages from the app's feedback button

Public endpoints live under ``/host``. Admin endpoints need either ``LEGO_ADMIN_TOKEN``
(sent as ``X-Admin-Token``) or the host owner's own access token.
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Request

from . import config as cfgmod, custom, datasets, jobs, model as modelmod, runs, settings
from .train import Pruned, train_run

ENABLED = os.environ.get("LEGO_HOST") == "1"
ADMIN_TOKEN = os.environ.get("LEGO_ADMIN_TOKEN", "")
NAME = os.environ.get("LEGO_HOST_NAME", "Lego AI leaderboard")
DIR = Path(__file__).resolve().parent.parent / "data" / "host"
SUBS = DIR / "submissions"
MAX_BODY = 1_500_000  # bytes of JSON per submission
MAX_PENDING_PER_PLAYER = 10
_lock = threading.Lock()

router = APIRouter(prefix="/host")

DEFAULT_TRACKS = [
    {
        "id": "mnist-sprint", "name": "MNIST sprint", "dataset": "mnist",
        "description": "The best digit reader you can train in three passes over the data, with at most 100,000 weights.",
        "max_params": 100_000, "max_epochs": 3, "max_train_examples": None, "max_seconds": 180, "allow_custom_blocks": False,
    },
    {
        "id": "mnist-1k", "name": "MNIST from 1,000 examples", "dataset": "mnist",
        "description": "Only 1,000 training digits. What matters here is learning a lot from a little.",
        "max_params": 500_000, "max_epochs": 30, "max_train_examples": 1000, "max_seconds": 180, "allow_custom_blocks": False,
    },
    {
        "id": "fashion-small", "name": "Fashion-MNIST, small models", "dataset": "fashion_mnist",
        "description": "Clothing in five epochs with at most 250,000 weights.",
        "max_params": 250_000, "max_epochs": 5, "max_train_examples": None, "max_seconds": 400, "allow_custom_blocks": False,
    },
    {
        "id": "news-quick", "name": "AG News quick read", "dataset": "ag_news",
        "description": "Sort news into four topics from 20,000 examples in two epochs.",
        "max_params": 3_000_000, "max_epochs": 2, "max_train_examples": 20000, "max_seconds": 400, "allow_custom_blocks": False,
    },
]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def tracks() -> list[dict]:
    path = DIR / "tracks.json"
    if not path.exists():
        DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DEFAULT_TRACKS, indent=2))
    return json.loads(path.read_text())


def _track(track_id: str) -> dict:
    for t in tracks():
        if t["id"] == track_id:
            return t
    raise HTTPException(404, detail="No such track on this leaderboard.")


def _sub_path(sub_id: str) -> Path:
    if not sub_id.isalnum():
        raise HTTPException(404, detail="No such submission.")
    return SUBS / f"{sub_id}.json"


def _load(sub_id: str) -> dict:
    p = _sub_path(sub_id)
    if not p.exists():
        raise HTTPException(404, detail="No such submission.")
    return json.loads(p.read_text())


def _store(sub: dict):
    with _lock:
        SUBS.mkdir(parents=True, exist_ok=True)
        tmp = _sub_path(sub["id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(runs.sanitize(sub), indent=2))
        tmp.replace(_sub_path(sub["id"]))


def _all() -> list[dict]:
    if not SUBS.exists():
        return []
    out = []
    for p in SUBS.glob("*.json"):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda s: s["submitted_at"])


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
def check(track: dict, cfg: dict) -> tuple[list[str], dict]:
    """Does a run config obey a track's limits? Returns (problems, facts about the config)."""
    problems = []
    cfg = cfgmod.normalize(cfg)
    t = cfg["training"]
    facts: dict = {"params": None, "epochs": t["epochs"], "train_examples": None}
    if cfg["dataset"]["id"] != track["dataset"]:
        name = datasets.DATASETS[track["dataset"]].name if track["dataset"] in datasets.DATASETS else track["dataset"]
        return [f"This track is for {name}; the model was trained on something else."], facts
    info = datasets.info_for(cfg["dataset"])
    if cfg.get("init_from"):
        problems.append("The model was fine-tuned from another run's weights. A submission must train from scratch so it can be reproduced.")
    code = cfg.get("custom_blocks") or {}
    if code and not track.get("allow_custom_blocks"):
        problems.append("This track does not accept custom blocks (they are code the host would have to run). Use built-in blocks, or unfuse fused ones.")
    overlay = custom.build_registry_overlay(code) if code and track.get("allow_custom_blocks") else {}
    if not problems:
        report = modelmod.inspect(cfg["architecture"], info.input_shape, info.ctx(), overlay)
        if not report.ok:
            problems.append(f"The model does not build: {report.error}")
        else:
            facts["params"] = report.total_params
            if track.get("max_params") and report.total_params > track["max_params"]:
                problems.append(f"Too many weights: {report.total_params:,}, and the limit is {track['max_params']:,}.")
    if track.get("max_epochs") and t["epochs"] > track["max_epochs"]:
        problems.append(f"Too many epochs: {t['epochs']}, and the limit is {track['max_epochs']}.")
    n = min(info.train_size or 0, cfg["dataset"].get("train_subset") or info.train_size or 0)
    facts["train_examples"] = n
    if track.get("max_train_examples") and n > track["max_train_examples"]:
        problems.append(f"Too much training data: {n:,} examples, and the limit is {track['max_train_examples']:,}. Set the training subset on the Train page.")
    return problems, facts


def _admin(request: Request):
    # Either the admin token, or the host owner's own access token (the cookie set by opening
    # the app with ?token=...). Where a request comes from proves nothing: behind a tunnel or
    # a proxy every request looks local.
    given = request.headers.get("x-admin-token", "")
    if ADMIN_TOKEN and hmac.compare_digest(given, ADMIN_TOKEN):
        return
    own = request.cookies.get("lego_token", "") or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if settings.ACCESS_TOKEN and hmac.compare_digest(own, settings.ACCESS_TOKEN):
        return
    raise HTTPException(401, detail="Wrong or missing admin token.")


def _on():
    if not ENABLED:
        raise HTTPException(404, detail="This copy of Lego AI is not hosting a leaderboard.")


def _brief(s: dict) -> dict:
    r = s.get("result") or {}
    return {
        "id": s["id"], "player": s["player"], "name": s["name"], "track": s["track"], "status": s["status"], "reason": s.get("reason"),
        "submitted_at": s["submitted_at"], "params": s["facts"].get("params"), "epochs": s["facts"].get("epochs"),
        "train_examples": s["facts"].get("train_examples"), "block_types": [b["type"] for b in s["config"]["architecture"]["blocks"]],
        # The score that counts is the one the host measured; the claim is only shown beside it.
        "val_acc": r.get("best_val_acc"), "test_acc": r.get("test_acc"), "seconds": r.get("duration"),
        "claimed_test_acc": (s.get("claimed") or {}).get("test_acc"), "match": s.get("match"),
    }


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------
@router.get("/info")
def info():
    _on()
    return {"name": NAME, "version": settings.version(), "tracks": tracks(), "queue": sum(1 for s in _all() if s["status"] == "pending")}


@router.get("/board")
def board(track: str):
    _on()
    _track(track)
    subs = [s for s in _all() if s["track"] == track]
    done = sorted((s for s in subs if s["status"] == "verified"), key=lambda s: (-(s["result"].get("test_acc") or 0), s["facts"].get("params") or 0, s["submitted_at"]))
    waiting = [s for s in subs if s["status"] in ("pending", "verifying")]
    return {"entries": [_brief(s) for s in done], "waiting": [_brief(s) for s in waiting]}


@router.post("/check")
def check_config(body: dict = Body(...)):
    _on()
    try:
        problems, facts = check(_track(str(body.get("track"))), body.get("config") or {})
    except (ValueError, KeyError) as e:
        raise HTTPException(400, detail=f"That is not a valid run config: {e}") from None
    return {"ok": not problems, "problems": problems, "facts": facts}


@router.post("/submit")
async def submit(request: Request):
    _on()
    raw = await request.body()
    if len(raw) > MAX_BODY:
        raise HTTPException(413, detail="That submission is too large.")
    try:
        body = json.loads(raw)
        cfg = cfgmod.normalize(body.get("config") or {})
    except ValueError as e:
        raise HTTPException(400, detail=f"That is not a valid run config: {e}") from None
    player = " ".join(str(body.get("player") or "").split())[:40]
    if not player:
        raise HTTPException(400, detail="Choose a name to appear on the leaderboard.")
    track = _track(str(body.get("track")))
    problems, facts = check(track, cfg)
    if problems:
        raise HTTPException(400, detail="This run breaks the track's rules: " + " ".join(problems))
    mine = [s for s in _all() if s["player"].lower() == player.lower() and s["status"] == "pending"]
    if len(mine) >= MAX_PENDING_PER_PLAYER:
        raise HTTPException(429, detail="You already have several submissions waiting. Wait for those to be checked first.")
    cfg.pop("environment", None)
    key = json.dumps([cfg["architecture"], cfg["training"], cfg["dataset"]], sort_keys=True)
    if any(s["track"] == track["id"] and s["status"] != "rejected" and s.get("key") == key for s in _all()):
        raise HTTPException(409, detail="This exact recipe has already been submitted to this track.")
    claimed = {k: (body.get("summary") or {}).get(k) for k in ("best_val_acc", "test_acc", "test_loss", "duration")}
    sub = {
        "id": secrets.token_hex(6), "player": player, "name": str(cfg["name"])[:80], "track": track["id"], "status": "pending",
        "submitted_at": time.time(), "config": cfg, "claimed": claimed, "facts": facts, "key": key, "result": None, "match": None, "reason": None,
        "client": {"version": str(body.get("version") or "")[:20], "ip": request.client.host if request.client else None},
    }
    _store(sub)
    return {"id": sub["id"], "status": "pending", "ahead": sum(1 for s in _all() if s["status"] == "pending") - 1}


@router.get("/submissions/{sub_id}")
def submission(sub_id: str):
    _on()
    return _brief(_load(sub_id))


@router.get("/submissions/{sub_id}/config")
def submission_config(sub_id: str):
    """The recipe of a checked entry, so anyone can open it, learn from it and retrain it."""
    _on()
    s = _load(sub_id)
    if s["status"] != "verified":
        raise HTTPException(404, detail="Only checked entries can be downloaded.")
    return s["config"]


@router.post("/feedback")
async def feedback(request: Request):
    _on()
    raw = await request.body()
    if len(raw) > 60_000:
        raise HTTPException(413, detail="That message is too long.")
    body = json.loads(raw)
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, detail="The message is empty.")
    entry = {
        "id": secrets.token_hex(5), "at": time.time(), "text": text[:8000], "player": str(body.get("player") or "")[:40],
        "contact": str(body.get("contact") or "")[:120], "kind": str(body.get("kind") or "idea")[:20],
        "version": str(body.get("version") or "")[:20], "context": body.get("context") if isinstance(body.get("context"), dict) else {},
    }
    with _lock:
        DIR.mkdir(parents=True, exist_ok=True)
        with open(DIR / "feedback.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Checking a submission: train it here and record what this machine measures
# ---------------------------------------------------------------------------
def verify(job: jobs.Job, sub_id: str, allow_code: bool = False):
    sub = _load(sub_id)
    track = _track(sub["track"])
    cfg = cfgmod.normalize(sub["config"])
    problems, _ = check(track, cfg)
    if problems:
        sub.update(status="rejected", reason=" ".join(problems))
        _store(sub)
        return sub
    if cfg.get("custom_blocks") and not allow_code:
        raise ValueError("This submission contains custom block code. Read it, then verify with code allowed.")
    cfg["training"]["device"] = "cpu"  # every entry is measured on the same device, and exactly reproducibly
    cfg["name"] = f"{sub['player']}: {sub['name']}"
    sub["status"] = "verifying"
    _store(sub)
    run = runs.create(cfg, submission={"id": sub_id, "track": track["id"], "player": sub["player"]}, demo=True)
    limit = track.get("max_seconds")
    started = time.time()

    def on_epoch(_epoch, _ep):
        if limit and time.time() - started > limit:
            raise Pruned()

    try:
        summary = train_run(job, run["id"], on_epoch=on_epoch)
    except Pruned:
        sub.update(status="rejected", reason=f"Took longer than the track's limit of {limit} seconds on the host's computer.")
        _store(sub)
        runs.delete(run["id"])
        return sub
    except jobs.Stopped:
        sub["status"] = "pending"
        _store(sub)
        runs.delete(run["id"])
        raise
    except Exception as e:  # noqa: BLE001
        sub.update(status="rejected", reason=f"Training failed on the host: {type(e).__name__}: {e}"[:300])
        _store(sub)
        runs.delete(run["id"])
        return sub
    if limit and summary.get("duration", 0) > limit:
        sub.update(status="rejected", reason=f"Took {summary['duration']:.0f} seconds on the host's computer; the track's limit is {limit}.")
        _store(sub)
        runs.delete(run["id"])
        return sub
    keep = ("params", "epochs_done", "best_val_acc", "best_val_loss", "test_acc", "test_loss", "duration", "train_examples", "shift_acc")
    sub["result"] = {k: summary.get(k) for k in keep}
    claim = sub["claimed"].get("test_acc")
    got = summary.get("test_acc")
    sub["match"] = None if claim is None or got is None else "exact" if claim == got else "close" if abs(claim - got) <= 0.01 else "different"
    sub.update(status="verified", verified_at=time.time(), host_run=run["id"], reason=None)
    _store(sub)
    return sub


def _busy() -> bool:
    return any(j.kind in ("verify", "train", "tune", "seeds", "suite", "scale", "cook", "autopilot") for j in jobs.active())


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------
@router.get("/admin/queue")
def admin_queue(request: Request):
    _on()
    _admin(request)
    out = []
    for s in reversed(_all()):
        out.append({**_brief(s), "claimed": s["claimed"], "result": s["result"], "client": s["client"], "custom_blocks": {k: v.get("source", "") for k, v in (s["config"].get("custom_blocks") or {}).items()}})
    return {"submissions": out, "busy": _busy()}


@router.post("/admin/submissions/{sub_id}/verify")
def admin_verify(sub_id: str, request: Request, body: dict = Body(default={})):
    _on()
    _admin(request)
    _load(sub_id)
    if _busy():
        raise HTTPException(409, detail="Another training job is running on the host. One at a time keeps the timing fair.")
    job = jobs.start("verify", lambda job: verify(job, sub_id, bool(body.get("allow_code"))), meta={"name": f"Checking {sub_id}", "submission": sub_id})
    return {"job": job.to_json()}


@router.post("/admin/verify-all")
def admin_verify_all(request: Request):
    """Work through every waiting submission that contains no custom code, oldest first."""
    _on()
    _admin(request)
    if _busy():
        raise HTTPException(409, detail="Another training job is running on the host.")
    todo = [s["id"] for s in _all() if s["status"] == "pending" and not s["config"].get("custom_blocks")]

    def go(job):
        for sid in todo:
            job.check_stop()
            job.emit({"type": "status", "message": f"Checking {sid}…"})
            verify(job, sid)

    job = jobs.start("verify", go, meta={"name": f"Checking {len(todo)} submissions"})
    return {"job": job.to_json(), "count": len(todo)}


@router.post("/admin/submissions/{sub_id}/reject")
def admin_reject(sub_id: str, request: Request, body: dict = Body(default={})):
    _on()
    _admin(request)
    s = _load(sub_id)
    s.update(status="rejected", reason=str(body.get("reason") or "Removed by the host.")[:300])
    _store(s)
    return _brief(s)


@router.delete("/admin/submissions/{sub_id}")
def admin_delete(sub_id: str, request: Request):
    _on()
    _admin(request)
    s = _load(sub_id)
    if s.get("host_run"):
        runs.delete(s["host_run"])
    _sub_path(sub_id).unlink(missing_ok=True)
    return {"ok": True}


@router.put("/admin/tracks")
def admin_tracks(request: Request, body: list = Body(...)):
    _on()
    _admin(request)
    clean, seen = [], set()
    for t in body:
        tid = "".join(c for c in str(t.get("id") or "").lower() if c.isalnum() or c == "-")[:40]
        if not tid or tid in seen or t.get("dataset") not in datasets.DATASETS:
            raise HTTPException(400, detail="Every track needs a unique id and a dataset this host has.")
        seen.add(tid)
        num = lambda k: int(t[k]) if t.get(k) not in (None, "", 0) else None  # noqa: E731
        clean.append({
            "id": tid, "name": str(t.get("name") or tid)[:60], "dataset": t["dataset"], "description": str(t.get("description") or "")[:300],
            "max_params": num("max_params"), "max_epochs": num("max_epochs"), "max_train_examples": num("max_train_examples"),
            "max_seconds": num("max_seconds"), "allow_custom_blocks": bool(t.get("allow_custom_blocks")),
        })
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / "tracks.json").write_text(json.dumps(clean, indent=2))
    return clean


@router.get("/admin/feedback")
def admin_feedback(request: Request):
    _on()
    _admin(request)
    path = DIR / "feedback.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()][::-1]
