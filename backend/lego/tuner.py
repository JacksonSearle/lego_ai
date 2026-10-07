"""Optuna hyperparameter search.

Every trial is a normal, fully reproducible run (saved with its exact sampled
hyperparameters), tagged with the study it belongs to. Unpromising trials are
pruned after each epoch with a median pruner.
"""
from __future__ import annotations

import copy
import json
import secrets
import time
from pathlib import Path

import optuna

from . import config as cfgmod, runs
from .jobs import Job, Stopped
from .train import Pruned, train_run

STUDIES_DIR = Path(__file__).resolve().parent.parent / "data" / "studies"
optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNABLE = {h["name"]: h for h in cfgmod.HYPERPARAMS if "tune" in h}


def default_search_space() -> dict:
    return {name: {**h["tune"], "enabled": name in ("lr", "weight_decay", "dropout")} for name, h in TUNABLE.items()}


def _suggest(trial: optuna.Trial, name: str, spec: dict):
    kind = spec["type"]
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log")))
    low, high = float(spec["low"]), float(spec["high"])
    if spec.get("log") and low <= 0:
        raise ValueError(f"{name}: log scale needs a positive lower bound")
    return trial.suggest_float(name, low, high, log=bool(spec.get("log")))


def run_study(job: Job, base_config: dict, opts: dict):
    space = {k: v for k, v in (opts.get("space") or {}).items() if v.get("enabled", True) and k in TUNABLE}
    if not space:
        raise ValueError("Enable at least one hyperparameter to tune.")
    n_trials = int(opts.get("n_trials", 10))
    metric = opts.get("metric", "val_acc")
    direction = "minimize" if metric == "val_loss" else "maximize"
    base = cfgmod.normalize(base_config)
    if opts.get("epochs"):
        base["training"]["epochs"] = int(opts["epochs"])
    seed = base["training"]["seed"]

    study_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1) if opts.get("pruning", True) else optuna.pruners.NopPruner(),
    )
    record = {
        "id": study_id, "name": base["name"], "dataset": base["dataset"]["id"], "metric": metric, "direction": direction,
        "space": space, "n_trials": n_trials, "trials": [], "best": None, "created_at": time.time(), "status": "running",
    }
    job.meta["study_id"] = study_id
    job.emit({"type": "study_start", "study": record})

    def save():
        STUDIES_DIR.mkdir(parents=True, exist_ok=True)
        (STUDIES_DIR / f"{study_id}.json").write_text(json.dumps(record, indent=2))

    save()
    try:
        for i in range(n_trials):
            job.check_stop()
            trial = study.ask()
            params = {k: _suggest(trial, k, v) for k, v in space.items()}
            cfg = copy.deepcopy(base)
            cfg["training"].update(params)
            cfg["name"] = f"{base['name']} · trial {i + 1}"
            run = runs.create(cfg, study={"id": study_id, "trial": i + 1, "params": params, "best": False})
            job.emit({"type": "trial_start", "trial": i + 1, "params": params, "run_id": run["id"]})

            def on_epoch(epoch, ep, trial=trial):
                value = ep[metric]
                trial.report(value, epoch)
                if trial.should_prune():
                    raise Pruned()

            state, value = "complete", None
            try:
                summary = train_run(job, run["id"], tag={"trial": i + 1}, on_epoch=on_epoch)
                value = summary["best_val_acc"] if metric == "val_acc" else summary["best_val_loss"]
                study.tell(trial, value)
            except Pruned:
                state = "pruned"
                summary = runs.get(run["id"])["summary"]
                value = summary.get("best_val_acc" if metric == "val_acc" else "best_val_loss")
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            except Stopped:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                raise
            except Exception as e:  # noqa: BLE001 — a bad trial (e.g. NaN loss) shouldn't kill the study
                state = "failed"
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                job.emit({"type": "trial_error", "trial": i + 1, "message": str(e)})

            entry = {"trial": i + 1, "run_id": run["id"], "params": params, "value": value, "state": state}
            record["trials"].append(entry)
            done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if done:
                bt = study.best_trial
                best_entry = record["trials"][bt.number]
                record["best"] = {"trial": bt.number + 1, "value": bt.value, "params": bt.params, "run_id": best_entry["run_id"]}
            save()
            job.emit({"type": "trial_end", **entry, "best": record["best"]})
    finally:
        record["status"] = "stopped" if job.stopping else "completed"
        save()
        if record["best"]:
            for t in record["trials"]:
                try:
                    r = runs.get(t["run_id"])
                    r["study"]["best"] = t["run_id"] == record["best"]["run_id"]
                    runs.save(r)
                except KeyError:
                    pass
        job.emit({"type": "study_end", "study": record})
    return record


def list_studies() -> list[dict]:
    if not STUDIES_DIR.exists():
        return []
    out = [json.loads(p.read_text()) for p in STUDIES_DIR.glob("*.json")]
    return sorted(out, key=lambda s: s["created_at"], reverse=True)

