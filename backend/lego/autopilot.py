"""Autopilot: start from a trained model and keep trying small changes, keeping what helps.

Where Optuna samples a search space you describe, the autopilot does a local search around a
model that already works. Each round it makes one or two small changes to the current best
(a hyperparameter nudged, a layer widened, a block repeated once more), trains the result from
scratch, and adopts it if validation accuracy improves. Rounds that fall clearly behind the
best run at the same epoch are cut short.

Every round is an ordinary reproducible run, tagged with the study it belongs to. The events
match an Optuna study's, so the Train page shows both the same way.
"""
from __future__ import annotations

import copy
import json
import random
import time

from . import config as cfgmod, custom, datasets, model as modelmod, runs
from .jobs import Job, Stopped
from .registry import REGISTRY
from .scaling import WIDTH_PARAMS
from .train import Pruned, train_run
from .tuner import STUDIES_DIR

FIXED_BLOCKS = {"classifier", "flatten", "embedding", "patch_embed", "image_to_sequence", "positional_encoding"}
MAX_GROWTH = 4.0  # never let the model grow beyond this many times the starting size


def _fmt(v) -> str:
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def _mutate_training(t: dict, rng: random.Random) -> str | None:
    kind = rng.choice(["lr", "lr", "weight_decay", "batch_size", "dropout", "label_smoothing", "scheduler"])
    old = t.get(kind)
    if kind == "lr":
        new = float(f"{old * rng.choice([0.5, 0.7, 1.4, 2.0]):.3g}")
    elif kind == "weight_decay":
        new = 0.01 if not old else float(f"{old * rng.choice([0.3, 3.0]):.3g}")
    elif kind == "batch_size":
        new = int(min(512, max(16, old * rng.choice([0.5, 2]))))
    elif kind == "dropout":
        new = round(min(0.5, max(0.0, (old or 0.0) + rng.choice([-0.1, 0.1]))), 2)
    elif kind == "label_smoothing":
        new = 0.1 if not old else 0.0
    else:
        new = rng.choice([s for s in ("none", "cosine", "onecycle") if s != old])
    if new == old:
        return None
    t[kind] = new
    return f"{kind.replace('_', ' ')} {_fmt(old) if old is not None else 'off'} → {_fmt(new)}"


def _mutate_arch(blocks: list[dict], specs: dict, rng: random.Random) -> str | None:
    idx = [i for i, b in enumerate(blocks) if b["type"] not in FIXED_BLOCKS]
    if not idx:
        return None
    i = rng.choice(idx)
    b = blocks[i]
    spec = specs.get(b["type"])
    name = spec.name if spec else b["type"]
    widths = [p for p in (spec.params if spec else []) if p.name in WIDTH_PARAMS and p.type == "int" and b.get("params", {}).get(p.name, p.default)]
    if widths and rng.random() < 0.65:
        p = rng.choice(widths)
        old = int(b["params"].get(p.name, p.default))
        heads = int(b["params"].get("heads", 1) or 1)
        new = max(heads, int(round(old * rng.choice([0.5, 0.75, 1.5, 2.0]) / heads)) * heads)
        if new == old:
            return None
        b.setdefault("params", {})[p.name] = new
        return f"{name} {p.label.lower() if p.label else p.name} {old} → {new}"
    old = int(b.get("repeat") or 1)
    new = old + rng.choice([-1, 1, 1])
    if new < 1 or new > 16:
        return None
    b["repeat"] = new
    return f"{name} ×{old} → ×{new}"


def propose(best: dict, rng: random.Random, base_params: int, tune_arch: bool) -> tuple[dict, list[str]] | None:
    """A valid variation of ``best`` and a description of what changed (None if no luck)."""
    info = datasets.info_for(best["dataset"])
    overlay = custom.build_registry_overlay(best.get("custom_blocks"))
    specs = {**REGISTRY, **overlay}
    for _ in range(20):
        cfg = copy.deepcopy(best)
        changes = []
        for _ in range(rng.choice([1, 1, 2])):
            if tune_arch and rng.random() < 0.5:
                c = _mutate_arch(cfg["architecture"]["blocks"], specs, rng)
            else:
                c = _mutate_training(cfg["training"], rng)
            if c:
                changes.append(c)
        if not changes:
            continue
        report = modelmod.inspect(cfg["architecture"], info.input_shape, info.ctx(), overlay)
        if report.ok and report.total_params <= base_params * MAX_GROWTH:
            return cfg, changes
    return None


def run_autopilot(job: Job, base_run_id: str, opts: dict):
    base_run = runs.get(base_run_id)
    if base_run["status"] != "completed" or base_run["summary"].get("best_val_acc") is None:
        raise ValueError("The autopilot needs a model that has finished training to start from.")
    if base_run.get("combine"):
        raise ValueError("The autopilot starts from an ordinary trained model, not a soup or an ensemble.")
    rounds = max(1, min(int(opts.get("rounds", 8)), 100))
    tune_arch = bool(opts.get("architecture", True))
    best = cfgmod.normalize(base_run["config"])
    best.pop("environment", None)
    best.pop("init_from", None)
    base_name = best["name"]
    base_params = base_run["summary"].get("params") or 1
    rng = random.Random(best["training"]["seed"] * 1000 + rounds)
    score = (base_run["summary"]["best_val_acc"], -(base_run["summary"].get("best_val_loss") or 0.0))
    best_id = base_run_id
    curve = [e["val_acc"] for e in runs.metrics(base_run_id)["epochs"]]

    study_id = runs.new_id()
    record = {
        "id": study_id, "kind": "autopilot", "name": base_name, "dataset": best["dataset"]["id"], "metric": "val_acc", "direction": "maximize",
        "space": {}, "n_trials": rounds, "trials": [], "created_at": time.time(), "status": "running", "base_run": base_run_id,
        "best": {"trial": 0, "value": score[0], "params": {"change": "starting model"}, "run_id": base_run_id},
    }
    job.meta.update({"study_id": study_id, "name": base_name})
    job.emit({"type": "study_start", "study": record})

    def save():
        STUDIES_DIR.mkdir(parents=True, exist_ok=True)
        (STUDIES_DIR / f"{study_id}.json").write_text(json.dumps(record, indent=2))

    save()
    tried: set[str] = set()
    try:
        for i in range(1, rounds + 1):
            job.check_stop()
            got = None
            for _ in range(10):
                got = propose(best, rng, base_params, tune_arch)
                key = json.dumps([got[0]["architecture"], got[0]["training"]], sort_keys=True) if got else None
                if got and key not in tried:
                    tried.add(key)
                    break
                got = None
            if not got:
                break
            cfg, changes = got
            params = {"change": "; ".join(changes)}
            cfg["name"] = f"{base_name} · autopilot {i}"
            run = runs.create(cfg, study={"id": study_id, "trial": i, "params": params, "best": False, "kind": "autopilot"})
            job.emit({"type": "trial_start", "trial": i, "params": params, "run_id": run["id"]})

            def on_epoch(epoch, ep, curve=curve):
                # Give up on a round that is clearly behind where the best run was at this point.
                if epoch >= 2 and epoch <= len(curve) and ep["val_acc"] < curve[epoch - 1] - 0.05:
                    raise Pruned()

            state, value = "complete", None
            try:
                s = train_run(job, run["id"], tag={"trial": i}, on_epoch=on_epoch)
                value = s["best_val_acc"]
                new = (value, -(s.get("best_val_loss") or 0.0))
                if new > score:
                    score, best_id = new, run["id"]
                    best = copy.deepcopy(cfg)
                    best["name"] = base_name
                    curve = [e["val_acc"] for e in runs.metrics(run["id"])["epochs"]]
                    record["best"] = {"trial": i, "value": value, "params": params, "run_id": run["id"]}
                    params["kept"] = "yes"
            except Pruned:
                state = "pruned"
                value = runs.get(run["id"])["summary"].get("best_val_acc")
            except Stopped:
                raise
            except Exception as e:  # noqa: BLE001 - one bad variation (NaN loss, out of memory) should not end the search
                state = "failed"
                job.emit({"type": "trial_error", "trial": i, "message": str(e)})
            entry = {"trial": i, "run_id": run["id"], "params": params, "value": value, "state": state}
            record["trials"].append(entry)
            save()
            job.emit({"type": "trial_end", **entry, "best": record["best"]})
    finally:
        record["status"] = "stopped" if job.stopping else "completed"
        save()
        for t in record["trials"]:
            try:
                r = runs.get(t["run_id"])
                r["study"]["best"] = t["run_id"] == best_id
                runs.save(r)
            except KeyError:
                pass
        job.emit({"type": "study_end", "study": record})
    return record
