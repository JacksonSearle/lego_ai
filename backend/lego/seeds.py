"""Train one configuration with several random seeds to see how much the result depends on luck.

Each seed is an ordinary, reproducible run. They share a ``group`` record that
holds the mean and spread of their scores once all have finished.
"""
from __future__ import annotations

import copy
import secrets
import statistics
import time

from . import config as cfgmod, runs
from .jobs import Job
from .train import train_run

KEYS = ["best_val_acc", "test_acc", "best_val_loss", "test_loss"]


def stats(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": statistics.fmean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals), "max": max(vals), "n": len(vals),
    }


def run_seeds(job: Job, base_config: dict, n: int):
    base = cfgmod.normalize(base_config)
    first = base["training"]["seed"]
    seeds = [first + i for i in range(n)]
    group_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
    job.meta["group_id"] = group_id
    job.emit({"type": "seeds_start", "group": group_id, "seeds": seeds})
    members: list[str] = []
    try:
        for i, seed in enumerate(seeds):
            job.check_stop()
            cfg = copy.deepcopy(base)
            cfg["training"]["seed"] = seed
            cfg["name"] = f"{base['name']} · seed {seed}"
            run = runs.create(cfg, group={"id": group_id, "index": i, "n": n, "seeds": seeds, "name": base["name"]})
            members.append(run["id"])
            job.emit({"type": "seed_start", "index": i, "seed": seed, "run_id": run["id"], "n": n})
            summary = train_run(job, run["id"], tag={"seed_index": i})
            job.emit({"type": "seed_end", "index": i, "seed": seed, "run_id": run["id"], **{k: summary.get(k) for k in KEYS}})
    finally:
        done = [r for r in (runs.get(m) for m in members) if r["status"] == "completed"]
        agg = {k: stats([r["summary"].get(k) for r in done]) for k in KEYS}
        for m in members:
            r = runs.get(m)
            r["group"]["stats"] = agg
            r["group"]["completed"] = len(done)
            runs.save(r)
        job.emit({"type": "seeds_end", "group": group_id, "stats": agg, "completed": len(done), "n": n})
    return agg


def run_suite(job: Job, base_config: dict, dataset_ids: list[str]):
    """Train one model on several datasets, one after another.

    The runs share a ``suite`` record, and the leaderboard's model view lines them up side by
    side. Text datasets use their default sequence length.
    """
    base = cfgmod.normalize(base_config)
    suite_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
    n = len(dataset_ids)
    job.meta["suite_id"] = suite_id
    job.emit({"type": "seeds_start", "group": suite_id, "seeds": dataset_ids})
    from . import datasets

    for i, ds in enumerate(dataset_ids):
        job.check_stop()
        info = datasets.get_info(ds)
        cfg = copy.deepcopy(base)
        same = ds == base["dataset"]["id"]
        cfg["dataset"] = {"id": ds, "train_subset": base["dataset"].get("train_subset"), "seq_len": base["dataset"].get("seq_len") if same else None}
        if not info.augment:
            cfg["training"]["augment"] = False
        cfg["name"] = base["name"]
        run = runs.create(cfg, suite={"id": suite_id, "index": i, "n": n, "datasets": dataset_ids, "name": base["name"]})
        job.emit({"type": "seed_start", "index": i, "seed": i, "label": info.name, "run_id": run["id"], "n": n})
        try:
            summary = train_run(job, run["id"], tag={"seed_index": i})
        except FloatingPointError as e:  # settings that suit one dataset can diverge on another: carry on
            job.emit({"type": "seed_end", "index": i, "seed": i, "label": info.name, "run_id": run["id"], "error": str(e)})
            continue
        job.emit({"type": "seed_end", "index": i, "seed": i, "label": info.name, "run_id": run["id"], **{k: summary.get(k) for k in KEYS}})
