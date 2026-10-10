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
