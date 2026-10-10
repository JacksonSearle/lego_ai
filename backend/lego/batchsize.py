"""Find which batch sizes fit in memory on the chosen device.

For each candidate batch size we estimate memory (parameters + gradients +
optimizer state + activations). Sizes whose estimate fits are then really
tested: one forward/backward/optimizer step, timed.

The real tests run in a separate short-lived process (``probe_worker``).
GPU allocators keep memory cached inside the process that used it, so testing
in the server itself left gigabytes pinned after every check and could bring
the whole machine to a crawl. A child process returns everything when it
exits, cannot disturb running jobs, and can be killed if it hangs or runs out
of memory.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import torch

from . import model as modelmod
from .train import build_model, resolve_device

CANDIDATES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
SAFETY = 0.7  # fraction of free memory we are willing to plan for
QUICK_OK = 0.3  # a single requested size this far under budget is accepted without a test
BACKEND_DIR = Path(__file__).resolve().parent.parent


def _budget(device) -> tuple[int, str]:
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        return int(free * SAFETY), "free GPU memory"
    if device.type == "mps":
        # Unified memory: Metal's recommended working set, minus what's already in use.
        limit = torch.mps.recommended_max_memory()
        used = torch.mps.driver_allocated_memory()
        return int((limit - used) * SAFETY), "Metal recommended working set"
    return int(psutil.virtual_memory().available * SAFETY), "free system RAM"


def plan(config: dict) -> dict:
    """Cheap, in-process part: build the model on the CPU and estimate memory per batch size."""
    t = config["training"]
    device = resolve_device(t["device"])
    net, info, report = build_model(config, seed=0)
    n_params = modelmod.count_params(net)
    act = report.act_bytes_per_sample
    del net
    opt_slots = {"adam": 2, "adamw": 2, "sgd": 1, "rmsprop": 2, "adagrad": 1}.get(t["optimizer"], 2)
    budget, label = _budget(device)
    n_train = info.train_size or 0
    subset = config["dataset"].get("train_subset")
    if subset:
        n_train = min(n_train, subset) if n_train else subset
    return {
        "device": str(device),
        "budget": budget,
        "budget_label": label,
        "params": n_params,
        "fixed": n_params * 4 * (2 + opt_slots),  # weights + grads + optimizer state
        "act_per_sample": act,
        "n_train": n_train,
    }


def estimate(p: dict, bs: int) -> int:
    # Activations are kept for backward, and their gradients need roughly as much again.
    return p["fixed"] + p["act_per_sample"] * bs * 2


def _run_worker(config: dict, sizes: list[int], time_limit: float) -> tuple[dict[int, dict], int | None, str | None]:
    """Test ``sizes`` in a child process. Returns (rows by batch size, size being tested if it died, error)."""
    rows: dict[int, dict] = {}
    testing: int | None = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "lego.probe_worker"],
            cwd=BACKEND_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except OSError as e:
        return rows, None, f"could not start the test process: {e}"
    killed = threading.Event()

    def kill():
        killed.set()
        proc.kill()

    # Generous allowance for starting Python and loading the dataset, then a hard stop.
    timer = threading.Timer(time_limit + 45, kill)
    timer.start()
    try:
        proc.stdin.write(json.dumps({"config": config, "sizes": sizes, "time_limit": time_limit}))
        proc.stdin.close()
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "testing" in msg:
                testing = msg["testing"]
            elif "batch_size" in msg:
                rows[msg["batch_size"]] = msg
                testing = None
        code = proc.wait()
        err = None
        if killed.is_set():
            err = "the test took too long and was stopped"
        elif code != 0:
            tail = (proc.stderr.read() or "").strip().splitlines()[-1:] or [""]
            err = f"the test process stopped unexpectedly ({tail[0][:160] or f'exit code {code}'})"
        return rows, testing, err
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()


def probe(config: dict, requested: int | None = None, time_limit: float = 20.0, only_requested: bool = False) -> dict:
    p = plan(config)
    budget = p["budget"]
    max_n = p["n_train"] or 10**9
    candidates = sorted({c for c in CANDIDATES if c <= max_n} | ({requested} if requested else set()))
    if only_requested and requested:
        candidates = [requested]

    to_test = [bs for bs in candidates if estimate(p, bs) <= budget]
    # Starting a test process costs a few seconds; skip it when a single size is nowhere near the limit.
    quick = only_requested and len(to_test) == 1 and estimate(p, to_test[0]) <= budget * QUICK_OK
    rows, died_at, err = ({}, None, None) if (quick or not to_test) else _run_worker(config, to_test, time_limit)

    oom_from = min(
        [bs for bs, r in rows.items() if r.get("fits") is False] + ([died_at] if died_at else []),
        default=None,
    )
    results = []
    for bs in candidates:
        est = estimate(p, bs)
        row = {"batch_size": bs, "estimated_mb": est / 2**20, "fits": None, "tested": False}
        if est > budget:
            row.update({"fits": False, "reason": f"estimate exceeds {p['budget_label']}"})
        elif bs in rows:
            row.update(rows[bs])
            if row.get("step_ms") and p["n_train"]:
                row["epoch_s"] = row["step_ms"] / 1000 * (p["n_train"] / bs)
        elif oom_from is not None and bs >= oom_from:
            row.update({"fits": False, "reason": "out of memory (crashed the test)" if bs == died_at else "out of memory"})
        else:
            row["fits"] = True
            row["reason"] = "estimated (well within memory)" if quick else f"estimated ({err or 'not tested: larger sizes were too slow to time'})"
        results.append(row)

    fitting = [r for r in results if r["fits"]]
    tested = [r for r in fitting if r.get("samples_per_s")]
    req_row = next((r for r in results if r["batch_size"] == requested), None)
    return {
        "device": p["device"],
        "budget_mb": budget / 2**20,
        "budget_label": p["budget_label"],
        "params": p["params"],
        "fixed_mb": p["fixed"] / 2**20,
        "activation_kb_per_sample": p["act_per_sample"] / 1024,
        "train_examples": p["n_train"],
        "results": results,
        "largest_fitting": max((r["batch_size"] for r in fitting), default=None),
        "fastest": max(tested, key=lambda r: r["samples_per_s"])["batch_size"] if tested else None,
        "requested": requested,
        "requested_fits": None if req_row is None else req_row["fits"],
        "note": err,
    }
