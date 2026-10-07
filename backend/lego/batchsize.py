"""Find which batch sizes fit in memory on the chosen device.

For each candidate batch size we estimate memory (parameters + gradients +
optimizer state + activations) and, if the estimate fits, actually run one
forward/backward/optimizer step to confirm and time it. GPUs report real
allocator usage; on CPU we compare the estimate against free system RAM.
"""
from __future__ import annotations

import gc
import time

import psutil
import torch

from . import datasets, model as modelmod
from .rng import isolated
from .train import apply_dropout_override, build_model, compute_loss, make_optimizer, resolve_device

CANDIDATES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
SAFETY = 0.7  # fraction of free memory we are willing to plan for


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


def _measured(device) -> int | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    if device.type == "mps":
        return torch.mps.driver_allocated_memory()
    return None


def _clear(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        torch.mps.empty_cache()


def probe(config: dict, requested: int | None = None, time_limit: float = 20.0, only_requested: bool = False) -> dict:
    t = config["training"]
    device = resolve_device(t["device"])
    data = datasets.load(config["dataset"]["id"])
    net, info, _ = build_model(config, seed=0)
    apply_dropout_override(net, t.get("dropout"))
    n_params = modelmod.count_params(net)
    with isolated():
        act_per_sample = modelmod.activation_bytes_per_sample(net, info.input_shape, info.input_kind)
    net.to(device)
    opt = make_optimizer(net.parameters(), t)
    opt_slots = {"adam": 2, "adamw": 2, "sgd": 1, "rmsprop": 2, "adagrad": 1}.get(t["optimizer"], 2)
    fixed = n_params * 4 * (2 + opt_slots)  # weights + grads + optimizer state
    budget, budget_label = _budget(device)
    max_n = len(data.train_x)

    candidates = sorted({c for c in CANDIDATES if c <= max_n} | ({requested} if requested else set()))
    if only_requested and requested:
        candidates = [requested]
    results = []
    started = time.time()
    oom_seen = False
    for bs in candidates:
        # Activations are kept for backward, and their gradients need roughly as much again.
        est = fixed + act_per_sample * bs * 2
        row = {"batch_size": bs, "estimated_mb": est / 2**20, "fits": None, "tested": False}
        if oom_seen or est > budget:
            row["fits"] = False
            row["reason"] = "out of memory" if oom_seen else f"estimate exceeds {budget_label}"
            results.append(row)
            continue
        if time.time() - started > time_limit:
            row["fits"] = True
            row["reason"] = "estimated (skipped test: time limit)"
            results.append(row)
            continue
        _clear(device)
        try:
            x = data.prepare(data.train_x[:bs], device)
            y = data.train_y[:bs].to(device)
            t0 = time.perf_counter()
            for _ in range(2):  # first step warms up kernels; time the second
                t0 = time.perf_counter()
                with isolated(device):  # dropout must not disturb running jobs' RNG
                    logits = net(x)
                    loss = compute_loss(logits, y, t, logits.shape[-1])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                opt.step()
                if device.type == "mps":
                    torch.mps.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            measured = _measured(device)
            row.update({
                "fits": True, "tested": True, "step_ms": dt * 1000, "samples_per_s": bs / dt,
                "epoch_s": dt * (max_n / bs),
            })
            if measured is not None:
                row["measured_mb"] = measured / 2**20
            del logits, loss, x, y
        except (RuntimeError, MemoryError) as e:
            msg = str(e).lower()
            if "out of memory" in msg or "mps backend out of memory" in msg or isinstance(e, MemoryError):
                row.update({"fits": False, "reason": "out of memory (tested)"})
                oom_seen = True
            else:
                raise
        results.append(row)
        if row.get("step_ms", 0) > 4000:
            time_limit = 0  # very slow model: estimate the rest
    _clear(device)

    fitting = [r for r in results if r["fits"]]
    largest = max((r["batch_size"] for r in fitting), default=None)
    tested = [r for r in fitting if r.get("samples_per_s")]
    fastest = max(tested, key=lambda r: r["samples_per_s"])["batch_size"] if tested else None
    req_row = next((r for r in results if r["batch_size"] == requested), None)
    return {
        "device": str(device),
        "budget_mb": budget / 2**20,
        "budget_label": budget_label,
        "params": n_params,
        "fixed_mb": fixed / 2**20,
        "activation_kb_per_sample": act_per_sample / 1024,
        "results": results,
        "largest_fitting": largest,
        "fastest": fastest,
        "requested": requested,
        "requested_fits": None if req_row is None else req_row["fits"],
    }
