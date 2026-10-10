"""Child process that really tries batch sizes: ``python -m lego.probe_worker``.

Reads ``{"config", "sizes", "time_limit"}`` as JSON on stdin and prints one JSON
line per event on stdout:

    {"testing": 64}                      about to try this size (so a crash can be attributed)
    {"batch_size": 64, "fits": true, "tested": true, "step_ms": ..., ...}

It stops at the first size that runs out of memory, once steps become too slow
to be worth timing, or when the time limit is reached. All memory it used is
returned to the system when it exits.
"""
from __future__ import annotations

import json
import sys
import time

import torch

from . import blocks  # noqa: F401  (registers built-in blocks)
from . import datasets
from .train import apply_dropout_override, build_model, compute_loss, make_optimizer, resolve_device

SLOW_STEP_MS = 2500  # beyond this, bigger batches are not worth waiting for


def emit(msg: dict):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
    req = json.loads(sys.stdin.read())
    config, sizes, time_limit = req["config"], sorted(req["sizes"]), float(req.get("time_limit", 20))
    t = config["training"]
    device = resolve_device(t["device"])
    data = datasets.load_for(config["dataset"])
    net, _info, _ = build_model(config, seed=0)
    apply_dropout_override(net, t.get("dropout"))
    net.to(device)
    opt = make_optimizer(net.parameters(), t)

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    started = time.time()
    for bs in sizes:
        if time.time() - started > time_limit:
            break
        bs_real = min(bs, len(data.train_x))
        emit({"testing": bs})
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            x = data.prepare(data.train_x[:bs_real], device)
            y = data.train_y[:bs_real].to(device)
            dt = 0.0
            for _ in range(2):  # the first step warms up kernels; time the second
                t0 = time.perf_counter()
                logits = net(x)
                loss = compute_loss(logits, y, t, logits.shape[-1])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sync()
                dt = time.perf_counter() - t0
            row = {"batch_size": bs, "fits": True, "tested": True, "step_ms": dt * 1000, "samples_per_s": bs_real / dt}
            if device.type == "cuda":
                row["measured_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
            elif device.type == "mps":
                row["measured_mb"] = torch.mps.driver_allocated_memory() / 2**20
            emit(row)
            del logits, loss, x, y
            if dt * 1000 > SLOW_STEP_MS:
                break
        except (RuntimeError, MemoryError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, MemoryError):
                emit({"batch_size": bs, "fits": False, "tested": True, "reason": "out of memory (tested)"})
                break
            raise


if __name__ == "__main__":
    main()
