"""What this machine has to train with, in plain terms."""
from __future__ import annotations

import os
import platform
import subprocess

import psutil
import torch


def _cpu_name() -> str:
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=2).strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def report() -> dict:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    gb = 2**30
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            gpus.append({"device": "cuda", "name": torch.cuda.get_device_name(i), "memory_gb": total / gb, "free_gb": free / gb, "shared": False})
    if torch.backends.mps.is_available():
        # Apple Silicon: the GPU shares the machine's RAM instead of having its own.
        gpus.append({
            "device": "mps", "name": f"{_cpu_name()} GPU (Metal)", "memory_gb": torch.mps.recommended_max_memory() / gb,
            "free_gb": max(0.0, (torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()) / gb), "shared": True,
        })
    notes = []
    if not gpus:
        notes.append("No GPU found: everything trains on the CPU. Small models are fine; convolutional nets on CIFAR-10 and transformers will be slow.")
    if any(g["shared"] for g in gpus):
        notes.append("The GPU shares memory with the rest of the system, so closing other apps frees room for bigger batches.")
    if vm.total < 12 * gb:
        notes.append(f"{vm.total / gb:.0f} GB of RAM is on the small side: keep batch sizes moderate and use the memory checker before big runs.")
    if sw.total and sw.used / max(1, sw.total) > 0.5:
        notes.append("Memory is under pressure right now (a lot of swap is in use). Training will be slower until other apps are closed.")
    return {
        "cpu": _cpu_name(),
        "cores": psutil.cpu_count(logical=False) or os.cpu_count(),
        "threads": torch.get_num_threads(),
        "ram_gb": vm.total / gb,
        "ram_free_gb": vm.available / gb,
        "swap_used_gb": sw.used / gb,
        "gpus": gpus,
        "best_device": gpus[0]["device"] if gpus else "cpu",
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# "Can I train this?" estimates
# ---------------------------------------------------------------------------
# Reference GPUs: memory in GB and a rough sustained training rate in multiply-adds per second
# (conservative figures for convolutional and transformer training). Real speed varies a lot with the model,
# the batch size and the software, so these are order-of-magnitude figures.
REFERENCE_GPUS = [
    {"name": "NVIDIA T4", "memory_gb": 16, "rate": 1.5e12, "note": "free in Colab and Kaggle"},
    {"name": "NVIDIA RTX 4090", "memory_gb": 24, "rate": 1.2e13, "note": "top consumer card"},
    {"name": "NVIDIA A100", "memory_gb": 80, "rate": 1.5e13, "note": "the data-center workhorse"},
    {"name": "NVIDIA H100", "memory_gb": 80, "rate": 2.5e13, "note": "current high end"},
]
OPT_SLOTS = {"adam": 2, "adamw": 2, "sgd": 1, "rmsprop": 2, "adagrad": 1}
_rates: dict[str, float] = {}


def _measure_rate(device: str) -> float:
    """Roughly how many multiply-adds per second this device sustains while training (cached).

    Measured with a few real training steps of a small convolutional network, which is far
    closer to actual training speed than a raw matrix-multiply benchmark.
    """
    import time

    import torch.nn as nn

    if device in _rates:
        return _rates[device]
    try:
        dev = torch.device(device)
        gen = torch.Generator().manual_seed(0)
        net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64 * 16 * 16, 10),
        ).to(dev)
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        batch = 64
        x = torch.randn(batch, 3, 32, 32, generator=gen).to(dev)
        y = torch.randint(0, 10, (batch,), generator=gen).to(dev)
        macs = 3 * (32 * 32 * 32 * 27 + 32 * 32 * 64 * 288 + 64 * 16 * 16 * 10)  # per example, forward + backward

        def step():
            loss = nn.functional.cross_entropy(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            return loss

        step().item()  # warm up
        reps = 4
        t0 = time.perf_counter()
        for _ in range(reps):
            last = step()
        last.item()
        _rates[device] = reps * batch * macs / max(time.perf_counter() - t0, 1e-6)
        del net, opt, x, y
        if dev.type == "mps":
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        _rates[device] = 3e10
    return _rates[device]


def _fit(fixed: float, per_sample: float, batch: int, capacity: float) -> dict:
    """How a model fits in ``capacity`` bytes: at this batch size, and the largest batch that does."""
    need = fixed + per_sample * batch * 2
    max_batch = int((capacity - fixed) // (per_sample * 2)) if per_sample > 0 and capacity > fixed else (10**9 if capacity > fixed else 0)
    return {"fits": need <= capacity, "need_gb": need / 2**30, "max_batch": max(0, min(max_batch, 10**6))}


def estimate(params: int, act_bytes: int, flops: int, batch_size: int, optimizer: str, n_train: int, epochs: int, device: str) -> dict:
    """Memory and time needed to train a model, on this machine and on reference GPUs."""
    gb = 2**30
    fixed = params * 4 * (2 + OPT_SLOTS.get(optimizer, 2))
    batch_size = max(1, int(batch_size))
    need = fixed + act_bytes * batch_size * 2
    train_flops = 3 * flops * n_train  # forward + backward is about three forward passes

    vm = psutil.virtual_memory()
    here = []
    devices = [("cpu", "This computer (CPU)", vm.total * 0.6)]
    if torch.backends.mps.is_available():
        devices.append(("mps", "This computer (GPU)", torch.mps.recommended_max_memory() * 0.7))
    if torch.cuda.is_available():
        devices.append(("cuda", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory * 0.9))
    for dev, label, cap in devices:
        f = _fit(fixed, act_bytes, batch_size, cap)
        rate = _measure_rate(dev)
        here.append({"device": dev, "label": label, "capacity_gb": cap / gb, **f, "epoch_seconds": train_flops / rate if flops else None, "selected": dev == device})

    gpus = []
    for g in REFERENCE_GPUS:
        cap = g["memory_gb"] * gb * 0.9
        f = _fit(fixed, act_bytes, batch_size, cap)
        # With several GPUs the batch is split between them; the weights are copied to each.
        count = 1
        while count < 64 and fixed + act_bytes * (batch_size / count) * 2 > cap:
            count += 1
        possible = fixed + act_bytes * 2 <= cap
        gpus.append({
            "name": g["name"], "memory_gb": g["memory_gb"], "note": g["note"], **f,
            "count": count if possible else None,
            "epoch_seconds": train_flops / (g["rate"] * (count if possible else 1)) if flops else None,
        })

    sel = next((h for h in here if h["selected"]), here[0])
    best_here = min((h for h in here if h["fits"]), key=lambda h: h["epoch_seconds"] or 0, default=None)
    total = (sel["epoch_seconds"] or 0) * max(1, epochs)
    if not sel["fits"]:
        verdict = "too_big"
        headline = (
            f"Too big for {_lc(sel['label'])} at batch size {batch_size}: it needs about {need / gb:.1f} GB and {sel['capacity_gb']:.1f} GB is usable."
            + (f" A batch size of {sel['max_batch']} or less would fit." if sel["max_batch"] >= 1 else " It would not fit even one example at a time.")
        )
    elif total > 6 * 3600:
        verdict = "slow"
        headline = f"It fits, but {epochs} epochs would take very roughly {_dur(total)} on {_lc(sel['label'])}."
    elif total > 30 * 60:
        verdict = "long"
        headline = f"It fits. Expect very roughly {_dur(total)} for {epochs} epochs on {_lc(sel['label'])}."
    else:
        verdict = "ok"
        headline = f"Fine for this computer: about {need / gb:.1f} GB and very roughly {_dur(total)} for {epochs} epochs."
    return {
        "verdict": verdict, "headline": headline,
        "params": params, "need_gb": need / gb, "fixed_gb": fixed / gb, "per_sample_mb": act_bytes * 2 / 2**20,
        "batch_size": batch_size, "epochs": epochs, "n_train": n_train, "flops_per_sample": flops,
        "here": here, "gpus": gpus, "faster_here": best_here["device"] if best_here and best_here["device"] != sel["device"] and (best_here["epoch_seconds"] or 0) < 0.6 * (sel["epoch_seconds"] or 0) else None,
    }


def _lc(label: str) -> str:
    return label[:1].lower() + label[1:]


def _dur(s: float) -> str:
    if s < 90:
        return f"{max(1, round(s))} seconds"
    if s < 5400:
        return f"{round(s / 60)} minutes"
    if s < 48 * 3600:
        return f"{s / 3600:.1f} hours"
    return f"{s / 86400:.1f} days"
