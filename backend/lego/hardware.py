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
