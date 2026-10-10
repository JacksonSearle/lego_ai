"""Per-job random streams.

PyTorch's global RNG (used by weight init and dropout) is shared by every
thread. To keep each training run bit-for-bit reproducible while other jobs (or
the builder's shape inspection) run alongside it, every job owns an
``RngStream``. Entering the stream takes a global lock, installs the job's RNG
state, and on exit saves the advanced state and restores nothing else: anyone
else touching the global RNG must also go through ``RNG_LOCK``.
"""
from __future__ import annotations

import contextlib
import threading

import torch

RNG_LOCK = threading.RLock()


def _get_dev(device):
    if device.type == "mps":
        return torch.mps.get_rng_state()
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    return None


def _set_dev(device, state):
    if state is None:
        return
    if device.type == "mps":
        torch.mps.set_rng_state(state)
    elif device.type == "cuda":
        torch.cuda.set_rng_state(state, device)


class RngStream:
    def __init__(self, seed: int, device: torch.device):
        self.device = device
        with RNG_LOCK, isolated(device):
            torch.manual_seed(seed)
            self.cpu = torch.get_rng_state()
            self.dev = _get_dev(device)

    def __enter__(self):
        RNG_LOCK.acquire()
        torch.set_rng_state(self.cpu)
        _set_dev(self.device, self.dev)
        return self

    def __exit__(self, *exc):
        try:
            self.cpu = torch.get_rng_state()
            self.dev = _get_dev(self.device)
        finally:
            RNG_LOCK.release()

    def state(self) -> dict:
        return {"cpu": self.cpu, "dev": self.dev}

    def load(self, state: dict):
        self.cpu = state["cpu"]
        self.dev = state.get("dev")


@contextlib.contextmanager
def isolated(device: torch.device | None = None):
    """Use the global RNG freely inside; its previous state is restored afterwards."""
    with RNG_LOCK:
        cpu = torch.get_rng_state()
        dev = _get_dev(device) if device is not None else None
        try:
            yield
        finally:
            torch.set_rng_state(cpu)
            if device is not None:
                _set_dev(device, dev)
