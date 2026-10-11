"""Shifted test data: how well does a model hold up when the inputs look a bit different?

A model is normally scored on test examples drawn from the same source as its training data.
Real use is rarely that kind: photos are blurrier, digits are written with a different pen.
For the image datasets this module builds several *shifted* test sets:

* corruptions of the dataset's own test images (noise, blur, rotation, a sideways shift,
  low contrast, low resolution), always the same ones, so every model faces identical inputs
* where one exists, a second real dataset with the same classes collected by other people:
  USPS digits for MNIST, CIFAR-10.1 for CIFAR-10

``measure`` scores a model on each and returns the accuracies plus their average.
"""
from __future__ import annotations

import gzip
import math
import threading
import urllib.request

import torch
import torch.nn.functional as F

from . import datasets

N_CORRUPT = 2000  # test images used for each corruption (a fixed random subset)
SEED = 777

CORRUPTIONS = [
    ("noise", "Noise", "Random speckle added to every pixel, like a poor camera sensor."),
    ("blur", "Blur", "Slightly out of focus."),
    ("rotate", "Rotated", "Turned 15 to 25 degrees left or right."),
    ("shift", "Off-center", "Moved a few pixels sideways and up or down."),
    ("contrast", "Low contrast", "Washed out: the difference between light and dark is much smaller."),
    ("pixelate", "Low resolution", "Shrunk to half size and blown back up."),
]
REAL = {
    "mnist": ("usps", "USPS digits", "Real digits from a different source: scanned from envelopes by the US Postal Service."),
    "cifar10": ("cifar10.1", "CIFAR-10.1", "2,000 new photos of the same ten classes, collected years later."),
}

_cache: dict[str, dict] = {}
_failed: set[str] = set()
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Corruptions (uint8 N×C×H×W in, uint8 out)
# ---------------------------------------------------------------------------
def _affine(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def corrupt(x: torch.Tensor, kind: str, gen: torch.Generator) -> torch.Tensor:
    f = x.float() / 255.0
    n, _, h, w = f.shape
    if kind == "noise":
        f = f + torch.randn(f.shape, generator=gen) * 0.18
    elif kind == "blur":
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        kernel = (k[:, None] * k[None, :] / 256.0).expand(f.shape[1], 1, 5, 5).contiguous()
        f = F.conv2d(F.pad(f, (2, 2, 2, 2), mode="replicate"), kernel, groups=f.shape[1])
    elif kind == "rotate":
        deg = (15 + 10 * torch.rand(n, generator=gen)) * (torch.randint(0, 2, (n,), generator=gen) * 2 - 1)
        a = deg * math.pi / 180
        theta = torch.stack([torch.stack([a.cos(), -a.sin(), torch.zeros(n)], 1), torch.stack([a.sin(), a.cos(), torch.zeros(n)], 1)], 1)
        f = _affine(f, theta)
    elif kind == "shift":
        px = max(2, round(w * 0.11))
        dx = (torch.randint(0, 2, (n,), generator=gen) * 2 - 1) * px * 2.0 / w
        dy = (torch.randint(0, 2, (n,), generator=gen) * 2 - 1) * px * 2.0 / h
        theta = torch.zeros(n, 2, 3)
        theta[:, 0, 0] = theta[:, 1, 1] = 1.0
        theta[:, 0, 2], theta[:, 1, 2] = dx, dy
        f = _affine(f, theta)
    elif kind == "contrast":
        mean = f.mean(dim=(1, 2, 3), keepdim=True)
        f = mean + (f - mean) * 0.35
    elif kind == "pixelate":
        f = F.interpolate(F.avg_pool2d(f, 2), size=(h, w), mode="nearest")
    else:
        raise ValueError(f"Unknown corruption {kind}")
    return (f.clamp(0, 1) * 255).round().to(torch.uint8)


# ---------------------------------------------------------------------------
# Second, real datasets
# ---------------------------------------------------------------------------
USPS_URL = "https://hastie.su.domains/ElemStatLearn/datasets/zip.test.gz"


def _usps() -> tuple[torch.Tensor, torch.Tensor]:
    """The USPS test digits (2,007 of them), as published with "The Elements of Statistical Learning".

    Each line is the digit followed by 256 gray values between -1 and 1 (a 16×16 image).
    """
    with urllib.request.urlopen(USPS_URL, timeout=30) as r:
        rows = [[float(v) for v in line.split()] for line in gzip.decompress(r.read()).decode().splitlines() if line.strip()]
    t = torch.tensor(rows)
    y = t[:, 0].long()
    x = ((t[:, 1:] + 1) * 127.5).view(-1, 1, 16, 16)  # digits fill the whole frame
    # MNIST digits sit in a 20×20 box inside a 28×28 frame: match that framing.
    x = F.pad(F.interpolate(x, size=(22, 22), mode="bilinear", align_corners=False), (3, 3, 3, 3))
    return x.clamp(0, 255).round().to(torch.uint8), y


def _cifar101() -> tuple[torch.Tensor, torch.Tensor]:
    import io

    import numpy as np

    base = "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_"

    def get(name):
        with urllib.request.urlopen(base + name, timeout=30) as r:
            return np.load(io.BytesIO(r.read()))

    x = torch.as_tensor(get("data.npy")).permute(0, 3, 1, 2).contiguous().to(torch.uint8)
    return x, torch.as_tensor(get("labels.npy"), dtype=torch.long)


_REAL_LOADERS = {"usps": _usps, "cifar10.1": _cifar101}


def _real(key: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """A second real test set, downloaded once. None when it cannot be fetched (e.g. offline)."""
    path = datasets.DATA_DIR / f"shift_{key.replace('.', '_')}.pt"
    if path.exists():
        d = torch.load(path)
        return d["x"], d["y"]
    if key in _failed:
        return None
    try:
        x, y = _REAL_LOADERS[key]()
    except Exception:  # noqa: BLE001 - no network: simply leave this test out for now
        _failed.add(key)
        return None
    datasets.DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "y": y}, path)
    return x, y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def available(dataset_id: str) -> bool:
    info = datasets.DATASETS.get(dataset_id)
    return bool(info and info.input_kind == "image" and info.task == "classification")


def build(dataset_id: str) -> dict:
    """{shift id: {"name", "description", "x", "y", "real"}} for a dataset (cached)."""
    if not available(dataset_id):
        return {}
    with _lock:
        hit = _cache.get(dataset_id)
        if hit is not None and (dataset_id not in REAL or REAL[dataset_id][0] in hit or REAL[dataset_id][0] in _failed):
            return hit
        data = datasets.load(dataset_id)
        gen = torch.Generator().manual_seed(SEED)
        idx = torch.randperm(len(data.test_x), generator=gen)[:N_CORRUPT]
        x, y = data.test_x[idx], data.test_y[idx]
        out = {"clean": {"name": "Unchanged", "description": "The same test images with nothing done to them, for comparison.", "x": x, "y": y, "real": False}}
        for i, (key, name, desc) in enumerate(CORRUPTIONS):
            g = torch.Generator().manual_seed(SEED + 1 + i)
            out[key] = {"name": name, "description": desc, "x": corrupt(x, key, g), "y": y, "real": False}
        if dataset_id in REAL:
            key, name, desc = REAL[dataset_id]
            got = _real(key)
            if got is not None:
                out[key] = {"name": name, "description": desc, "x": got[0], "y": got[1], "real": True}
        _cache[dataset_id] = out
        return out


@torch.no_grad()
def measure(net: torch.nn.Module, dataset: dict, device=None, stop=None) -> dict | None:
    """Accuracy of ``net`` on every shifted test set of a run's dataset.

    Returns {"acc": average over the shifted sets, "clean": accuracy on the unchanged
    subset, "each": {shift id: accuracy}}, or None for datasets without shifted tests.
    """
    sets = build(dataset["id"])
    if not sets:
        return None
    data = datasets.load_for(dataset)
    device = device or next((p.device for p in net.parameters()), torch.device("cpu"))
    was_training = net.training
    net.eval()
    each = {}
    for key, s in sets.items():
        correct = 0
        for i in range(0, len(s["x"]), 500):
            if stop:
                stop()
            logits = net(data.prepare(s["x"][i : i + 500], device))
            correct += (logits.argmax(-1).cpu() == s["y"][i : i + 500]).sum().item()
        each[key] = correct / len(s["x"])
    if was_training:
        net.train()
    clean = each.pop("clean")
    return {"acc": sum(each.values()) / len(each), "clean": clean, "each": each}


def describe(dataset_id: str, example: int = 0) -> list[dict]:
    """The shifted sets of a dataset with one example image each (the same image where possible)."""
    out = []
    for key, s in build(dataset_id).items():
        x = s["x"][example % len(s["x"])]
        out.append({
            "id": key, "name": s["name"], "description": s["description"], "real": s["real"], "n": len(s["x"]),
            "image": x.flatten().tolist(), "shape": list(x.shape), "label": int(s["y"][example % len(s["y"])]),
        })
    return out
