"""Run configuration: the reproducible JSON description of a training run.

A run config contains everything needed to retrain a model to the same
result: dataset, architecture, every hyperparameter, the random seed, the
source of any custom blocks, and the software/hardware environment it ran on.

    {
      "format": "lego-ai.run", "version": 1,
      "name": "...",
      "dataset": {"id": "mnist", "train_subset": null},
      "architecture": {"blocks": [...]},
      "training": {...},
      "custom_blocks": {"file.py": {"source": "...", "sha256": "..."}},
      "environment": {...}            # filled in when training starts
    }
"""
from __future__ import annotations

import copy
import os
import platform
import sys

import numpy as np
import torch

FORMAT = "lego-ai.run"
VERSION = 1

# Hyperparameter specs drive the training form and the Optuna search-space
# editor. ``tune`` is the default Optuna range; ``log`` means sample on a log scale.
HYPERPARAMS: list[dict] = [
    {"name": "epochs", "label": "Epochs", "type": "int", "default": 5, "min": 1, "max": 500, "group": "Basics",
     "help": "Full passes over the training set."},
    {"name": "batch_size", "label": "Batch size", "type": "int", "default": 128, "min": 1, "max": 8192, "group": "Basics",
     "help": "Examples per gradient step. Larger is faster per epoch but needs more memory.",
     "tune": {"type": "categorical", "choices": [32, 64, 128, 256]}},
    {"name": "lr", "label": "Learning rate", "type": "float", "default": 1e-3, "min": 1e-7, "max": 10, "group": "Optimizer",
     "help": "Step size of each update. The most important hyperparameter.",
     "tune": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    {"name": "optimizer", "label": "Optimizer", "type": "choice", "default": "adamw", "group": "Optimizer",
     "choices": ["sgd", "adam", "adamw", "rmsprop", "adagrad"],
     "help": "Update rule. AdamW is a robust default; SGD+momentum can generalise better with tuning.",
     "tune": {"type": "categorical", "choices": ["adam", "adamw", "sgd"]}},
    {"name": "momentum", "label": "Momentum", "type": "float", "default": 0.9, "min": 0, "max": 0.999, "group": "Optimizer",
     "help": "Momentum for SGD / RMSprop (ignored by Adam-family optimizers).", "show_if": {"optimizer": ["sgd", "rmsprop"]},
     "tune": {"type": "float", "low": 0.8, "high": 0.99, "log": False}},
    {"name": "weight_decay", "label": "Weight decay", "type": "float", "default": 1e-4, "min": 0, "max": 1, "group": "Regularization",
     "help": "L2 penalty pulling weights toward zero (decoupled for AdamW).",
     "tune": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True}},
    {"name": "dropout", "label": "Dropout override", "type": "float", "default": None, "min": 0, "max": 0.95, "nullable": True, "group": "Regularization",
     "help": "If set, overrides the drop probability of every dropout layer in the model. Leave blank to use each block's own setting.",
     "tune": {"type": "float", "low": 0.0, "high": 0.5, "log": False}},
    {"name": "label_smoothing", "label": "Label smoothing", "type": "float", "default": 0.0, "min": 0, "max": 0.5, "group": "Regularization",
     "help": "Soften one-hot targets slightly; reduces over-confidence.",
     "tune": {"type": "float", "low": 0.0, "high": 0.2, "log": False}},
    {"name": "augment", "label": "Data augmentation", "type": "bool", "default": False, "group": "Regularization",
     "help": "Random crop + horizontal flip for colour image datasets (CIFAR-10)."},
    {"name": "scheduler", "label": "LR scheduler", "type": "choice", "default": "cosine", "group": "Schedule",
     "choices": ["none", "cosine", "onecycle", "step", "exponential", "plateau"],
     "help": "How the learning rate changes over training. Cosine decays smoothly to 0; OneCycle ramps up then down; Step multiplies by γ every N epochs; Plateau cuts the LR when validation loss stalls.",
     "tune": {"type": "categorical", "choices": ["none", "cosine", "onecycle"]}},
    {"name": "warmup_epochs", "label": "Warm-up epochs", "type": "float", "default": 0.0, "min": 0, "max": 50, "group": "Schedule",
     "help": "Linearly ramp the LR from 0 over this many epochs (fractions allowed). Helps transformers.",
     "show_if": {"scheduler": ["none", "cosine", "step", "exponential"]}},
    {"name": "step_size", "label": "Step size (epochs)", "type": "int", "default": 2, "min": 1, "max": 100, "group": "Schedule",
     "help": "Epochs between LR drops.", "show_if": {"scheduler": ["step"]}},
    {"name": "gamma", "label": "γ (decay factor)", "type": "float", "default": 0.5, "min": 0.01, "max": 1, "group": "Schedule",
     "help": "Multiply the LR by this at each drop (step/plateau) or every epoch (exponential).",
     "show_if": {"scheduler": ["step", "exponential", "plateau"]}},
    {"name": "grad_clip", "label": "Gradient clipping", "type": "float", "default": 1.0, "min": 0, "max": 100, "nullable": True, "group": "Stability",
     "help": "Rescale gradients whose total norm exceeds this. Blank/0 disables. Important for RNNs."},
    {"name": "loss", "label": "Loss", "type": "choice", "default": "cross_entropy", "choices": ["cross_entropy", "mse"], "group": "Stability",
     "help": "Cross-entropy for classification. MSE regresses one-hot targets (old-school least squares)."},
    {"name": "seed", "label": "Random seed", "type": "int", "default": 42, "min": 0, "max": 2**31 - 1, "group": "Reproducibility",
     "help": "Seeds weight init, data shuffling, dropout and augmentation. Same seed + same config + same environment = same result."},
    {"name": "device", "label": "Device", "type": "choice", "default": "cpu", "choices": ["cpu", "mps", "cuda"], "group": "Reproducibility",
     "help": "CPU runs are bit-for-bit reproducible. GPU backends (Apple MPS, CUDA) are faster but some kernels are non-deterministic."},
    {"name": "max_steps", "label": "Max steps", "type": "int", "default": None, "min": 1, "nullable": True, "group": "Reproducibility",
     "help": "Optional hard cap on optimizer steps (useful for quick experiments)."},
]

DEFAULT_TRAINING = {h["name"]: h["default"] for h in HYPERPARAMS}


def environment() -> dict:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "num_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "available_devices": devices,
    }


def normalize(cfg: dict) -> dict:
    """Fill defaults and validate the shape of a run config. Returns a new dict."""
    cfg = copy.deepcopy(cfg or {})
    if cfg.get("format") not in (None, FORMAT):
        raise ValueError(f"Not a Lego AI run config (format={cfg.get('format')!r})")
    out = {
        "format": FORMAT,
        "version": VERSION,
        "name": cfg.get("name") or "Untitled model",
        "dataset": {"id": "mnist", "train_subset": None, **(cfg.get("dataset") or {})},
        "architecture": {"blocks": list((cfg.get("architecture") or {}).get("blocks") or [])},
        "training": {**DEFAULT_TRAINING, **{k: v for k, v in (cfg.get("training") or {}).items() if k in DEFAULT_TRAINING}},
        "custom_blocks": cfg.get("custom_blocks") or {},
    }
    if "environment" in cfg:
        out["environment"] = cfg["environment"]
    for i, b in enumerate(out["architecture"]["blocks"]):
        b.setdefault("id", f"b{i + 1}")
        b.setdefault("params", {})
        b["repeat"] = max(1, int(b.get("repeat") or 1))
    t = out["training"]
    for h in HYPERPARAMS:
        v = t[h["name"]]
        if v in ("", None):
            t[h["name"]] = None if h.get("nullable") else h["default"]
            continue
        if h["type"] == "int":
            t[h["name"]] = int(v)
        elif h["type"] == "float":
            t[h["name"]] = float(v)
        elif h["type"] == "bool":
            t[h["name"]] = bool(v)
        elif h["type"] == "choice" and v not in h["choices"]:
            raise ValueError(f"{h['label']} must be one of {h['choices']}")
    return out


def env_differences(recorded: dict | None) -> list[str]:
    """Human-readable list of environment differences that could change results."""
    if not recorded:
        return []
    now = environment()
    diffs = []
    for k in ("torch", "python", "numpy", "machine", "num_threads"):
        if recorded.get(k) is not None and str(recorded.get(k)) != str(now.get(k)):
            diffs.append(f"{k}: recorded {recorded.get(k)}, now {now.get(k)}")
    return diffs
