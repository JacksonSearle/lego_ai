"""Scaling curves: train one design at several sizes and see how the result changes.

Two axes:

* ``width``  every layer's width is multiplied by each factor (a 4x wider model has roughly
  16x the weights)
* ``data``   the model stays the same and the number of training examples changes

Each point is an ordinary run, tagged with the study it belongs to. The plot and the
power-law fit are drawn by the Scaling page from those runs.
"""
from __future__ import annotations

import copy

from . import config as cfgmod, custom, datasets, model as modelmod, runs
from .jobs import Job
from .registry import REGISTRY
from .train import train_run

# Block parameters that set how wide a layer is.
WIDTH_PARAMS = {"out_features", "out_channels", "hidden", "dim", "width", "token_hidden", "channel_hidden"}
HEAD_PARAMS = {"heads", "num_heads"}
MAX_PARAMS = 150_000_000


def scale_blocks(blocks: list[dict], factor: float, overlay: dict | None = None) -> list[dict]:
    """A copy of ``blocks`` with every width multiplied by ``factor``.

    Widths are rounded to a multiple of the largest attention head count in the model, so
    attention layers still divide evenly. A width of 0 means "same as the input" and is kept.
    """
    specs = {**REGISTRY, **(overlay or {})}
    multiple = 1
    for b in blocks:
        spec = specs.get(b["type"])
        for p in (spec.params if spec else []):
            if p.name in HEAD_PARAMS:
                multiple = max(multiple, int(b.get("params", {}).get(p.name, p.default) or 1))
    out = copy.deepcopy(blocks)
    for b in out:
        spec = specs.get(b["type"])
        for p in (spec.params if spec else []):
            if p.name not in WIDTH_PARAMS or p.type != "int":
                continue
            v = b.setdefault("params", {}).get(p.name, p.default)
            if not v:
                continue
            scaled = max(multiple, int(round(v * factor / multiple)) * multiple)
            b["params"][p.name] = max(scaled, int(p.min or 1))
    return out


def plan(base: dict, axis: str, factors: list[float]) -> list[dict]:
    """The configs a study would train: [{factor, config, params, n_train, error}]."""
    base = cfgmod.normalize(base)
    info = datasets.info_for(base["dataset"])
    overlay = custom.build_registry_overlay(base.get("custom_blocks"))
    full = info.train_size or 0
    out = []
    last: tuple[float, int] | None = None  # the previous size that was built, to predict the next one
    for f in sorted({float(x) for x in factors if float(x) > 0}):
        cfg = copy.deepcopy(base)
        cfg.pop("environment", None)
        if axis == "data":
            n = max(64, int(round(full * f)))
            if n > full:
                continue
            cfg["dataset"]["train_subset"] = None if n >= full else n
            label = f"{n:,} examples"
        else:
            cfg["architecture"]["blocks"] = scale_blocks(base["architecture"]["blocks"], f, overlay)
            n = min(full, base["dataset"].get("train_subset") or full)
            label = f"{f:g}× width"
        cfg["name"] = f"{base['name']} · {label}"
        # Checking a size means building it. Weights grow with the square of the width, so skip
        # sizes that would clearly be too large to even build on this machine.
        predicted = last[1] * (f / last[0]) ** 2 if axis != "data" and last else 0
        if predicted > MAX_PARAMS:
            out.append({"factor": f, "label": label, "config": cfg, "params": int(predicted), "n_train": n, "error": f"About {predicted / 1e6:,.0f} million weights: too large to build on this computer."})
            continue
        report = modelmod.inspect(cfg["architecture"], info.input_shape, info.ctx(), overlay)
        if report.ok:
            last = (f, report.total_params)
        out.append({"factor": f, "label": label, "config": cfg, "params": report.total_params, "n_train": n, "error": None if report.ok else report.error})
    # Two factors can round to the same model (tiny widths): keep one of each.
    seen, unique = set(), []
    for p in out:
        key = (p["params"], p["n_train"])
        if p["error"] or key not in seen:
            unique.append(p)
            seen.add(key)
    return unique


def run_scaling(job: Job, base_config: dict, axis: str, factors: list[float]):
    points = [p for p in plan(base_config, axis, factors) if not p["error"]]
    if len(points) < 2:
        raise ValueError("A scaling curve needs at least two sizes that build correctly.")
    study_id = runs.new_id()
    base_name = cfgmod.normalize(base_config)["name"]
    job.meta.update({"scale_id": study_id, "name": base_name})
    job.emit({"type": "scale_start", "id": study_id, "axis": axis, "name": base_name, "points": [{k: p[k] for k in ("factor", "label", "params", "n_train")} for p in points]})
    for i, p in enumerate(points):
        job.check_stop()
        run = runs.create(p["config"], scale={"id": study_id, "axis": axis, "index": i, "n": len(points), "factor": p["factor"], "name": base_name, "n_train": p["n_train"]})
        job.emit({"type": "point_start", "index": i, "factor": p["factor"], "label": p["label"], "run_id": run["id"], "n": len(points)})
        try:
            s = train_run(job, run["id"], tag={"point": i})
        except FloatingPointError as e:  # one diverging size should not end the study
            job.emit({"type": "point_end", "index": i, "factor": p["factor"], "run_id": run["id"], "error": str(e)})
            continue
        job.emit({
            "type": "point_end", "index": i, "factor": p["factor"], "run_id": run["id"], "params": s.get("params"), "n_train": s.get("train_examples"),
            **{k: s.get(k) for k in ("best_val_acc", "best_val_loss", "test_acc", "test_loss", "duration")},
        })
    job.emit({"type": "scale_end", "id": study_id})
