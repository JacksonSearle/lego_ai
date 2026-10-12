"""Training loop.

Reproducibility: one seed drives weight init, the shuffling generator, the
augmentation generator and dropout. Batches are drawn by indexing in-memory
tensors with ``torch.randperm(generator=...)`` (no worker processes), PyTorch's
deterministic-algorithm mode is enabled, and checkpoints store every RNG state
so a resumed run continues exactly where it left off.
"""
from __future__ import annotations

import math
import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as cfgmod
from . import custom, datasets, model as modelmod, runs, shifts
from .jobs import Job, Stopped
from .rng import RngStream, isolated


class Pruned(Exception):
    """Raised by an epoch callback (Optuna) to abandon a run early."""


def resolve_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name not in ("cpu", None):
        raise ValueError(f"Device '{name}' is not available on this machine.")
    return torch.device("cpu")


def configure_determinism():
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_model(config: dict, *, seed: int | None = None):
    """Build (model, dataset info, report) for a run config, using the config's embedded custom-block sources.

    With ``seed`` the global RNG is seeded inside an isolated section; without
    it the caller must already be inside its own ``RngStream``.
    """
    info = datasets.info_for(config["dataset"])
    overlay = custom.build_registry_overlay(config.get("custom_blocks"))
    # Runs saved before "init_stream" existed drew their shape-check batches from the weight stream.
    legacy = int(config.get("init_stream") or 1) < 2
    kw = dict(strict=True, overlay=overlay, legacy_stream=legacy)
    if seed is None:
        net, report = modelmod.build(config["architecture"], info.input_shape, info.ctx(), **kw)
        return net, info, report
    with isolated():
        torch.manual_seed(seed)
        net, report = modelmod.build(config["architecture"], info.input_shape, info.ctx(), **kw)
    return net, info, report


def apply_dropout_override(net: nn.Module, p: float | None):
    if p is None:
        return
    for m in net.modules():
        if isinstance(m, nn.Dropout):
            m.p = p
        elif isinstance(m, nn.MultiheadAttention):
            m.dropout = p
        elif isinstance(m, nn.RNNBase) and m.num_layers > 1:
            m.dropout = p


def make_optimizer(params, t: dict):
    name, lr, wd = t["optimizer"], t["lr"], t["weight_decay"] or 0.0
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=t["momentum"], weight_decay=wd, nesterov=t["momentum"] > 0)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=t["momentum"], weight_decay=wd)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer {name}")


def lr_lambda(t: dict, steps_per_epoch: int, total_steps: int) -> Callable[[int], float]:
    sched = t["scheduler"]
    warm = (t.get("warmup_epochs") or 0) * steps_per_epoch
    gamma, step_size = t["gamma"], t["step_size"]

    def f(s: int) -> float:
        if sched == "onecycle":
            up = max(1, int(0.3 * total_steps))
            if s < up:
                return 0.04 + 0.96 * s / up
            p = min(1.0, (s - up) / max(1, total_steps - up))
            return 1e-3 + (1 - 1e-3) * 0.5 * (1 + math.cos(math.pi * p))
        if warm > 0 and s < warm:
            return (s + 1) / warm
        if sched == "cosine":
            p = min(1.0, (s - warm) / max(1, total_steps - warm))
            return 0.5 * (1 + math.cos(math.pi * p))
        if sched == "step":
            return gamma ** math.floor(s / steps_per_epoch / step_size)
        if sched == "exponential":
            return gamma ** (s / steps_per_epoch)
        return 1.0

    return f


def compute_loss(logits, y, t: dict, num_classes: int):
    flat = logits.reshape(-1, logits.shape[-1])
    yf = y.reshape(-1)
    if t["loss"] == "mse":
        target = F.one_hot(yf, num_classes).float()
        if t["label_smoothing"]:
            target = target * (1 - t["label_smoothing"]) + t["label_smoothing"] / num_classes
        return F.mse_loss(flat, target)
    return F.cross_entropy(flat, yf, label_smoothing=t["label_smoothing"] or 0.0)


@torch.no_grad()
def evaluate(net, data: datasets.Data, split: str, device, t: dict, batch_size: int = 512) -> dict:
    net.eval()
    x_all, y_all = data.split(split)
    loss_sum, correct, count = 0.0, 0, 0
    for i in range(0, len(x_all), batch_size):
        x = data.prepare(x_all[i : i + batch_size], device)
        y = y_all[i : i + batch_size].to(device)
        logits = net(x)
        n = y.numel()
        loss_sum += compute_loss(logits, y, t, logits.shape[-1]).item() * n
        correct += (logits.argmax(-1) == y).sum().item()
        count += n
    net.train()
    return {"loss": loss_sum / max(1, count), "acc": correct / max(1, count)}


def _subset(data: datasets.Data, n: int | None):
    if not n or n >= len(data.train_x):
        return data.train_x, data.train_y
    return data.train_x[:n], data.train_y[:n]


def train_run(
    job: Job,
    run_id: str,
    *,
    resume: bool = False,
    tag: dict | None = None,
    on_epoch: Callable[[int, dict], None] | None = None,
) -> dict:
    """Train (or resume) the run stored at ``run_id``. Emits events to ``job``.

    ``tag`` is merged into every event (e.g. {"stage": 3} for the history page).
    ``on_epoch(epoch, metrics)`` may raise ``Pruned`` to stop early.
    """
    tag = {"run_id": run_id, **(tag or {})}
    emit = lambda e: job.emit({**tag, **e})  # noqa: E731
    run = runs.get(run_id)
    config = run["config"]
    t = config["training"]
    started = time.time()
    prev_duration = run["summary"].get("duration", 0.0) if resume else 0.0

    run["status"] = "running"
    run["error"] = None
    env = cfgmod.environment()
    if not resume:
        config["environment"] = env
    run["config"] = config
    runs.save(run)

    try:
        emit({"type": "status", "message": "Loading dataset…"})
        data = datasets.load_for(config["dataset"])
        device = resolve_device(t["device"])
        configure_determinism()
        rng = RngStream(t["seed"], device)

        emit({"type": "status", "message": "Building model…"})
        with rng:
            net, info, _ = build_model(config)
        if config.get("init_from"):
            # Fine-tuning: replace the fresh random weights with another run's trained ones.
            from .combine import best_state

            try:
                src, start = best_state(config["init_from"]["run_id"])
            except KeyError:
                raise ValueError("The model this run starts from has been deleted.") from None
            # Copy every layer whose shape still matches. Layers that changed (a different input
            # size, more classes) keep their fresh random weights. So does the output head when
            # the dataset is a different one: its outputs meant other classes there.
            skip: tuple[str, ...] = ()
            a, b = src["config"]["dataset"]["id"], config["dataset"]["id"]
            # (The Frontier Lab's datasets share one vocabulary, so their outputs mean the same thing.)
            if a != b and not (a.startswith("lab_") and b.startswith("lab_")):
                skip = tuple(f"blocks.b_{str(b['id']).replace('.', '_')}." for b in config["architecture"]["blocks"] if b["type"] == "classifier")
            own = net.state_dict()
            copied = [k for k, v in start.items() if k in own and own[k].shape == v.shape and not k.startswith(skip or ("\0",))]
            if not copied:
                raise ValueError("Nothing in the starting model fits this architecture, so there are no weights to carry over.")
            own.update({k: start[k] for k in copied})
            net.load_state_dict(own)
            fresh = sorted({k.rsplit(".", 1)[0] for k in own if k not in copied and own[k].is_floating_point()})
            run["summary"]["init_from"] = {"copied": len(copied), "of": len(own), "fresh": fresh}
        apply_dropout_override(net, t.get("dropout"))
        net.to(device)
        n_params = modelmod.count_params(net)

        train_x, train_y = _subset(data, config["dataset"].get("train_subset"))
        n = len(train_x)
        bs = min(t["batch_size"], n)
        steps_per_epoch = math.ceil(n / bs)
        total_steps = steps_per_epoch * t["epochs"]
        if t.get("max_steps"):
            total_steps = min(total_steps, t["max_steps"])

        opt = make_optimizer(net.parameters(), t)
        if t["scheduler"] == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=t["gamma"], patience=1)
            step_sched = None
        else:
            sched = step_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(t, steps_per_epoch, total_steps))

        data_gen = torch.Generator().manual_seed(t["seed"])
        aug_gen = torch.Generator().manual_seed(t["seed"] + 1)
        augment = bool(t.get("augment")) and info.input_kind == "image"

        start_epoch, global_step = 0, 0
        best = {"val_acc": -1.0, "val_loss": float("inf"), "epoch": None}
        history = {"steps": [], "epochs": []}
        partial = None  # set when resuming a run that was paused in the middle of an epoch
        if resume:
            ckpt = torch.load(runs.checkpoint_path(run_id, "last"), map_location=device, weights_only=False)
            net.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optimizer"])
            sched.load_state_dict(ckpt["scheduler"])
            start_epoch, global_step, best = ckpt["epoch"], ckpt["global_step"], ckpt["best"]
            rng.load(ckpt["rng"])
            data_gen.set_state(ckpt["data_gen"])
            aug_gen.set_state(ckpt["aug_gen"])
            partial = ckpt.get("partial")
            history = runs.metrics(run_id)
            history["steps"] = [s for s in history["steps"] if s["step"] <= global_step]
            history["epochs"] = [e for e in history["epochs"] if e["epoch"] <= start_epoch]

        emit({
            "type": "start", "name": config["name"], "params": n_params, "epochs": t["epochs"],
            "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "resume_epoch": start_epoch + (partial["batch"] / steps_per_epoch if partial else 0),
            "history": history if resume else None, "device": str(device), "train_examples": n,
        })
        run["summary"].update({"params": n_params, "steps_per_epoch": steps_per_epoch, "train_examples": n})
        run["summary"].pop("paused_at", None)
        runs.save(run)

        def checkpoint(epochs_done: int, partial_state: dict | None = None) -> dict:
            return {
                "model": net.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "epoch": epochs_done, "global_step": global_step, "best": best,
                "rng": rng.state(), "data_gen": data_gen.get_state(), "aug_gen": aug_gen.get_state(),
                "partial": partial_state,
            }

        log_every = max(1, steps_per_epoch // 40)
        net.train()
        stop_reason = None
        epoch = start_epoch
        for epoch in range(start_epoch, t["epochs"]):
            if global_step >= total_steps:
                break
            ep_start = time.time()
            if partial:
                # Continue the interrupted epoch: same shuffled order, same running totals.
                perm, first_batch = partial["perm"], partial["batch"]
                acc_loss = torch.tensor(partial["acc_loss"], device=device)
                acc_correct = torch.tensor(partial["acc_correct"], device=device)
                acc_count = partial["acc_count"]
                ep_loss, ep_correct, ep_count = partial["ep_loss"], partial["ep_correct"], partial["ep_count"]
                partial = None
            else:
                perm, first_batch = torch.randperm(n, generator=data_gen), 0
                acc_loss = torch.zeros((), device=device)
                acc_correct = torch.zeros((), device=device)
                acc_count = 0
                ep_loss, ep_correct, ep_count = 0.0, 0, 0
            t_data = t_compute = 0.0  # where this epoch's time goes (for the hardware panel)
            for b in range(first_batch, steps_per_epoch):
                if job.stopping:
                    if b > 0:
                        # Paused mid-epoch: save exactly where we are so resuming loses nothing.
                        torch.save(checkpoint(epoch, {
                            "batch": b, "perm": perm, "acc_loss": acc_loss.item(), "acc_correct": acc_correct.item(),
                            "acc_count": acc_count, "ep_loss": ep_loss, "ep_correct": ep_correct, "ep_count": ep_count,
                        }), runs.checkpoint_path(run_id, "last"))
                        runs.save_metrics(run_id, history)
                    run["summary"].update({
                        "steps": global_step, "paused_at": epoch + b / steps_per_epoch,
                        "duration": prev_duration + time.time() - started,
                    })
                    runs.save(run)
                    raise Stopped()
                t0 = time.perf_counter()
                idx = perm[b * bs : (b + 1) * bs]
                x = data.prepare(train_x[idx], device, augment=augment, gen=aug_gen)
                y = train_y[idx].to(device)
                t1 = time.perf_counter()
                t_data += t1 - t0
                with rng:  # dropout draws from this run's own random stream
                    logits = net(x)
                    loss = compute_loss(logits, y, t, logits.shape[-1])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                if t.get("grad_clip"):
                    nn.utils.clip_grad_norm_(net.parameters(), t["grad_clip"])
                opt.step()
                if step_sched is not None:
                    step_sched.step()
                global_step += 1

                with torch.no_grad():
                    k = y.numel()
                    acc_loss += loss.detach() * k
                    acc_correct += (logits.argmax(-1) == y).sum()
                    acc_count += k
                if global_step % log_every == 0 or b == steps_per_epoch - 1:
                    lsum, csum = acc_loss.item(), acc_correct.item()
                    if not math.isfinite(lsum):
                        raise FloatingPointError("Loss became NaN/inf: try a lower learning rate or gradient clipping.")
                    point = {
                        "step": global_step, "epoch": epoch + (b + 1) / steps_per_epoch,
                        "loss": lsum / acc_count, "acc": csum / acc_count, "lr": opt.param_groups[0]["lr"],
                    }
                    history["steps"].append(point)
                    emit({"type": "step", **point})
                    ep_loss += lsum
                    ep_correct += csum
                    ep_count += acc_count
                    acc_loss.zero_()
                    acc_correct.zero_()
                    acc_count = 0
                t_compute += time.perf_counter() - t1
                if global_step >= total_steps:
                    stop_reason = "max_steps"
                    break

            t_eval = time.perf_counter()
            val = evaluate(net, data, "val", device, t)
            t_eval = time.perf_counter() - t_eval
            if sched is not None and step_sched is None:
                sched.step(val["loss"])
            ep = {
                "epoch": epoch + 1, "step": global_step, "train_loss": ep_loss / max(1, ep_count),
                "train_acc": ep_correct / max(1, ep_count), "val_loss": val["loss"], "val_acc": val["acc"],
                "lr": opt.param_groups[0]["lr"], "seconds": time.time() - ep_start,
                "data_seconds": t_data, "compute_seconds": t_compute, "eval_seconds": t_eval,
            }
            history["epochs"].append(ep)
            improved = val["acc"] > best["val_acc"] or (val["acc"] == best["val_acc"] and val["loss"] < best["val_loss"])
            if improved:
                best = {"val_acc": val["acc"], "val_loss": val["loss"], "epoch": epoch + 1}
            torch.save(checkpoint(epoch + 1), runs.checkpoint_path(run_id, "last"))
            if improved:
                torch.save({"model": net.state_dict(), "epoch": epoch + 1, "val": val}, runs.checkpoint_path(run_id, "best"))
            runs.save_metrics(run_id, history)
            run["summary"].update({
                "epochs_done": epoch + 1, "steps": global_step, "best_val_acc": best["val_acc"],
                "best_val_loss": min(best["val_loss"], min(e["val_loss"] for e in history["epochs"])),
                "best_epoch": best["epoch"], "final_val_acc": val["acc"], "final_val_loss": val["loss"],
                "final_train_loss": ep["train_loss"], "duration": prev_duration + time.time() - started,
            })
            runs.save(run)
            emit({"type": "epoch", **ep, "best_val_acc": best["val_acc"]})
            if on_epoch:
                on_epoch(epoch + 1, ep)
            if stop_reason:
                break

        emit({"type": "status", "message": "Evaluating on the test set…"})
        test = evaluate(net, data, "test", device, t)
        run["summary"].update({"test_acc": test["acc"], "test_loss": test["loss"], "duration": prev_duration + time.time() - started})
        if shifts.available(config["dataset"]["id"]):
            emit({"type": "status", "message": "Testing on shifted data…"})
            try:
                shifted = shifts.measure(net, config["dataset"], device)
                run["summary"].update({"shift_acc": shifted["acc"], "shifts": shifted})
            except Exception:  # noqa: BLE001 - an extra measurement must never fail a finished run
                pass
        run["status"] = "completed"
        _check_reproduction(run)
        runs.save(run)
        emit({"type": "done", "summary": run["summary"], "status": "completed"})
        return run["summary"]

    except Stopped:
        run = runs.get(run_id)
        run["status"] = "stopped"
        runs.save(run)
        emit({"type": "done", "summary": run["summary"], "status": "stopped"})
        raise
    except Pruned:
        run = runs.get(run_id)
        run["status"] = "pruned"
        runs.save(run)
        emit({"type": "done", "summary": run["summary"], "status": "pruned"})
        raise
    except Exception as e:
        run = runs.get(run_id)
        run["status"] = "failed"
        run["error"] = f"{type(e).__name__}: {e}"
        runs.save(run)
        raise


def release_memory():
    """Hand cached accelerator memory back to the system once a job has finished.

    GPU allocators keep freed memory in a per-process cache; on machines where
    the GPU shares RAM with everything else (Apple Silicon) that starves the
    rest of the system if it is never emptied.
    """
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _check_reproduction(run: dict):
    """If this run re-trains another run's config, record whether it matched exactly."""
    src = run.get("reproduce_of")
    if not src:
        return
    try:
        orig = runs.get(src)
    except KeyError:
        return
    if orig["status"] != "completed" or orig["config"]["training"] != run["config"]["training"]:
        return
    a, b = orig["summary"], run["summary"]
    keys = ["test_acc", "test_loss", "final_val_loss", "final_train_loss"]
    run["summary"]["reproduced"] = all(a.get(k) == b.get(k) for k in keys)
    run["summary"]["reproduce_diff"] = {k: [a.get(k), b.get(k)] for k in keys}
