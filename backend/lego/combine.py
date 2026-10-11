"""Make one model out of several trained ones (or squeeze more out of a single one).

* ``soup``       average the weights of models with the same architecture (uniform or greedy)
* ``ensemble``   keep every model and average their predictions, or learn a gate that decides
                 how much to trust each one for each input (a small mixture of experts)
* ``monosoup``   "Model soups need only one ingredient" (Abdollahpoorrostam et al., 2026):
                 re-weight the strong and weak directions of a single model's update
* ``cook``       the full soup recipe: fine-tune several variations from one trained model,
                 then average them

Every result is stored as an ordinary run, so it shows up on the leaderboard and works in
Explore. Its ``combine`` record says what it was made from.
"""
from __future__ import annotations

import copy
import json
import math
import time

import torch
import torch.nn as nn

from . import config as cfgmod, datasets, runs, shifts
from .jobs import Job
from .rng import isolated
from .train import build_model, evaluate, train_run

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def best_state(run_id: str) -> tuple[dict, dict]:
    """(run, weights) of a trained run's best checkpoint."""
    run = runs.get(run_id)
    path = runs.checkpoint_path(run_id, "best")
    if not path.exists():
        path = runs.checkpoint_path(run_id, "last")
    if not path.exists():
        raise ValueError(f"“{run['name']}” has no saved weights.")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise ValueError(f"“{run['name']}” is an ensemble of several models, so it has no single set of weights.")
    return run, ckpt["model"]


def _net(cfg: dict, state: dict) -> nn.Module:
    net, _, _ = build_model(cfg, seed=0)
    net.load_state_dict(state)
    net.eval()
    return net


def _same_data(members: list[dict]):
    keys = {(r["config"]["dataset"]["id"], r["config"]["dataset"].get("seq_len")) for r in members}
    if len(keys) > 1:
        raise ValueError("These models were trained on different datasets (or sequence lengths), so they cannot be combined.")


def _start(cfg: dict) -> str:
    """Identifies the weights a run started from (two runs with the same value began identical)."""
    origin = (cfg.get("init_from") or {}).get("run_id") or f"seed {cfg['training']['seed']}"
    return json.dumps([origin, cfg.get("init_stream"), cfg["architecture"]], sort_keys=True)


def _average(states: list[dict]) -> dict:
    out = {}
    for k, v in states[0].items():
        out[k] = torch.stack([s[k] for s in states]).mean(0) if v.is_floating_point() else v.clone()
    return out


def _score(net: nn.Module, data: datasets.Data, cfg: dict, split: str) -> dict:
    return evaluate(net, data, split, CPU, cfg["training"])


def _row(run: dict, **extra) -> dict:
    s = run["summary"]
    return {"id": run["id"], "name": run["name"], "val_acc": s.get("best_val_acc"), "test_acc": s.get("test_acc"), **extra}


def _shifted(net: nn.Module, cfg: dict) -> dict | None:
    try:
        return shifts.measure(net, cfg["dataset"], CPU)
    except Exception:  # noqa: BLE001
        return None


def _save(cfg: dict, name: str, ckpt: dict, val: dict, test: dict, params: int, combine: dict, started: float, net: nn.Module | None = None) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["name"] = name
    cfg["environment"] = cfgmod.environment()
    run = runs.create(cfg, combine=combine)
    torch.save(ckpt, runs.checkpoint_path(run["id"], "best"))
    run["status"] = "completed"
    run["summary"] = {
        "params": params, "epochs_done": 0, "best_val_acc": val["acc"], "best_val_loss": val["loss"],
        "final_val_acc": val["acc"], "final_val_loss": val["loss"], "test_acc": test["acc"], "test_loss": test["loss"],
        "duration": time.time() - started,
    }
    shifted = _shifted(net, cfg) if net is not None else None
    if shifted:
        run["summary"].update({"shift_acc": shifted["acc"], "shifts": shifted})
    runs.save(run)
    return run


def _count(state: dict) -> int:
    # Buffers such as batch-norm running averages are not trainable weights.
    return sum(v.numel() for k, v in state.items() if v.is_floating_point() and not k.endswith(("running_mean", "running_var")))


# ---------------------------------------------------------------------------
# Model soup
# ---------------------------------------------------------------------------
def soup(job: Job, run_ids: list[str], method: str = "uniform", name: str | None = None) -> dict:
    started = time.time()
    if len(run_ids) < 2:
        raise ValueError("A soup needs at least two models.")
    loaded = [best_state(r) for r in run_ids]
    members = [r for r, _ in loaded]
    _same_data(members)
    shapes = [{k: tuple(v.shape) for k, v in s.items()} for _, s in loaded]
    if any(sh != shapes[0] for sh in shapes[1:]):
        raise ValueError("Soup ingredients must have exactly the same architecture: the same blocks with the same sizes. Use an ensemble to combine different architectures.")
    cfg = members[0]["config"]
    data = datasets.load_for(cfg["dataset"])
    shared = len({_start(r["config"]) for r in members}) == 1

    job.emit({"type": "status", "message": "Tasting each ingredient…"})
    scored = []
    for run, state in loaded:
        job.check_stop()
        scored.append((run, state, _score(_net(run["config"], state), data, run["config"], "val")))
    scored.sort(key=lambda x: (-x[2]["acc"], x[2]["loss"]))
    cfg = scored[0][0]["config"]

    used = [True] * len(scored)
    if method == "greedy":
        # Start from the best model; add each next-best one only if the soup does not get worse.
        kept = [scored[0][1]]
        current = scored[0][2]
        for i, (run, state, _) in enumerate(scored[1:], 1):
            job.check_stop()
            job.emit({"type": "status", "message": f"Trying ingredient {i + 1} of {len(scored)}…"})
            trial = _score(_net(cfg, _average(kept + [state])), data, cfg, "val")
            if trial["acc"] >= current["acc"]:
                kept.append(state)
                current = trial
            else:
                used[i] = False
        final = _average(kept)
    else:
        final = _average([s for _, s, _ in scored])

    job.emit({"type": "status", "message": "Tasting the soup…"})
    net = _net(cfg, final)
    val, test = _score(net, data, cfg, "val"), _score(net, data, cfg, "test")
    best_single = scored[0][2]["acc"]
    if val["acc"] >= best_single:
        note = "The soup is at least as good as its best ingredient, at the cost of a single model."
    elif not shared:
        note = (
            "The soup is worse than its best ingredient. These models did not start from the same weights, so the "
            "same job ended up in different places in each one and averaging scrambles them. Soups work when "
            "every ingredient starts from the same trained model: use “Cook a soup” on a run."
        )
    else:
        note = "The soup is a little worse than its best ingredient. The ingredients drifted too far apart: try a lower learning rate or fewer epochs for the variations, or a greedy soup."
    combine = {
        "kind": "soup", "method": method, "shared_start": shared, "note": note,
        "members": [_row(r, val_acc=v["acc"], used=u) for (r, _, v), u in zip(scored, used)],
    }
    label = f"{'Greedy soup' if method == 'greedy' else 'Soup'} of {sum(used)} · {name or cfg['name']}"
    run = _save(cfg, label, {"model": final, "epoch": 0, "val": val}, val, test, _count(final), combine, started, net)
    job.emit({"type": "combined", "run_id": run["id"], "name": run["name"], "kind": "soup", "shift_acc": run["summary"].get("shift_acc"), "val_acc": val["acc"], "test_acc": test["acc"], "best_member": best_single, "note": note})
    return run


# ---------------------------------------------------------------------------
# Ensembles and a learned gate (mixture of experts)
# ---------------------------------------------------------------------------
class Ensemble(nn.Module):
    """Runs every member and mixes their predicted probabilities.

    Without a gate the mix is a plain average. With one, a small layer looks at what each
    expert predicts and decides how much to trust each of them for this particular input.
    The output is the log of the mixed probabilities, so it can be used like ordinary logits.
    """

    def __init__(self, members: list[nn.Module], gate: nn.Module | None = None):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.gate = gate

    def mix(self, probs: torch.Tensor) -> torch.Tensor:
        """probs: (members, ..., classes) → (..., classes)"""
        if self.gate is None:
            return probs.mean(0)
        weights = self.gate(torch.cat(list(probs), -1)).softmax(-1)  # (..., members)
        return (probs * weights.movedim(-1, 0).unsqueeze(-1)).sum(0)

    def forward(self, x):
        probs = torch.stack([m(x).softmax(-1) for m in self.members])
        return self.mix(probs).clamp_min(1e-12).log()


def build_ensemble(spec: dict) -> Ensemble:
    """Rebuild an ensemble from its checkpoint (which carries every member's config and weights)."""
    nets = [_net(m["config"], m["model"]) for m in spec["members"]]
    gate = None
    if spec.get("gate"):
        w = spec["gate"]["weight"]
        gate = nn.Linear(w.shape[1], w.shape[0])
        gate.load_state_dict(spec["gate"])
    return Ensemble(nets, gate).eval()


@torch.no_grad()
def _probs(net: nn.Module, data: datasets.Data, split: str, job: Job) -> torch.Tensor:
    xs, _ = data.split(split)
    out = []
    for i in range(0, len(xs), 256):
        job.check_stop()
        out.append(net(data.prepare(xs[i : i + 256], CPU)).softmax(-1))
    return torch.cat(out)


def _metrics(mixed: torch.Tensor, y: torch.Tensor) -> dict:
    flat, yf = mixed.reshape(-1, mixed.shape[-1]), y.reshape(-1)
    nll = -flat.gather(1, yf[:, None]).clamp_min(1e-12).log().mean().item()
    return {"loss": nll, "acc": (flat.argmax(-1) == yf).float().mean().item()}


def ensemble(job: Job, run_ids: list[str], method: str = "average", name: str | None = None) -> dict:
    started = time.time()
    if len(run_ids) < 2:
        raise ValueError("An ensemble needs at least two models.")
    loaded = [best_state(r) for r in run_ids]
    members = [r for r, _ in loaded]
    _same_data(members)
    data = datasets.load_for(members[0]["config"]["dataset"])
    val_p, test_p = [], []
    for i, (run, state) in enumerate(loaded):
        job.emit({"type": "status", "message": f"Asking model {i + 1} of {len(loaded)} for its predictions…"})
        net = _net(run["config"], state)
        val_p.append(_probs(net, data, "val", job))
        test_p.append(_probs(net, data, "test", job))
    val_p, test_p = torch.stack(val_p), torch.stack(test_p)
    singles = [_metrics(p, data.val_y) for p in val_p]

    ens = Ensemble([], None)
    gate_state = None
    if method == "gate":
        # Fit the gate on one half of the validation set and score it on the other half, so the
        # reported number is for examples the gate never saw.
        job.emit({"type": "status", "message": "Training the gate that decides which expert to trust…"})
        half = len(data.val_y) // 2
        m, c = val_p.shape[0], val_p.shape[-1]
        with isolated():
            torch.manual_seed(0)
            gate = nn.Linear(m * c, m)
        nn.init.zeros_(gate.weight)  # starts out as a plain average
        nn.init.zeros_(gate.bias)
        ens = Ensemble([], gate)
        opt = torch.optim.Adam(gate.parameters(), lr=0.02)
        fit_p, fit_y = val_p[:, :half], data.val_y[:half].reshape(-1)
        for _ in range(150):
            job.check_stop()
            mixed = ens.mix(fit_p).reshape(-1, c)
            loss = -mixed.gather(1, fit_y[:, None]).clamp_min(1e-12).log().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        gate_state = {k: v.detach().clone() for k, v in gate.state_dict().items()}
        with torch.no_grad():
            val = _metrics(ens.mix(val_p[:, half:]), data.val_y[half:])
            singles = [_metrics(p[half:], data.val_y[half:]) for p in val_p]
            test = _metrics(ens.mix(test_p), data.test_y)
    else:
        val, test = _metrics(val_p.mean(0), data.val_y), _metrics(test_p.mean(0), data.test_y)

    order = sorted(range(len(loaded)), key=lambda i: -singles[i]["acc"])
    best_single = singles[order[0]]["acc"]
    gain = val["acc"] - best_single
    note = (
        f"{'The gated mixture' if method == 'gate' else 'The ensemble'} beats its best member by {gain * 100:.2f} points on validation. It costs {len(loaded)} models’ worth of memory and time for every prediction."
        if gain > 0 else
        "No gain over the best member here. Ensembles help most when the members are about equally good and make different mistakes."
    )
    if method == "gate":
        note += " The gate was fitted on half of the validation set; its validation score is from the other half."
    cfg = members[order[0]]["config"]
    combine = {
        "kind": "ensemble", "method": method, "note": note,
        "members": [_row(members[i], val_acc=singles[i]["acc"], used=True, blocks=len(members[i]["config"]["architecture"]["blocks"])) for i in order],
    }
    ckpt = {"ensemble": {"members": [{"config": members[i]["config"], "model": loaded[i][1]} for i in order], "gate": gate_state}}
    params = sum(_count(loaded[i][1]) for i in order) + (sum(v.numel() for v in gate_state.values()) if gate_state else 0)
    label = f"{'Mixture of experts' if method == 'gate' else 'Ensemble'} of {len(loaded)} · {name or cfg['name']}"
    run = _save(cfg, label, ckpt, val, test, params, combine, started, build_ensemble(ckpt["ensemble"]))
    job.emit({"type": "combined", "run_id": run["id"], "name": run["name"], "kind": "ensemble", "shift_acc": run["summary"].get("shift_acc"), "val_acc": val["acc"], "test_acc": test["acc"], "best_member": best_single, "note": note})
    return run


# ---------------------------------------------------------------------------
# MonoSoup: a soup from one ingredient
# ---------------------------------------------------------------------------
def _mono_layer(w0: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, dict | None]:
    """Re-weight one weight matrix's update. Returns the edited weights and the layer's statistics."""
    delta = (w - w0).reshape(w.shape[0], -1).double()
    if delta.norm() == 0:
        return w, None
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    p = s / s.sum()
    entropy = -(p[p > 0] * p[p > 0].log()).sum()
    k = min(len(s), max(1, math.ceil(math.exp(entropy.item()) - 1e-9)))  # effective rank
    high = (u[:, :k] * s[:k]) @ vh[:k]
    low = delta - high
    energy = s.square().sum()
    p_k = (s[:k].square().sum() / energy).item()
    rho = (s[k] / s[0]).square().item() if k < len(s) else 0.0  # how flat the spectrum is
    cos = math.sqrt(min(1.0, max(0.0, 1.0 - p_k)))  # share of the update in the weak directions
    lam_low = min(1.0, max(0.0, rho + (1 - rho) * cos))
    lam_high = 1.0 - lam_low
    edited = w0.reshape(w.shape[0], -1).double() + lam_high * high + lam_low * low
    return edited.reshape(w.shape).to(w.dtype), {"rank": k, "of": len(s), "lam_high": lam_high, "lam_low": lam_low}


def monosoup(job: Job, run_id: str, name: str | None = None) -> dict:
    started = time.time()
    run, state = best_state(run_id)
    cfg = run["config"]
    src = (cfg.get("init_from") or {}).get("run_id")
    if src:
        try:
            base_run, start = best_state(src)
        except KeyError:
            raise ValueError("The model this one was fine-tuned from has been deleted, so its starting weights are gone.") from None
        origin = f"“{base_run['name']}”, the model it was fine-tuned from"
    else:
        # A run trained from scratch started at the random weights its seed produces: rebuild them.
        with isolated():
            torch.manual_seed(cfg["training"]["seed"])
            net0, _, _ = build_model(cfg)
        start = net0.state_dict()
        origin = "its random starting weights"

    job.emit({"type": "status", "message": "Splitting each layer’s update into strong and weak directions…"})
    edited, layers = {}, []
    for key, w in state.items():
        job.check_stop()
        if w.is_floating_point() and w.ndim >= 2 and key in start and start[key].shape == w.shape:
            edited[key], stats = _mono_layer(start[key], w)
            if stats:
                layers.append({"layer": key, **stats})
        else:
            edited[key] = w.clone()
    if not layers:
        raise ValueError("This model has no weight matrices that changed during training.")

    data = datasets.load_for(cfg["dataset"])
    net = _net(cfg, edited)
    val, test = _score(net, data, cfg, "val"), _score(net, data, cfg, "test")
    original = _net(cfg, state)
    before = _score(original, data, cfg, "val")
    shift_before, shift_after = _shifted(original, cfg), _shifted(net, cfg)
    mean_high = sum(l["lam_high"] for l in layers) / len(layers)
    note = (
        f"Measured each update from {origin}. On average the strong directions were kept at {mean_high * 100:.0f}% strength. "
        + ("Validation accuracy went up." if val["acc"] > before["acc"] else "Validation accuracy did not go up." if val["acc"] < before["acc"] else "Validation accuracy is unchanged.")
        + (" The method was designed for models fine-tuned from a trained one, where it mainly protects what the original model knew; from random starting weights there is little to protect." if not src else "")
    )
    if shift_before and shift_after:
        d = (shift_after["acc"] - shift_before["acc"]) * 100
        note += f" On shifted test data, where the method is meant to help, accuracy went from {shift_before['acc'] * 100:.2f}% to {shift_after['acc'] * 100:.2f}% ({d:+.2f} points)."

    combine = {
        "kind": "monosoup", "method": "monosoup", "note": note, "from_scratch": not src,
        "members": [_row(run, val_acc=before["acc"], used=True)], "layers": layers,
        "shift_before": shift_before and shift_before["acc"],
    }
    out = _save(cfg, name or f"MonoSoup · {cfg['name']}", {"model": edited, "epoch": 0, "val": val}, val, test, _count(edited), combine, started, net)
    job.emit({"type": "combined", "run_id": out["id"], "name": out["name"], "kind": "monosoup", "shift_acc": out["summary"].get("shift_acc"), "shift_before": shift_before and shift_before["acc"], "val_acc": val["acc"], "test_acc": test["acc"], "best_member": before["acc"], "note": note})
    return out


# ---------------------------------------------------------------------------
# The full recipe: fine-tune several variations of one model, then average them
# ---------------------------------------------------------------------------
LR_SPREAD = [1.0, 0.5, 2.0, 0.7, 1.4, 0.35, 2.8, 0.85]


def cook(job: Job, base_id: str, n: int = 4, epochs: int = 2, lr_scale: float = 0.3) -> dict:
    """Fine-tune ``n`` variations of a trained model (different data order, dropout draws and
    learning rates), then make a uniform soup, a greedy soup and an ensemble from them."""
    base_run, _ = best_state(base_id)
    base = cfgmod.normalize(base_run["config"])
    base.pop("environment", None)
    n = max(2, min(int(n), len(LR_SPREAD)))
    group_id = runs.new_id()
    seed0 = base["training"]["seed"]
    seeds = [seed0 + 101 + i for i in range(n)]
    job.meta["group_id"] = group_id
    job.emit({"type": "seeds_start", "group": group_id, "seeds": seeds})
    made: list[str] = []
    for i in range(n):
        job.check_stop()
        cfg = copy.deepcopy(base)
        cfg["init_from"] = {"run_id": base_id, "name": base_run["name"]}
        t = cfg["training"]
        t["seed"] = seeds[i]
        t["epochs"] = int(epochs)
        t["lr"] = float(f"{base['training']['lr'] * lr_scale * LR_SPREAD[i]:.3g}")
        t["warmup_epochs"] = 0
        t["max_steps"] = None
        if t["scheduler"] in ("onecycle", "plateau"):
            t["scheduler"] = "cosine"  # these raise the rate first, which would undo the starting point
        cfg["name"] = f"{base['name']} · variation {i + 1}"
        run = runs.create(cfg, variation={"id": group_id, "index": i, "n": n, "of": base_id})
        made.append(run["id"])
        job.emit({"type": "seed_start", "index": i, "seed": seeds[i], "run_id": run["id"], "n": n})
        summary = train_run(job, run["id"], tag={"seed_index": i})
        job.emit({"type": "seed_end", "index": i, "seed": seeds[i], "run_id": run["id"], **{k: summary.get(k) for k in ("best_val_acc", "test_acc", "best_val_loss", "test_loss")}})
    out = {}
    for kind, fn, method in (("soup", soup, "uniform"), ("greedy", soup, "greedy"), ("ensemble", ensemble, "average")):
        job.check_stop()
        out[kind] = fn(job, made, method, base["name"])["id"]
    return out
