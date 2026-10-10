"""Look at data, and at what a trained model does with it.

* ``samples``        random examples from a dataset
* ``predict``        a model's predictions on random examples
* ``evaluate``       a model over (part of) a split: accuracy, confusion matrix, its worst mistakes
* ``predict_input``  a model's answer for something the user supplies (a drawing, a photo, a sentence)
* ``generate``       text from a language model, continuing a prompt

Models are run in evaluation mode on the CPU. Evaluation never touches the
global random state (dropout is off and sampling uses its own generator), so
none of this can disturb a training run that is in progress.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import torch

from . import datasets, runs
from .train import build_model

MAX_LOADED_MODELS = 3
_models: OrderedDict[str, tuple] = OrderedDict()
_evals: OrderedDict[tuple, dict] = OrderedDict()
_lock = threading.Lock()


def _decode(data: datasets.Data, ids: torch.Tensor) -> str:
    vocab = data.vocab or []
    if data.info.id == "shakespeare":
        return "".join(vocab[i] for i in ids.tolist())
    return " ".join(vocab[i] for i in ids.tolist() if i != 0)


def _pick(n_total: int, n: int, seed: int) -> torch.Tensor:
    return torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))[:n]


def _example(data: datasets.Data, x: torch.Tensor) -> dict:
    if data.info.input_kind == "image":
        return {"image": x.flatten().tolist(), "shape": list(x.shape)}
    if data.info.input_kind == "tokens":
        return {"text": _decode(data, x)}
    return {"vector": x.tolist()}


def _split(data: datasets.Data, split: str):
    return data.split(split if split in ("train", "val", "test") else "test")


def samples(dataset_id: str, n: int, split: str = "test", seed: int = 0, label: int | None = None, seq_len: int | None = None) -> dict:
    data = datasets.load(dataset_id, seq_len)
    xs, ys = _split(data, split)
    pool = torch.arange(len(xs))
    if label is not None and ys.ndim == 1:
        pool = pool[ys == label]
    idx = pool[_pick(len(pool), n, seed)]
    out = []
    for i in idx:
        ex = _example(data, xs[i])
        y = ys[i]
        ex["label"] = _decode(data, y[:40]) if y.ndim else int(y)
        ex["index"] = int(i)
        out.append(ex)
    counts = torch.bincount(ys, minlength=data.info.num_classes).tolist() if ys.ndim == 1 else None
    return {
        "dataset": dataset_id, "kind": data.info.input_kind, "class_names": data.info.class_names,
        "examples": out, "total": len(xs), "class_counts": counts, "split": split,
    }


# ---------------------------------------------------------------------------
# Loading trained models
# ---------------------------------------------------------------------------
def _load(run_id: str):
    """(net, info, data) for a run's best checkpoint, cached for a few recently used runs."""
    path = runs.checkpoint_path(run_id, "best")
    if not path.exists():
        path = runs.checkpoint_path(run_id, "last")
    if not path.exists():
        raise FileNotFoundError("This run has no saved weights yet.")
    stamp = path.stat().st_mtime
    with _lock:
        hit = _models.get(run_id)
        if hit and hit[0] == stamp:
            _models.move_to_end(run_id)
            return hit[1:]
        cfg = runs.get(run_id)["config"]
        data = datasets.load_for(cfg["dataset"])
        net, info, _ = build_model(cfg, seed=0)
        state = torch.load(path, map_location="cpu", weights_only=False)
        net.load_state_dict(state["model"])
        net.eval()
        _models[run_id] = (stamp, net, info, data)
        while len(_models) > MAX_LOADED_MODELS:
            _models.popitem(last=False)
        for k in [k for k in _evals if k[0] == run_id and k[1] != stamp]:
            del _evals[k]
        return net, info, data


def _top(probs: torch.Tensor, k: int = 3) -> list[dict]:
    top = probs.topk(min(k, probs.shape[-1]))
    return [{"cls": int(c), "p": float(p)} for p, c in zip(top.values, top.indices)]


def _row(data, xs, ys, probs, i: int, j: int) -> dict:
    ex = _example(data, xs[i])
    ex.update({"index": int(i), "label": int(ys[i]), "pred": int(probs[j].argmax()), "top": _top(probs[j])})
    return ex


@torch.no_grad()
def predict(run_id: str, n: int = 12, seed: int = 0, split: str = "test") -> dict:
    net, info, data = _load(run_id)
    xs, ys = _split(data, split)
    idx = _pick(len(xs), n, seed)
    if info.task == "lm":
        prompt = xs[idx[0]]
        # A little randomness reads far better than always taking the single most likely character,
        # which tends to fall into loops.
        text = _generate_ids(net, info, prompt, 200, temperature=0.7, seed=seed)
        return {"kind": "generation", "prompt": _decode(data, prompt), "generated": _decode(data, text)}
    probs = net(data.prepare(xs[idx], "cpu")).softmax(-1)
    return {"kind": info.input_kind, "class_names": info.class_names, "examples": [_row(data, xs, ys, probs, int(i), j) for j, i in enumerate(idx)]}


# ---------------------------------------------------------------------------
# Whole-split evaluation: where does the model go wrong?
# ---------------------------------------------------------------------------
@torch.no_grad()
def _eval_all(run_id: str, split: str, limit: int) -> dict:
    net, info, data = _load(run_id)
    if info.task == "lm":
        raise ValueError("Mistake analysis is for classifiers. For a text generator, use the chat instead.")
    stamp = _models[run_id][0]
    key = (run_id, stamp, split, limit)
    if key in _evals:
        return _evals[key]
    xs, ys = _split(data, split)
    n = min(limit, len(xs)) if limit else len(xs)
    idx = _pick(len(xs), n, 0) if n < len(xs) else torch.arange(len(xs))
    probs = torch.cat([net(data.prepare(xs[idx[i : i + 256]], "cpu")).softmax(-1) for i in range(0, n, 256)])
    out = {"idx": idx, "probs": probs, "pred": probs.argmax(-1), "y": ys[idx], "total": len(xs)}
    _evals[key] = out
    while len(_evals) > 4:
        _evals.popitem(last=False)
    return out


def evaluate(run_id: str, split: str = "test", limit: int = 2000, true: int | None = None, pred: int | None = None, show: int = 48) -> dict:
    ev = _eval_all(run_id, split, limit)
    _net, info, data = _load(run_id)
    xs, ys = _split(data, split)
    y, p, probs, idx = ev["y"], ev["pred"], ev["probs"], ev["idx"]
    k = info.num_classes
    confusion = torch.zeros(k, k, dtype=torch.long)
    confusion.index_put_((y, p), torch.ones_like(y), accumulate=True)
    conf = probs.max(-1).values
    wrong = p != y
    per_class = [{"cls": c, "n": int((y == c).sum()), "acc": float((p[y == c] == c).float().mean()) if (y == c).any() else None} for c in range(k)]

    def rows(mask: torch.Tensor, order: torch.Tensor) -> list[dict]:
        sel = torch.nonzero(mask).flatten()
        sel = sel[order[sel].argsort(descending=True)][:show]
        return [_row(data, xs, ys, probs, int(idx[j]), int(j)) for j in sel]

    mask = wrong
    if true is not None:
        mask = y == true if pred is None else (y == true) & (p == pred)
    elif pred is not None:
        mask = p == pred
    return {
        "kind": info.input_kind, "class_names": info.class_names, "split": split,
        "n": len(y), "total": ev["total"], "accuracy": float((~wrong).float().mean()), "wrong_total": int(wrong.sum()),
        "confusion": confusion.tolist(), "per_class": per_class,
        # The selected examples, most confident first: confident mistakes are the most informative.
        "examples": rows(mask, conf),
        "selected": int(mask.sum()),
        # Right or wrong, the ones it was least sure about.
        "unsure": rows(torch.ones_like(wrong), -conf)[:12],
    }


# ---------------------------------------------------------------------------
# The user's own input
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_input(run_id: str, image: list | None = None, text: str | None = None, vector: list | None = None) -> dict:
    net, info, data = _load(run_id)
    if info.task == "lm":
        raise ValueError("This model generates text. Use the chat to talk to it.")
    if info.input_kind == "image":
        c, h, w = info.input_shape
        if image is None or len(image) != c * h * w:
            raise ValueError(f"Expected an image of {c}×{h}×{w} = {c * h * w} values.")
        x = torch.tensor(image, dtype=torch.float32).clamp(0, 255).to(torch.uint8).view(1, c, h, w)
        shown = {"image": x.flatten().tolist(), "shape": [c, h, w]}
    elif info.input_kind == "tokens":
        words = datasets._TOKEN.findall((text or "").lower())
        if not words:
            raise ValueError("Type some text first.")
        stoi = _stoi(data)
        ids = [stoi.get(wd, 1) for wd in words[: info.seq_len]]
        x = torch.zeros(1, info.seq_len, dtype=torch.long)
        x[0, : len(ids)] = torch.tensor(ids)
        unknown = [wd for wd in words[: info.seq_len] if wd not in stoi]
        shown = {"text": " ".join(words[: info.seq_len]), "unknown": unknown[:20], "truncated": len(words) > info.seq_len}
    else:
        if vector is None or len(vector) != info.input_shape[0]:
            raise ValueError(f"Expected {info.input_shape[0]} numbers.")
        x = torch.tensor([vector], dtype=torch.float32)
        shown = {"vector": vector}
    probs = net(data.prepare(x, "cpu")).softmax(-1)[0]
    return {"kind": info.input_kind, "class_names": info.class_names, "input": shown, "pred": int(probs.argmax()), "probs": probs.tolist()}


_stoi_cache: dict[int, dict] = {}


def _stoi(data: datasets.Data) -> dict:
    key = id(data.vocab)
    if key not in _stoi_cache:
        _stoi_cache.clear()
        _stoi_cache[key] = {w: i for i, w in enumerate(data.vocab or [])}
    return _stoi_cache[key]


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def _generate_ids(net, info, prompt: torch.Tensor, max_new: int, temperature: float, seed: int, top_k: int | None = None) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    seq = prompt.clone()
    for _ in range(max_new):
        logits = net(seq[-info.seq_len :].unsqueeze(0))[0, -1]
        if temperature <= 0:
            nxt = logits.argmax().view(1)
        else:
            logits = logits / temperature
            if top_k:
                kth = logits.topk(min(top_k, len(logits))).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), 1, generator=gen)
        seq = torch.cat([seq, nxt])
    return seq[len(prompt) :]


def generate(run_id: str, prompt: str, max_new: int = 200, temperature: float = 0.8, seed: int = 0, top_k: int | None = None) -> dict:
    net, info, data = _load(run_id)
    if info.task != "lm":
        raise ValueError("Only language models (trained on Tiny Shakespeare) can generate text.")
    stoi = _stoi(data)
    kept = [ch for ch in prompt if ch in stoi]
    dropped = sorted({ch for ch in prompt if ch not in stoi})
    ids = torch.tensor([stoi[ch] for ch in kept] or [stoi.get("\n", 0)], dtype=torch.long)
    out = _generate_ids(net, info, ids, max(1, min(int(max_new), 1000)), float(temperature), int(seed), top_k)
    return {
        "text": _decode(data, out), "seq_len": info.seq_len,
        # What the model actually saw: only the last seq_len characters fit in its window.
        "context": "".join(kept)[-info.seq_len :],
        "dropped": dropped,
    }

