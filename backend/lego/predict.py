"""Show examples from a dataset, and what a trained model predicts for them."""
from __future__ import annotations

import torch

from . import datasets, runs
from .rng import isolated
from .train import build_model


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


def samples(dataset_id: str, n: int, split: str = "test", seed: int = 0) -> dict:
    data = datasets.load(dataset_id)
    xs, ys = data.split(split if split in ("train", "val", "test") else "test")
    idx = _pick(len(xs), n, seed)
    out = []
    for i in idx:
        ex = _example(data, xs[i])
        y = ys[i]
        ex["label"] = _decode(data, y[:40]) if y.ndim else int(y)
        out.append(ex)
    return {"dataset": dataset_id, "kind": data.info.input_kind, "class_names": data.info.class_names, "examples": out}


@torch.no_grad()
def predict(run_id: str, n: int = 12, seed: int = 0) -> dict:
    run = runs.get(run_id)
    cfg = run["config"]
    path = runs.checkpoint_path(run_id, "best")
    if not path.exists():
        path = runs.checkpoint_path(run_id, "last")
    if not path.exists():
        raise FileNotFoundError("This run has no saved weights yet.")
    data = datasets.load(cfg["dataset"]["id"])
    net, info, _ = build_model(cfg, seed=0)
    state = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(state["model"])
    net.eval()
    xs, ys = data.split("test")
    idx = _pick(len(xs), n, seed)

    if info.task == "lm":
        # Greedy generation from a test prompt: what has the model learned to write?
        prompt = xs[idx[0]]
        seq = prompt.clone()
        with isolated():
            for _ in range(160):
                logits = net(seq[-info.seq_len:].unsqueeze(0))
                seq = torch.cat([seq, logits[0, -1].argmax().view(1)])
        return {"kind": "generation", "prompt": _decode(data, prompt), "generated": _decode(data, seq[len(prompt):])}

    x = data.prepare(xs[idx], "cpu")
    with isolated():
        probs = net(x).softmax(-1)
    out = []
    for j, i in enumerate(idx):
        ex = _example(data, xs[i])
        top = probs[j].topk(min(3, probs.shape[-1]))
        ex.update({
            "label": int(ys[i]), "pred": int(top.indices[0]),
            "top": [{"cls": int(c), "p": float(p)} for p, c in zip(top.values, top.indices)],
        })
        out.append(ex)
    return {"kind": info.input_kind, "class_names": info.class_names, "examples": out}
