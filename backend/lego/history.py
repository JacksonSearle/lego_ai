"""A guided tour of model architectures, from least squares to MLP-Mixer.

Every stage trains briefly on the same MNIST subset with the same budget, so
the demo leaderboard compares architectures rather than training effort.
Stage runs are ordinary runs marked ``demo`` (kept off the main leaderboard).
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config as cfgmod, runs
from .jobs import Job
from .templates import B, _ids
from .train import train_run

HISTORY_FILE = Path(__file__).resolve().parent.parent / "data" / "history.json"

BUDGET = {"epochs": 3, "batch_size": 64, "device": "cpu", "seed": 0, "max_steps": None, "weight_decay": 0.0}
SUBSET = 12800  # 200 steps/epoch at batch 64


def S(id, year, title, era, blocks, training, blurb, paper_title, paper_url, authors, added):
    return {
        "id": id, "year": year, "title": title, "era": era, "blocks": _ids(blocks),
        "training": {**BUDGET, **training}, "blurb": blurb, "added": added,
        "paper": {"title": paper_title, "url": paper_url, "authors": authors},
    }


STAGES = [
    S("least_squares", 1805, "Least Squares", "Statistics",
      [B("flatten"), B("classifier")], {"loss": "mse", "optimizer": "sgd", "lr": 0.01, "momentum": 0.9, "scheduler": "none"},
      "Legendre and Gauss fit lines by minimizing squared error. Here every pixel gets a weight per digit and we regress "
      "one-hot targets directly: no probabilities, no neurons, just linear algebra. It already reads most digits.",
      "Nouvelles méthodes pour la détermination des orbites des comètes", "https://archive.org/details/nouvellesmthod00legegoog",
      "A.-M. Legendre", "A linear model fitted by minimizing squared error."),
    S("logistic", 1958, "Logistic Regression", "Statistics",
      [B("flatten"), B("classifier")], {"optimizer": "sgd", "lr": 0.05, "momentum": 0.9, "scheduler": "none"},
      "Cox's logistic (here softmax) regression turns the linear scores into class probabilities and fits them with "
      "cross-entropy. The same year Rosenblatt's perceptron framed this linear unit as an artificial neuron.",
      "The Regression Analysis of Binary Sequences", "https://doi.org/10.1111/j.2517-6161.1958.tb00292.x",
      "D. R. Cox", "Probabilistic outputs and the cross-entropy loss."),
    S("deep_linear", 1969, "Stacked Linear Layers", "Early nets",
      [B("flatten"), B("linear", 2, out_features=256, activation="none"), B("classifier")],
      {"optimizer": "sgd", "lr": 0.02, "momentum": 0.9, "scheduler": "none"},
      "Adding layers without a non-linearity buys nothing: a product of matrices is still one matrix, so this deep "
      "network is exactly as expressive as logistic regression. Minsky & Papert's critique of such limited "
      "perceptrons helped bring on the first AI winter.",
      "Perceptrons: An Introduction to Computational Geometry", "https://mitpress.mit.edu/9780262630221/perceptrons/",
      "M. Minsky, S. Papert", "Depth, but without non-linearity it collapses to a linear model."),
    S("mlp_sigmoid", 1986, "Multi-Layer Perceptron + Backprop", "Connectionism",
      [B("flatten"), B("linear", 2, out_features=256, activation="sigmoid"), B("classifier")],
      {"optimizer": "sgd", "lr": 0.1, "momentum": 0.9, "scheduler": "none"},
      "Rumelhart, Hinton & Williams popularised backpropagation for networks with hidden sigmoid units. The "
      "non-linearity lets hidden layers learn features of their own, breaking the linear ceiling.",
      "Learning representations by back-propagating errors", "https://www.nature.com/articles/323533a0",
      "D. Rumelhart, G. Hinton, R. Williams", "Non-linear hidden layers trained by backpropagation."),
    S("lenet", 1989, "Convolutional Network (LeNet)", "Connectionism",
      [B("conv2d", out_channels=6, kernel_size=5, activation="tanh"), B("avgpool2d"),
       B("conv2d", out_channels=16, kernel_size=5, padding="valid", activation="tanh"), B("avgpool2d"),
       B("flatten"), B("linear", out_features=120, activation="tanh"), B("linear", out_features=84, activation="tanh"), B("classifier")],
      {"optimizer": "sgd", "lr": 0.05, "momentum": 0.9, "scheduler": "none"},
      "LeCun's convolutional nets share small filters across the image, building in the prior that a stroke means "
      "the same thing wherever it appears. Far fewer parameters, much better accuracy: they read US zip codes and checks.",
      "Backpropagation Applied to Handwritten Zip Code Recognition", "https://doi.org/10.1162/neco.1989.1.4.541",
      "Y. LeCun et al.", "Weight-shared convolutions and pooling."),
    S("elman_rnn", 1990, "Recurrent Network (Elman)", "Connectionism",
      [B("image_to_sequence"), B("rnn", hidden=128, return_sequences=False), B("classifier")],
      {"optimizer": "adam", "lr": 2e-3, "scheduler": "none", "grad_clip": 1.0},
      "Elman's simple recurrent network keeps a hidden state that is fed back at every step, giving networks memory "
      "for sequences. Here it reads the digit one pixel-row at a time, 28 steps, and must remember the strokes it saw.",
      "Finding Structure in Time", "https://doi.org/10.1207/s15516709cog1402_1",
      "J. Elman", "A recurrent hidden state that carries information through time."),
    S("lstm", 1997, "LSTM", "Connectionism",
      [B("image_to_sequence"), B("lstm", hidden=128, return_sequences=False), B("classifier")],
      {"optimizer": "adam", "lr": 2e-3, "scheduler": "none", "grad_clip": 1.0},
      "Hochreiter & Schmidhuber's Long Short-Term Memory adds a gated cell state, a protected memory lane, so "
      "gradients survive many steps. It dominated speech and language for twenty years.",
      "Long Short-Term Memory", "https://www.bioinf.jku.at/publications/older/2604.pdf",
      "S. Hochreiter, J. Schmidhuber", "Gates and a cell state that fix vanishing gradients."),
    S("alexnet", 2012, "Deep CNN with ReLU + Dropout", "Deep learning",
      [B("conv2d", out_channels=32), B("maxpool2d"), B("conv2d", out_channels=64), B("maxpool2d"),
       B("flatten"), B("linear", out_features=256), B("dropout", p=0.5), B("classifier")],
      {"optimizer": "sgd", "lr": 0.05, "momentum": 0.9, "scheduler": "none"},
      "AlexNet won ImageNet by a landslide with ReLU activations (which don't saturate like sigmoid/tanh), max "
      "pooling, dropout regularization and GPU training, the start of the deep learning boom.",
      "ImageNet Classification with Deep Convolutional Neural Networks",
      "https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html",
      "A. Krizhevsky, I. Sutskever, G. Hinton", "ReLU, max-pooling and dropout at scale."),
    S("resnet", 2015, "Residual Network + BatchNorm", "Deep learning",
      [B("conv2d", out_channels=32, batch_norm=True), B("maxpool2d"), B("resblock", 2), B("resblock", out_channels=64, stride=2),
       B("global_avgpool"), B("classifier")],
      {"optimizer": "sgd", "lr": 0.05, "momentum": 0.9, "scheduler": "none"},
      "Batch normalization (Ioffe & Szegedy) stabilized training, and He et al.'s skip connections let each block "
      "learn a small correction to the identity, making networks with 100+ layers trainable.",
      "Deep Residual Learning for Image Recognition", "https://arxiv.org/abs/1512.03385",
      "K. He, X. Zhang, S. Ren, J. Sun", "Skip connections, batch norm and global average pooling."),
    S("transformer", 2017, "Transformer", "Attention",
      [B("image_to_sequence"), B("linear", out_features=64, activation="none"), B("positional_encoding"),
       B("transformer", 2, heads=4, dropout=0.0), B("seq_pool", mode="mean"), B("classifier")],
      {"optimizer": "adamw", "lr": 1e-3, "scheduler": "none", "warmup_epochs": 0.3},
      "Vaswani et al. dropped recurrence entirely: self-attention lets every row of the image look at every other "
      "row in one step, with positional encodings supplying order. The architecture behind today's language models.",
      "Attention Is All You Need", "https://arxiv.org/abs/1706.03762",
      "A. Vaswani et al.", "Self-attention, positional encodings, pre/post-norm residual blocks."),
    S("vit", 2020, "Vision Transformer", "Attention",
      [B("patch_embed", patch_size=4, dim=64), B("positional_encoding"), B("transformer", 2, heads=4, dropout=0.0),
       B("layernorm"), B("seq_pool", mode="mean"), B("classifier")],
      {"optimizer": "adamw", "lr": 1e-3, "scheduler": "none", "warmup_epochs": 0.3},
      "Dosovitskiy et al. showed that \"an image is worth 16×16 words\": split it into patches, embed each, and a "
      "plain transformer matches CNNs, given enough data. On a tiny budget like this one, CNNs still have the edge.",
      "An Image is Worth 16x16 Words", "https://arxiv.org/abs/2010.11929",
      "A. Dosovitskiy et al.", "Patch embeddings: images as sequences of tokens."),
    S("mixer", 2021, "MLP-Mixer", "Attention",
      [B("patch_embed", patch_size=4, dim=64), B("mixer", 2, token_hidden=64, channel_hidden=128),
       B("layernorm"), B("seq_pool", mode="mean"), B("classifier")],
      {"optimizer": "adamw", "lr": 1e-3, "scheduler": "none", "warmup_epochs": 0.3},
      "Full circle: Tolstikhin et al. replaced attention with plain MLPs that alternately mix across patches and "
      "across channels, showing much of the transformer's power comes from the overall recipe, not attention itself.",
      "MLP-Mixer: An all-MLP Architecture for Vision", "https://arxiv.org/abs/2105.01601",
      "I. Tolstikhin et al.", "Token-mixing and channel-mixing MLPs instead of attention."),
]


def stage_config(stage: dict) -> dict:
    return cfgmod.normalize({
        "name": f"{stage['year']} · {stage['title']}",
        "dataset": {"id": "mnist", "train_subset": SUBSET},
        "architecture": {"blocks": stage["blocks"]},
        "training": stage["training"],
    })


def _load_latest() -> dict:
    return json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else {}


def _save_latest(d: dict):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(d, indent=2))


def state() -> dict:
    """Stages with their most recent demo run (summary + metrics)."""
    latest = _load_latest()
    out = []
    for s in STAGES:
        entry = {**s, "run": None, "metrics": None}
        rid = latest.get(s["id"])
        if rid:
            try:
                r = runs.get(rid)
                entry["run"] = {"id": r["id"], "status": r["status"], "summary": r["summary"]}
                entry["metrics"] = runs.metrics(rid)
            except KeyError:
                pass
        out.append(entry)
    return {"stages": out, "budget": {**BUDGET, "train_subset": SUBSET, "dataset": "mnist"}}


def run_all(job: Job, only: list[str] | None = None):
    latest = _load_latest()
    todo = [s for s in STAGES if not only or s["id"] in only]
    for s in todo:
        job.check_stop()
        idx = STAGES.index(s)
        old = latest.get(s["id"])
        run = runs.create(stage_config(s), demo=True, history_stage=s["id"])
        latest[s["id"]] = run["id"]
        _save_latest(latest)
        if old:
            runs.delete(old)
        job.emit({"type": "stage_start", "stage": idx, "stage_id": s["id"], "run_id": run["id"]})
        train_run(job, run["id"], tag={"stage": idx, "stage_id": s["id"]})
        job.emit({"type": "stage_end", "stage": idx, "stage_id": s["id"], "run_id": run["id"], "summary": runs.get(run["id"])["summary"]})
