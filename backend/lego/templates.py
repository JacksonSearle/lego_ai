"""Ready-made starting architectures for each dataset.

Each template is a list of blocks (input → output) plus training settings that
are known to work well with it. One per dataset is marked ``recommended``.
"""
from __future__ import annotations

import copy


def B(type, repeat=1, **params):
    return {"type": type, "repeat": repeat, "params": params}


def _ids(blocks):
    out = []
    for i, b in enumerate(blocks):
        b = copy.deepcopy(b)
        b["id"] = f"b{i + 1}"
        out.append(b)
    return out


def T(id, name, description, blocks, training=None, recommended=False):
    return {
        "id": id, "name": name, "description": description,
        "blocks": _ids(blocks), "training": training or {}, "recommended": recommended,
    }


def _mnist_like():
    return [
        T("logreg", "Logistic Regression",
          "The statistical baseline: one linear layer from pixels to classes. ~92% on MNIST.",
          [B("flatten"), B("classifier")], {"epochs": 3, "lr": 1e-3}),
        T("mlp", "MLP (2 hidden layers)",
          "Two fully-connected ReLU layers with dropout. Fast on CPU, ~98% on MNIST.",
          [B("flatten"), B("linear", 2, out_features=256, activation="relu"), B("dropout", p=0.2), B("classifier")],
          {"epochs": 5, "lr": 1e-3}),
        T("cnn", "Simple CNN",
          "Two conv + pool stages then a small dense head. The best accuracy-per-second starting point (~99%+).",
          [B("conv2d", out_channels=32, batch_norm=True), B("maxpool2d"), B("conv2d", out_channels=64, batch_norm=True),
           B("maxpool2d"), B("flatten"), B("linear", out_features=128), B("dropout", p=0.3), B("classifier")],
          {"epochs": 5, "lr": 2e-3}, recommended=True),
        T("resnet", "Mini ResNet",
          "Stem conv, residual blocks, global average pool. Deeper and more accurate, slower on CPU.",
          [B("conv2d", out_channels=32, batch_norm=True), B("resblock", 2), B("resblock", out_channels=64, stride=2),
           B("resblock"), B("global_avgpool"), B("classifier")],
          {"epochs": 5, "lr": 2e-3}),
        T("row_lstm", "Row-by-row LSTM",
          "Reads the image one row of pixels at a time with an LSTM. A classic sequence-model demo.",
          [B("image_to_sequence"), B("lstm", hidden=128, return_sequences=False), B("classifier")],
          {"epochs": 5, "lr": 2e-3}),
        T("vit", "Tiny Vision Transformer",
          "4×4 patches → 49 tokens → 4 transformer blocks → mean pool. Needs warm-up and more epochs than a CNN.",
          [B("patch_embed", patch_size=4, dim=64), B("positional_encoding"), B("transformer", 4, heads=4, dropout=0.1),
           B("layernorm"), B("seq_pool", mode="mean"), B("classifier")],
          {"epochs": 8, "lr": 1e-3, "warmup_epochs": 1, "weight_decay": 0.05}),
    ]


TEMPLATES: dict[str, list[dict]] = {
    "mnist": _mnist_like(),
    "fashion_mnist": _mnist_like(),
    "cifar10": [
        T("mlp", "MLP baseline",
          "Flattened pixels through dense layers. Shows why convolutions matter (~50%).",
          [B("flatten"), B("linear", 2, out_features=512), B("dropout", p=0.3), B("classifier")], {"epochs": 10, "lr": 1e-3}),
        T("vgg", "VGG-style CNN",
          "Three stages of paired 3×3 convs with batch norm and pooling. Solid ~85% with augmentation.",
          [B("conv2d", 2, out_channels=64, batch_norm=True), B("maxpool2d"),
           B("conv2d", 2, out_channels=128, batch_norm=True), B("maxpool2d"),
           B("conv2d", 2, out_channels=256, batch_norm=True), B("maxpool2d"),
           B("global_avgpool"), B("dropout", p=0.3), B("classifier")],
          {"epochs": 15, "lr": 3e-3, "scheduler": "onecycle", "augment": True, "weight_decay": 5e-4, "device": "mps"},
          recommended=True),
        T("resnet", "ResNet-10",
          "ResNet with 4 stages of residual blocks. Strongest CNN here; use a GPU (MPS) if you can.",
          [B("conv2d", out_channels=64, batch_norm=True), B("resblock"),
           B("resblock", out_channels=128, stride=2), B("resblock", out_channels=256, stride=2),
           B("resblock", out_channels=512, stride=2), B("global_avgpool"), B("classifier")],
          {"epochs": 15, "lr": 3e-3, "scheduler": "onecycle", "augment": True, "weight_decay": 5e-4, "device": "mps"}),
        T("mobilenet", "MobileNet-style",
          "Depthwise-separable convs: a fraction of the parameters of the VGG model.",
          [B("conv2d", out_channels=32, batch_norm=True), B("depthwise_conv", out_channels=64),
           B("depthwise_conv", out_channels=128, stride=2), B("depthwise_conv", out_channels=128),
           B("depthwise_conv", out_channels=256, stride=2), B("depthwise_conv", out_channels=256),
           B("global_avgpool"), B("classifier")],
          {"epochs": 15, "lr": 3e-3, "scheduler": "onecycle", "augment": True}),
        T("vit", "Tiny ViT",
          "4×4 patches → 64 tokens → 6 transformer blocks. Transformers need lots of data/epochs to beat CNNs here.",
          [B("patch_embed", patch_size=4, dim=128), B("positional_encoding"), B("transformer", 6, heads=4, dropout=0.1),
           B("layernorm"), B("seq_pool", mode="mean"), B("classifier")],
          {"epochs": 20, "lr": 1e-3, "warmup_epochs": 2, "weight_decay": 0.05, "augment": True, "device": "mps"}),
    ],
    "ag_news": [
        T("fasttext", "Bag of Embeddings (fastText)",
          "Average the word vectors, then classify. Surprisingly strong (~90%) and very fast.",
          [B("embedding", dim=64), B("seq_pool", mode="mean"), B("classifier")], {"epochs": 5, "lr": 3e-3}, recommended=True),
        T("bilstm", "Bidirectional LSTM",
          "Reads the article in both directions; final states feed the classifier.",
          [B("embedding", dim=128), B("lstm", hidden=128, bidirectional=True, return_sequences=False), B("dropout", p=0.3), B("classifier")],
          {"epochs": 4, "lr": 2e-3}),
        T("transformer", "Transformer Encoder",
          "Two transformer blocks over word embeddings, then mean pooling (a tiny BERT).",
          [B("embedding", dim=128), B("positional_encoding"), B("transformer", 2, heads=4), B("seq_pool", mode="mean"), B("classifier")],
          {"epochs": 4, "lr": 5e-4, "warmup_epochs": 0.5}),
    ],
    "shakespeare": [
        T("char_rnn", "Char RNN (Elman)",
          "A vanilla recurrent net predicting the next character. Learns words and spacing, struggles with longer structure.",
          [B("embedding", dim=64), B("rnn", hidden=256), B("classifier")], {"epochs": 5, "lr": 2e-3, "grad_clip": 1.0}),
        T("char_lstm", "Char LSTM",
          "Karpathy's char-rnn: two-layer LSTM over characters.",
          [B("embedding", dim=64), B("lstm", hidden=256, layers=2, dropout=0.1), B("classifier")],
          {"epochs": 5, "lr": 3e-3, "grad_clip": 1.0}, recommended=True),
        T("char_gru", "Char GRU", "Same idea as the LSTM with lighter GRU cells.",
          [B("embedding", dim=64), B("gru", hidden=256, layers=2), B("classifier")], {"epochs": 5, "lr": 3e-3, "grad_clip": 1.0}),
        T("tiny_gpt", "Tiny GPT",
          "Decoder-only transformer with causal attention — a miniature GPT.",
          [B("embedding", dim=128), B("positional_encoding"), B("transformer", 4, heads=4, dropout=0.1, causal="yes"),
           B("layernorm"), B("classifier")],
          {"epochs": 5, "lr": 1e-3, "warmup_epochs": 0.5, "weight_decay": 0.1}),
    ],
    "spirals": [
        T("logreg", "Logistic Regression", "A straight line can't separate spirals: ~35-50%.",
          [B("classifier")], {"epochs": 20, "lr": 1e-2, "batch_size": 64}),
        T("mlp", "MLP", "Three hidden layers carve out the spirals: ~99%.",
          [B("linear", 3, out_features=64, activation="relu"), B("classifier")], {"epochs": 30, "lr": 1e-2, "batch_size": 64},
          recommended=True),
        T("deep_linear", "Deep Linear Network", "Three hidden layers but no activation — still just a linear model.",
          [B("linear", 3, out_features=64, activation="none"), B("classifier")], {"epochs": 20, "lr": 1e-2, "batch_size": 64}),
    ],
}


def for_dataset(dataset_id: str) -> list[dict]:
    """Templates for a dataset; GPU suggestions fall back to whatever accelerator (or CPU) exists here."""
    import torch

    accel = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    out = copy.deepcopy(TEMPLATES.get(dataset_id, []))
    for t in out:
        if t["training"].get("device") in ("mps", "cuda"):
            t["training"]["device"] = accel
    return out
