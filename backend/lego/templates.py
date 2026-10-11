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
          "Decoder-only transformer with causal attention, a miniature GPT.",
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
        T("deep_linear", "Deep Linear Network", "Three hidden layers but no activation, still just a linear model.",
          [B("linear", 3, out_features=64, activation="none"), B("classifier")], {"epochs": 20, "lr": 1e-2, "batch_size": 64}),
    ],
}


# ---------------------------------------------------------------------------
# Famous architectures, rebuilt block for block.
#
# These are the real layer layouts and sizes from the papers. The image models use the
# usual small-image adaptation (a 3×3 first convolution and no initial max-pool), because
# the datasets here are 32×32 or 28×28 rather than ImageNet's 224×224.
# ---------------------------------------------------------------------------
def F(id, name, year, authors, paper, note, blocks, training=None, params=None):
    t = T(id, name, note, blocks, training)
    t["famous"] = {"year": year, "authors": authors, "paper": paper}
    t["params"] = params
    return t


def _lenet():
    return [B("conv2d", out_channels=6, kernel_size=5, activation="tanh"), B("avgpool2d"),
            B("conv2d", out_channels=16, kernel_size=5, padding="valid", activation="tanh"), B("avgpool2d"),
            B("flatten"), B("linear", out_features=120, activation="tanh"), B("linear", out_features=84, activation="tanh"), B("classifier")]


def _vgg16():
    out = []
    for ch, n in [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]:
        out += [B("conv2d", n, out_channels=ch, batch_norm=True), B("maxpool2d")]
    return out + [B("flatten"), B("linear", out_features=512), B("dropout", p=0.5), B("classifier")]


def _resnet(kind, layers):
    out = [B("conv2d", out_channels=64, batch_norm=True)]
    for i, n in enumerate(layers):
        width = 64 * 2**i
        stride = 1 if i == 0 else 2
        if kind == "basic":
            out.append(B("resblock", out_channels=width, stride=stride))
            if n > 1:
                out.append(B("resblock", n - 1, out_channels=width))
        else:
            out.append(B("bottleneck", width=width, stride=stride))
            if n > 1:
                out.append(B("bottleneck", n - 1, width=width))
    return out + [B("global_avgpool"), B("classifier")]


def _transformer(dim, heads, layers, dropout, *, gpt=False):
    return [
        B("embedding", dim=dim),
        B("positional_encoding", kind="learned" if gpt else "sinusoidal", dropout=dropout),
        B("transformer", layers, heads=heads, ff_mult=4, dropout=dropout, activation="gelu" if gpt else "relu", norm_first=gpt, causal="yes"),
        *([B("layernorm")] if gpt else []),
        B("classifier"),
    ]


_BIG = {"device": "mps", "scheduler": "onecycle", "augment": True, "weight_decay": 5e-4}
_SMALL_IMG = [
    F("lenet5", "LeNet-5", 1998, "Y. LeCun, L. Bottou, Y. Bengio, P. Haffner", "http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf",
      "The network that read bank checks: two conv + pool stages with tanh, then three dense layers. Tiny by modern standards and still about 99% on MNIST.",
      _lenet(), {"epochs": 5, "lr": 2e-3}, params=61_706),
]

FAMOUS: dict[str, list[dict]] = {
    "mnist": _SMALL_IMG,
    "fashion_mnist": _SMALL_IMG,
    "cifar10": [
        F("lenet5", "LeNet-5", 1998, "Y. LeCun, L. Bottou, Y. Bengio, P. Haffner", "http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf",
          "The 1998 digit reader, pointed at color photos. It was not built for this and tops out around 60 to 65%.",
          _lenet(), {"epochs": 10, "lr": 2e-3}, params=83_126),
        F("vgg16", "VGG-16", 2014, "K. Simonyan, A. Zisserman", "https://arxiv.org/abs/1409.1556",
          "Thirteen 3×3 conv layers in five stages, each followed by pooling. Showed that depth with small filters wins. (With batch norm and a small classifier, as usual for CIFAR.)",
          _vgg16(), {"epochs": 30, "lr": 3e-3, **_BIG}, params=14_986_698),
        F("resnet18", "ResNet-18", 2015, "K. He, X. Zhang, S. Ren, J. Sun", "https://arxiv.org/abs/1512.03385",
          "Four stages of two basic residual blocks each. The standard strong baseline: about 95% on CIFAR-10 with augmentation.",
          _resnet("basic", [2, 2, 2, 2]), {"epochs": 30, "lr": 3e-3, **_BIG}, params=11_173_962),
        F("resnet34", "ResNet-34", 2015, "K. He, X. Zhang, S. Ren, J. Sun", "https://arxiv.org/abs/1512.03385",
          "The same design with 3, 4, 6 and 3 basic blocks per stage.",
          _resnet("basic", [3, 4, 6, 3]), {"epochs": 30, "lr": 3e-3, **_BIG}, params=21_282_122),
        F("resnet50", "ResNet-50", 2015, "K. He, X. Zhang, S. Ren, J. Sun", "https://arxiv.org/abs/1512.03385",
          "Switches to bottleneck blocks (1×1, 3×3, 1×1) in a 3-4-6-3 layout. The most cited vision model ever made.",
          _resnet("bottleneck", [3, 4, 6, 3]), {"epochs": 30, "lr": 3e-3, **_BIG}, params=23_520_842),
        F("resnet101", "ResNet-101", 2015, "K. He, X. Zhang, S. Ren, J. Sun", "https://arxiv.org/abs/1512.03385",
          "Bottleneck blocks in a 3-4-23-3 layout: 101 layers.",
          _resnet("bottleneck", [3, 4, 23, 3]), {"epochs": 30, "lr": 3e-3, **_BIG}, params=42_512_970),
        F("resnet152", "ResNet-152", 2015, "K. He, X. Zhang, S. Ren, J. Sun", "https://arxiv.org/abs/1512.03385",
          "Bottleneck blocks in a 3-8-36-3 layout: 152 layers, the ImageNet 2015 winner.",
          _resnet("bottleneck", [3, 8, 36, 3]), {"epochs": 30, "lr": 3e-3, **_BIG}, params=58_156_618),
        F("vit_base", "ViT-Base", 2020, "A. Dosovitskiy et al.", "https://arxiv.org/abs/2010.11929",
          "The Vision Transformer at its Base size: 12 blocks, width 768, 12 heads (here with 4×4 patches for 32×32 images). Needs far more data than CIFAR-10 to shine.",
          [B("patch_embed", patch_size=4, dim=768), B("positional_encoding"), B("transformer", 12, heads=12, dropout=0.1), B("layernorm"), B("seq_pool", mode="mean"), B("classifier")],
          {"epochs": 30, "lr": 3e-4, "warmup_epochs": 2, "weight_decay": 0.05, "augment": True, "device": "mps"}, params=85_150_474),
    ],
    "shakespeare": [
        F("transformer_base", "Transformer (base)", 2017, "A. Vaswani et al.", "https://arxiv.org/abs/1706.03762",
          "The “base” model of Attention Is All You Need: 6 layers, width 512, 8 heads, feed-forward 2048, dropout 0.1, sinusoidal positions, post-norm. The paper pairs an encoder with a decoder for translation; this is its decoder stack used as a language model.",
          _transformer(512, 8, 6, 0.1), {"epochs": 10, "lr": 3e-4, "warmup_epochs": 1, "weight_decay": 0.01, "device": "mps"}, params=18_980_929),
        F("transformer_big", "Transformer (big)", 2017, "A. Vaswani et al.", "https://arxiv.org/abs/1706.03762",
          "The paper’s “big” model: 6 layers, width 1024, 16 heads, feed-forward 4096, dropout 0.3. Same caveat: the decoder stack as a language model.",
          _transformer(1024, 16, 6, 0.3), {"epochs": 10, "lr": 2e-4, "warmup_epochs": 1, "weight_decay": 0.01, "device": "mps"}, params=75_710_529),
        F("gpt2_small", "GPT-2 (small)", 2019, "A. Radford et al.", "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
          "The layout of the smallest GPT-2: 12 pre-norm blocks, width 768, 12 heads, GELU, learned positions. The original reads 1,024 word-piece tokens; here it reads characters with the sequence length you choose.",
          _transformer(768, 12, 12, 0.1, gpt=True), {"epochs": 10, "lr": 3e-4, "warmup_epochs": 1, "weight_decay": 0.1, "device": "mps"}, params=85_205_057),
    ],
}


def _for_custom(dataset_id: str) -> list[dict]:
    """Starting points for an imported dataset: the templates of the most similar built-in one that fit it."""
    from . import datasets, model as modelmod

    info = datasets.DATASETS.get(dataset_id)
    if not info:
        return []
    if info.input_kind == "image":
        like = "mnist" if info.input_shape[0] == 1 else "cifar10"
    elif info.input_kind == "vector":
        like = "spirals"
    else:
        like = "shakespeare" if info.task == "lm" else "ag_news"
    out = []
    for t in copy.deepcopy(TEMPLATES.get(like, [])):
        if modelmod.inspect({"blocks": t["blocks"]}, info.input_shape, info.ctx()).ok:
            out.append(t)
    if out and not any(t.get("recommended") for t in out):
        out[0]["recommended"] = True
    return out


def for_dataset(dataset_id: str) -> list[dict]:
    """Templates for a dataset; GPU suggestions fall back to whatever accelerator (or CPU) exists here."""
    import torch

    accel = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    out = copy.deepcopy(TEMPLATES.get(dataset_id, []) + FAMOUS.get(dataset_id, []))
    if not out:
        out = _for_custom(dataset_id)
    for t in out:
        if t["training"].get("device") in ("mps", "cuda"):
            t["training"]["device"] = accel
    return out
