"""Export a model as standalone Python scripts.

``train.py`` contains the model written out as plain PyTorch (every layer with its concrete
sizes), the dataset loading and the training loop. It needs only ``torch`` (and
``torchvision`` for the image datasets): no Lego AI install. It follows the app's training
procedure step for step, so with the same seed on the CPU it reaches the same numbers.

A second script uses the trained weights: ``generate.py`` for language models (continue a
prompt) or ``predict.py`` for classifiers (classify your own image or text).
"""
from __future__ import annotations

import inspect
import io
import json
import pprint
import re
import zipfile

import torch.nn as nn

from . import config as cfgmod, custom, datasets
from .registry import ACTIVATIONS, CURVE_XS, REGISTRY, LegoModule
from .train import build_model


class ExportError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Model → code
# ---------------------------------------------------------------------------
def _expr(m: nn.Module, used: dict, indent: int = 2) -> str:
    """A Python expression that constructs module ``m``."""
    pad = "    " * indent
    cust = m.__dict__.get("_lego_custom")
    if cust:
        used["custom"].add(cust["name"])
        args = [repr(tuple(cust["in_shape"]))] + [f"{k}={v!r}" for k, v in cust["params"].items()]
        if cust["ctx"]:
            args.append("ctx=CTX")
        return f"{cust['name']}({', '.join(args)})"
    if isinstance(m, LegoModule):
        used["classes"][type(m).__name__] = type(m)
        args, kw = m._lego_args
        parts = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kw.items()]
        return f"{type(m).__name__}({', '.join(parts)})"
    if isinstance(m, nn.Sequential):
        inner = ",\n".join(f"{pad}    {_expr(c, used, indent + 1)}" for c in m)
        return f"nn.Sequential(\n{inner},\n{pad})"
    text = repr(m)
    if "\n" in text or not type(m).__module__.startswith("torch.nn"):
        raise ExportError(f"Don't know how to write a {type(m).__name__} layer as code.")
    return "nn." + text


def _class_sources(used: dict, customs: str = "") -> str:
    out = []
    need_act = "make_activation" in customs or "ACTIVATIONS" in customs
    for cls in used["classes"].values():
        src = inspect.getsource(cls).replace("(LegoModule)", "(nn.Module)")
        need_act = need_act or "make_activation" in src
        out.append(src.rstrip())
    pre = []
    if need_act:
        entries = ",\n".join(f'    "{k}": {_activation_expr(k)}' for k in ACTIVATIONS)
        pre.append(
            "ACTIVATIONS = {\n" + entries + ",\n}\n\n\n"
            "def make_activation(name):\n"
            '    """The non-linearity called `name` ("none" means no activation)."""\n'
            '    return nn.Identity() if name in (None, "", "none") else ACTIVATIONS[name]()'
        )
    if "DrawnActivation" in used["classes"]:
        pre.append(f"CURVE_XS = {CURVE_XS!r}  # x positions of a hand-drawn activation's points")
    return "\n\n\n".join(pre + out)


def _activation_expr(name: str) -> str:
    return {
        "relu": "nn.ReLU", "gelu": "nn.GELU", "silu": "nn.SiLU", "tanh": "nn.Tanh", "sigmoid": "nn.Sigmoid",
        "leaky_relu": "lambda: nn.LeakyReLU(0.01)", "elu": "nn.ELU", "mish": "nn.Mish", "softplus": "nn.Softplus",
        "identity": "nn.Identity",
    }[name]


def _custom_sources(used: dict, cfg: dict) -> str:
    if not used["custom"]:
        return ""
    files = []
    for fname, entry in (cfg.get("custom_blocks") or {}).items():
        src = entry["source"]
        if any(re.search(rf"^class {re.escape(n)}\b", src, flags=re.M) for n in used["custom"]):
            body = re.sub(r"^from lego(\.\w+)* import (\([^)]*\)|[^\n]*)\n", "", src, flags=re.M)
            files.append(f"# ---- custom blocks from {fname} ----\n{body.strip()}")
    if not files:
        raise ExportError("The source of a custom block used by this model is missing from its config.")
    shim = (
        "# Custom blocks were written for Lego AI's block decorator. Outside the app the decorator does\n"
        "# nothing, so these two stand-ins let that code run unchanged.\n"
        "class Param:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        pass\n\n\n"
        "def lego_block(*args, **kwargs):\n"
        "    return lambda obj: obj"
    )
    return shim + "\n\n\n" + "\n\n\n".join(files)


def _fmt(shape) -> str:
    return "×".join(str(s) for s in shape) if shape else "scalar"


def _model_code(cfg: dict) -> tuple[str, str, str, datasets.DatasetInfo, int]:
    """(helper classes, custom sources, Net class, dataset info, parameter count)"""
    net, info, report = build_model(cfg, seed=0)
    used = {"classes": {}, "custom": set()}
    init, fwd = [], []
    reports = {b.id: b for b in report.blocks}
    for i, raw in enumerate(cfg["architecture"]["blocks"], 1):
        key = "b_" + str(raw["id"]).replace(".", "_")
        module = net.blocks[key]
        spec = REGISTRY.get(raw["type"])
        label = spec.name if spec else raw["type"]
        repeat = max(1, int(raw.get("repeat") or 1))
        attr = f"layer{i}_" + re.sub(r"[^a-z0-9]+", "_", raw["type"].split(":")[-1].lower()).strip("_")
        r = reports[str(raw["id"])]
        init.append(f"        # {i}. {label}{f' ×{repeat}' if repeat > 1 else ''}: {_fmt(r.in_shape)} → {_fmt(r.out_shape)}   ({r.params:,} weights)")
        init.append(f"        self.{attr} = {_expr(module, used)}")
        fwd.append(f"        x = self.{attr}(x)")
    net_src = (
        "class Net(nn.Module):\n"
        f'    """{cfg["name"]}: {report.total_params:,} weights. Input {_fmt(report.input_shape)} → output {_fmt(report.output_shape)}."""\n\n'
        "    def __init__(self):\n        super().__init__()\n" + "\n".join(init) + "\n\n"
        "    def forward(self, x):\n" + "\n".join(fwd) + "\n        return x"
    )
    customs = _custom_sources(used, cfg)
    return _class_sources(used, customs), customs, net_src, info, report.total_params


# ---------------------------------------------------------------------------
# Dataset loading code, one variant per dataset
# ---------------------------------------------------------------------------
_SPLIT = '''
def split_train_val(x, y):
    """Hold out a fixed validation slice (the same one Lego AI uses), then apply the training subset."""
    n = len(x)
    n_val = min(5000, max(1, n // 10))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(12345))
    val, train = perm[:n_val], perm[n_val:]
    tx, ty = x[train].contiguous(), y[train].contiguous()
    if CONFIG["train_subset"]:
        tx, ty = tx[: CONFIG["train_subset"]], ty[: CONFIG["train_subset"]]
    return tx, ty, x[val].contiguous(), y[val].contiguous()
'''

_IMAGE_DATA = '''
MEAN, STD = __MEAN__, __STD__


def load_data():
    """__NAME__ via torchvision, as uint8 tensors of shape (N, C, H, W)."""
    import torchvision

    def get(train):
        ds = torchvision.datasets.__CLS__(root=str(DATA_DIR), train=train, download=True)
        x = torch.as_tensor(ds.data)
        x = x.unsqueeze(1) if x.ndim == 3 else x.permute(0, 3, 1, 2)
        return x.contiguous().to(torch.uint8), torch.as_tensor(ds.targets, dtype=torch.long)

    (x, y), (test_x, test_y) = get(True), get(False)
    return (*split_train_val(x, y), test_x, test_y)


def augment(x, gen):
    """Random crop (4 pixels of padding) and horizontal flip."""
    n, c, h, w = x.shape
    padded = F.pad(x, (4, 4, 4, 4))
    dx = torch.randint(0, 9, (n,), generator=gen)
    dy = torch.randint(0, 9, (n,), generator=gen)
    flip = torch.rand(n, generator=gen) < 0.5
    out = torch.empty_like(x)
    for i in range(n):
        img = padded[i, :, dy[i] : dy[i] + h, dx[i] : dx[i] + w]
        out[i] = img.flip(-1) if flip[i] else img
    return out


def prepare(x, device, gen=None):
    """uint8 pixels → normalized floats on the training device."""
    if gen is not None:
        x = augment(x, gen)
    x = x.to(device).float().div_(255.0)
    mean = torch.tensor(MEAN, device=device).view(1, -1, 1, 1)
    std = torch.tensor(STD, device=device).view(1, -1, 1, 1)
    return (x - mean) / std
'''

_SHAKESPEARE_DATA = '''
SEQ_LEN = __SEQ_LEN__
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text():
    path = DATA_DIR / "tinyshakespeare.txt"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(URL, timeout=60).read())
    return path.read_text()


def load_vocab():
    return sorted(set(load_text()))


def load_data():
    """Tiny Shakespeare as character ids, cut into windows of SEQ_LEN; the target is the next character."""
    text = load_text()
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cut = int(len(ids) * 0.9)

    def chunks(seq):
        n = (len(seq) - 1) // SEQ_LEN
        return seq[: n * SEQ_LEN].view(n, SEQ_LEN).contiguous(), seq[1 : n * SEQ_LEN + 1].view(n, SEQ_LEN).contiguous()

    (x, y), (test_x, test_y) = chunks(ids[:cut]), chunks(ids[cut:])
    return (*split_train_val(x, y), test_x, test_y)


def prepare(x, device, gen=None):
    return x.to(device)
'''

_AG_NEWS_DATA = '''
SEQ_LEN = __SEQ_LEN__
VOCAB_SIZE = 20002
BASE_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/"
TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text):
    return TOKEN.findall(text.replace("\\\\", " ").lower())


def _rows(name):
    path = DATA_DIR / f"ag_news_{name}"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(BASE_URL + name, timeout=60).read())
    for label, title, desc in csv.reader(io.StringIO(path.read_text(encoding="utf-8"))):
        yield int(label) - 1, tokenize(f"{title} {desc}")


def load_vocab():
    counts = Counter(w for _, toks in _rows("train.csv") for w in toks)
    return ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(VOCAB_SIZE - 2)]


def encode(tokens, stoi):
    ids = [stoi.get(w, 1) for w in tokens[:SEQ_LEN]]
    return ids + [0] * (SEQ_LEN - len(ids))


def load_data():
    """AG News headlines as word ids, padded or cut to SEQ_LEN words."""
    stoi = {w: i for i, w in enumerate(load_vocab())}

    def get(name):
        rows = list(_rows(name))
        return torch.tensor([encode(t, stoi) for _, t in rows], dtype=torch.long), torch.tensor([l for l, _ in rows], dtype=torch.long)

    (x, y), (test_x, test_y) = get("train.csv"), get("test.csv")
    return (*split_train_val(x, y), test_x, test_y)


def prepare(x, device, gen=None):
    return x.to(device)
'''

_SPIRALS_DATA = '''
def load_data():
    """Three interleaved 2D spirals (synthetic)."""
    g = torch.Generator().manual_seed(0)
    n_per, k = 1334, 3
    xs, ys = [], []
    for c in range(k):
        r = torch.linspace(0.05, 1.0, n_per)
        theta = c * 2 * math.pi / k + r * 4.0 + torch.randn(n_per, generator=g) * 0.25
        xs.append(torch.stack([r * torch.cos(theta), r * torch.sin(theta)], 1))
        ys.append(torch.full((n_per,), c, dtype=torch.long))
    x, y = torch.cat(xs), torch.cat(ys)
    perm = torch.randperm(len(x), generator=g)
    x, y = x[perm] * 2, y[perm]
    return (*split_train_val(x[:3000], y[:3000]), x[3000:4000], y[3000:4000])


def prepare(x, device, gen=None):
    return x.to(device)
'''


def _data_code(info: datasets.DatasetInfo) -> str:
    if info.input_kind == "image":
        cls = {"mnist": "MNIST", "fashion_mnist": "FashionMNIST", "cifar10": "CIFAR10"}[info.id]
        body = _IMAGE_DATA.replace("__MEAN__", repr(info.mean)).replace("__STD__", repr(info.std)).replace("__CLS__", cls).replace("__NAME__", info.name)
    elif info.id == "shakespeare":
        body = _SHAKESPEARE_DATA.replace("__SEQ_LEN__", str(info.seq_len))
    elif info.id == "ag_news":
        body = _AG_NEWS_DATA.replace("__SEQ_LEN__", str(info.seq_len))
    else:
        body = _SPIRALS_DATA
    return (_SPLIT + body).strip("\n")


# ---------------------------------------------------------------------------
# train.py
# ---------------------------------------------------------------------------
_TRAIN = '''"""__TITLE__

Standalone training script exported from Lego AI. It needs only PyTorch__TV__:

    pip install torch__TVPIP__
    python train.py

It trains the model below on __DATASET__ and saves the weights to model.pt.
__REPRO__
"""
import csv
import io
import math
import os
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

# Every setting of the run. Edit freely.
CONFIG = __CONFIG__

DATA_DIR = Path(os.environ.get("LEGO_DATA", "data"))  # datasets are downloaded here
NUM_CLASSES = __NUM_CLASSES__
CLASS_NAMES = __CLASS_NAMES__
__CTX__

# =============================================================================
# The model
# =============================================================================
__HELPERS__


__NET__


# =============================================================================
# The data
# =============================================================================
__DATA__


# =============================================================================
# Training
# =============================================================================
def pick_device(name):
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name != "cpu":
        print(f"Device {name!r} is not available here; using the CPU.")
    return torch.device("cpu")


def make_optimizer(params):
    c = CONFIG
    name, lr, wd = c["optimizer"], c["lr"], c["weight_decay"] or 0.0
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=c["momentum"], weight_decay=wd, nesterov=c["momentum"] > 0)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=c["momentum"], weight_decay=wd)
    return torch.optim.Adagrad(params, lr=lr, weight_decay=wd)


def lr_factor(step, steps_per_epoch, total_steps):
    """The learning-rate schedule: what to multiply the base rate by at each optimizer step."""
    c = CONFIG
    sched = c["scheduler"]
    warm = (c.get("warmup_epochs") or 0) * steps_per_epoch
    if sched == "onecycle":
        up = max(1, int(0.3 * total_steps))
        if step < up:
            return 0.04 + 0.96 * step / up
        p = min(1.0, (step - up) / max(1, total_steps - up))
        return 1e-3 + (1 - 1e-3) * 0.5 * (1 + math.cos(math.pi * p))
    if warm > 0 and step < warm:
        return (step + 1) / warm
    if sched == "cosine":
        p = min(1.0, (step - warm) / max(1, total_steps - warm))
        return 0.5 * (1 + math.cos(math.pi * p))
    if sched == "step":
        return c["gamma"] ** math.floor(step / steps_per_epoch / c["step_size"])
    if sched == "exponential":
        return c["gamma"] ** (step / steps_per_epoch)
    return 1.0


def loss_fn(logits, y):
    flat, yf = logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
    if CONFIG["loss"] == "mse":
        target = F.one_hot(yf, logits.shape[-1]).float()
        if CONFIG["label_smoothing"]:
            target = target * (1 - CONFIG["label_smoothing"]) + CONFIG["label_smoothing"] / logits.shape[-1]
        return F.mse_loss(flat, target)
    return F.cross_entropy(flat, yf, label_smoothing=CONFIG["label_smoothing"] or 0.0)


@torch.no_grad()
def evaluate(net, x_all, y_all, device, batch_size=512):
    net.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for i in range(0, len(x_all), batch_size):
        x, y = prepare(x_all[i : i + batch_size], device), y_all[i : i + batch_size].to(device)
        logits = net(x)
        n = y.numel()
        loss_sum += loss_fn(logits, y).item() * n
        correct += (logits.argmax(-1) == y).sum().item()
        count += n
    net.train()
    return loss_sum / max(1, count), correct / max(1, count)


def set_dropout(net, p):
    """CONFIG["dropout"], when set, overrides the drop probability of every dropout layer."""
    if p is None:
        return
    for m in net.modules():
        if isinstance(m, nn.Dropout):
            m.p = p
        elif isinstance(m, nn.MultiheadAttention):
            m.dropout = p
        elif isinstance(m, nn.RNNBase) and m.num_layers > 1:
            m.dropout = p


def main():
    c = CONFIG
    device = pick_device(c["device"])
    torch.use_deterministic_algorithms(True, warn_only=True)

    train_x, train_y, val_x, val_y, test_x, test_y = load_data()
    torch.manual_seed(c["seed"])  # one seed drives weight init and dropout...
    net = Net()
    set_dropout(net, c.get("dropout"))
    net.to(device)
    print(f"{sum(p.numel() for p in net.parameters()):,} weights · {len(train_x):,} training examples · {device}")

    n = len(train_x)
    bs = min(c["batch_size"], n)
    steps_per_epoch = math.ceil(n / bs)
    total_steps = steps_per_epoch * c["epochs"]
    if c.get("max_steps"):
        total_steps = min(total_steps, c["max_steps"])

    opt = make_optimizer(net.parameters())
    plateau = c["scheduler"] == "plateau"
    if plateau:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=c["gamma"], patience=1)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, steps_per_epoch, total_steps))

    shuffle = torch.Generator().manual_seed(c["seed"])  # ...and the order of the examples
    aug = torch.Generator().manual_seed(c["seed"] + 1) if c.get("augment") and __IS_IMAGE__ else None

    step, best_val = 0, 0.0
    net.train()
    for epoch in range(c["epochs"]):
        if step >= total_steps:
            break
        started = time.time()
        perm = torch.randperm(n, generator=shuffle)
        loss_sum, correct, count = 0.0, 0, 0
        for b in range(steps_per_epoch):
            idx = perm[b * bs : (b + 1) * bs]
            x, y = prepare(train_x[idx], device, aug), train_y[idx].to(device)
            logits = net(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if c.get("grad_clip"):
                nn.utils.clip_grad_norm_(net.parameters(), c["grad_clip"])
            opt.step()
            if not plateau:
                sched.step()
            step += 1
            loss_sum += loss.item() * y.numel()
            correct += (logits.argmax(-1) == y).sum().item()
            count += y.numel()
            if step >= total_steps:
                break
        val_loss, val_acc = evaluate(net, val_x, val_y, device)
        if plateau:
            sched.step(val_loss)
        best_val = max(best_val, val_acc)
        print(f"epoch {epoch + 1:>3}/{c['epochs']}  train loss {loss_sum / count:.4f}  acc {correct / count:.4f}  |  "
              f"val loss {val_loss:.4f}  acc {val_acc:.4f}  |  {time.time() - started:.1f}s")

    test_loss, test_acc = evaluate(net, test_x, test_y, device)
    print(f"\\ntest loss {test_loss!r}\\ntest accuracy {test_acc!r}\\nbest validation accuracy {best_val!r}")
    torch.save({"model": net.state_dict(), "config": CONFIG, "test_acc": test_acc, "test_loss": test_loss}, "model.pt")
    print("saved model.pt")
    return test_loss, test_acc


if __name__ == "__main__":
    main()
'''

_GENERATE = '''"""Generate text with the model trained by train.py.

    python generate.py "ROMEO:" --length 400 --temperature 0.8

The model continues the prompt one character at a time. It only ever sees the last
SEQ_LEN characters. Lower temperature = safer and more repetitive; higher = more varied.
"""
import argparse

import torch

from train import SEQ_LEN, Net, load_vocab


@torch.no_grad()
def generate(net, vocab, prompt, length=300, temperature=0.8, seed=0):
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = [stoi[c] for c in prompt if c in stoi] or [stoi.get("\\n", 0)]
    seq = torch.tensor(ids, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(length):
        logits = net(seq[-SEQ_LEN:].unsqueeze(0))[0, -1]
        if temperature <= 0:
            nxt = logits.argmax().view(1)
        else:
            nxt = torch.multinomial((logits / temperature).softmax(-1), 1, generator=gen)
        seq = torch.cat([seq, nxt])
    return "".join(vocab[i] for i in seq[len(ids):].tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", default="ROMEO:\\n")
    ap.add_argument("--length", type=int, default=300, help="characters to write")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default="model.pt")
    args = ap.parse_args()
    net = Net()
    net.load_state_dict(torch.load(args.weights, map_location="cpu")["model"])
    net.eval()
    prompt = args.prompt.replace("\\\\n", "\\n")
    print(prompt + generate(net, load_vocab(), prompt, args.length, args.temperature, args.seed))


if __name__ == "__main__":
    main()
'''

_PREDICT_IMAGE = '''"""Classify your own image with the model trained by train.py.

    python predict.py photo.png            # add --invert for dark ink on light paper

The image is center-cropped and shrunk to what the model was trained on (__SHAPE__).
"""
import argparse

import torch
from PIL import Image

from train import CLASS_NAMES, Net, prepare

C, H, W = __CHW__


def load_image(path, invert=False):
    img = Image.open(path).convert("L" if C == 1 else "RGB")
    s = max(W / img.width, H / img.height)
    img = img.resize((max(W, round(img.width * s)), max(H, round(img.height * s))))
    left, top = (img.width - W) // 2, (img.height - H) // 2
    img = img.crop((left, top, left + W, top + H))
    x = torch.tensor(list(img.getdata()), dtype=torch.uint8)
    x = x.view(H, W, C).permute(2, 0, 1)
    return (255 - x if invert else x).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--invert", action="store_true", help="flip dark-on-light to light-on-dark")
    ap.add_argument("--weights", default="model.pt")
    args = ap.parse_args()
    net = Net()
    net.load_state_dict(torch.load(args.weights, map_location="cpu")["model"])
    net.eval()
    with torch.no_grad():
        probs = net(prepare(load_image(args.image, args.invert), "cpu")).softmax(-1)[0]
    for p, i in zip(*probs.sort(descending=True)):
        print(f"{CLASS_NAMES[i]:>12}  {p.item() * 100:5.1f}%")


if __name__ == "__main__":
    main()
'''

_PREDICT_TEXT = '''"""Classify your own text with the model trained by train.py.

    python predict.py "Stocks rally as tech earnings beat forecasts"
"""
import argparse

import torch

from train import CLASS_NAMES, Net, encode, load_vocab, tokenize


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text")
    ap.add_argument("--weights", default="model.pt")
    args = ap.parse_args()
    net = Net()
    net.load_state_dict(torch.load(args.weights, map_location="cpu")["model"])
    net.eval()
    stoi = {w: i for i, w in enumerate(load_vocab())}
    x = torch.tensor([encode(tokenize(args.text), stoi)], dtype=torch.long)
    with torch.no_grad():
        probs = net(x).softmax(-1)[0]
    for p, i in zip(*probs.sort(descending=True)):
        print(f"{CLASS_NAMES[i]:>12}  {p.item() * 100:5.1f}%")


if __name__ == "__main__":
    main()
'''

_PREDICT_VECTOR = '''"""Classify a point with the model trained by train.py.

    python predict.py 0.5 -1.2
"""
import sys

import torch

from train import CLASS_NAMES, Net


def main():
    net = Net()
    net.load_state_dict(torch.load("model.pt", map_location="cpu")["model"])
    net.eval()
    with torch.no_grad():
        probs = net(torch.tensor([[float(v) for v in sys.argv[1:]]])).softmax(-1)[0]
    for p, i in zip(*probs.sort(descending=True)):
        print(f"{CLASS_NAMES[i]:>4}  {p.item() * 100:5.1f}%")


if __name__ == "__main__":
    main()
'''


def export_scripts(raw_config: dict) -> dict[str, str]:
    """{"train.py": ..., "generate.py" | "predict.py": ..., "README.md": ...} for a run config."""
    cfg = cfgmod.normalize(raw_config)
    if not cfg["custom_blocks"]:
        cfg["custom_blocks"] = custom.sources_for({b["type"] for b in cfg["architecture"]["blocks"]})
    helpers, customs, net_src, info, n_params = _model_code(cfg)
    t = cfg["training"]
    config = {"name": cfg["name"], "dataset": info.id, "train_subset": cfg["dataset"].get("train_subset"), **t}
    legacy = int(cfg.get("init_stream") or 1) < 2 and "environment" in cfg
    ctx = ""
    if customs and "ctx=CTX" in net_src:
        ctx = (
            f"CTX = SimpleNamespace(task={info.task!r}, num_classes={info.num_classes}, vocab_size={info.vocab_size!r}, "
            f"seq_len={info.seq_len!r}, input_kind={info.input_kind!r})  # dataset facts some custom blocks ask for\n"
        )
    image = info.input_kind == "image"
    repro = (
        "With the same seed on the CPU it follows Lego AI's training procedure exactly, so it\nreaches the same numbers as the app."
        if not legacy else
        "Note: this run was made with an earlier version of Lego AI whose weight initialization\nconsumed random numbers in a different order, so results will be close to, not identical to,\nthe original run."
    )
    train = _TRAIN
    for key, val in {
        "__TITLE__": f"{cfg['name']}: {n_params:,} weights, trained on {info.name}.",
        "__TV__": " and torchvision" if image else "", "__TVPIP__": " torchvision" if image else "",
        "__DATASET__": info.name, "__REPRO__": repro,
        "__CONFIG__": pprint.pformat(config, sort_dicts=False, width=100),
        "__NUM_CLASSES__": str(info.num_classes), "__CLASS_NAMES__": repr(info.class_names or [str(i) for i in range(info.num_classes)]),
        "__CTX__": ctx, "__HELPERS__": "\n\n\n".join(x for x in (helpers, customs) if x) or "# (this model uses only standard PyTorch layers)",
        "__NET__": net_src, "__DATA__": _data_code(info), "__IS_IMAGE__": "True" if image else "False",
    }.items():
        train = train.replace(key, val)
    files = {"train.py": train}
    if info.task == "lm":
        files["generate.py"] = _GENERATE
        second = "generate.py"
        use = 'python generate.py "ROMEO:" --length 400'
    elif image:
        files["predict.py"] = _PREDICT_IMAGE.replace("__SHAPE__", _fmt(info.input_shape)).replace("__CHW__", repr(tuple(info.input_shape)))
        second, use = "predict.py", "python predict.py my_image.png"
    elif info.input_kind == "tokens":
        files["predict.py"] = _PREDICT_TEXT
        second, use = "predict.py", 'python predict.py "some text"'
    else:
        files["predict.py"] = _PREDICT_VECTOR
        second, use = "predict.py", "python predict.py 0.5 -1.2"
    files["README.md"] = (
        f"# {cfg['name']}\n\nExported from Lego AI. {n_params:,} weights, trained on {info.name}.\n\n"
        f"```\npip install torch{' torchvision' if image else ''}\npython train.py        # trains and saves model.pt\n{use}\n```\n\n"
        f"- `train.py`: the model as plain PyTorch, the data loading and the training loop. Settings are in `CONFIG` at the top.\n"
        f"- `{second}`: uses the trained weights in `model.pt`.\n- `config.lego.json`: the same model as a Lego AI config (Builder → Import).\n\n"
        "Datasets download into `./data` on first run (set `LEGO_DATA` to use another folder).\n"
    )
    files["config.lego.json"] = json.dumps(cfg, indent=2)
    for name, src in files.items():
        if name.endswith(".py"):
            compile(src, name, "exec")  # fail here, not on the user's machine, if anything is malformed
    return files


def export_zip(raw_config: dict) -> tuple[str, bytes]:
    files = export_scripts(raw_config)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", (raw_config.get("name") or "model")).strip("_").lower() or "model"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, src in files.items():
            z.writestr(f"{slug}/{name}", src)
    return f"{slug}_python.zip", buf.getvalue()
