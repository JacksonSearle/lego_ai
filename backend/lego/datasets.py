"""Built-in datasets.

Each dataset is loaded fully into memory as tensors (they are all small), cached
to ``data/datasets/<id>.pt`` after the first download. Batching is done by
indexing these tensors with a seeded permutation, which keeps training fast and
bit-for-bit reproducible (no DataLoader worker processes to seed).

Splits: ``train`` / ``val`` (a fixed held-out slice of the official training
set, identical for every run so results are comparable) / ``test`` (official
test set, evaluated once at the end of training).
"""
from __future__ import annotations

import csv
import io
import math
import re
import threading
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch

from .registry import BuildContext

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets"
VAL_SPLIT_SEED = 12345


@dataclass
class DatasetInfo:
    id: str
    name: str
    description: str
    task: str  # classification | lm
    input_kind: str  # image | tokens | vector
    input_shape: list[int]
    num_classes: int
    vocab_size: int | None = None
    seq_len: int | None = None
    class_names: list[str] = field(default_factory=list)
    source: str = ""
    train_size: int | None = None
    test_size: int | None = None
    mean: list[float] | None = None
    std: list[float] | None = None
    augment: bool = False  # random crop/flip is sensible for this dataset
    # Text datasets only: the allowed range for the (configurable) sequence length.
    seq_len_range: list[int] | None = None
    # How hard it is to get good results: 1 easy, 2 medium, 3 hard, with guidance for learners.
    difficulty: int = 1
    difficulty_note: str = ""
    # Validation accuracy you can expect: a trivial baseline, a good result and a great one.
    targets: dict | None = None

    def ctx(self) -> BuildContext:
        return BuildContext(
            task=self.task,
            num_classes=self.num_classes,
            vocab_size=self.vocab_size,
            seq_len=self.seq_len,
            input_kind=self.input_kind,
        )

    def to_json(self):
        return asdict(self)


DATASETS: dict[str, DatasetInfo] = {
    "mnist": DatasetInfo(
        "mnist", "MNIST", "70k handwritten digits, 28×28 grayscale, 10 classes. The classic first benchmark.",
        "classification", "image", [1, 28, 28], 10, class_names=[str(i) for i in range(10)],
        source="http://yann.lecun.com/exdb/mnist/", train_size=55000, test_size=10000, mean=[0.1307], std=[0.3081],
        difficulty=1, targets={"baseline": 0.92, "good": 0.98, "great": 0.993},
        difficulty_note="Easy. A single hidden layer reaches 95%+ in seconds on a CPU; a small CNN passes 99%.",
    ),
    "fashion_mnist": DatasetInfo(
        "fashion_mnist", "Fashion-MNIST", "70k 28×28 grayscale clothing images in 10 classes. A harder drop-in for MNIST.",
        "classification", "image", [1, 28, 28], 10,
        class_names=["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"],
        source="https://github.com/zalandoresearch/fashion-mnist", train_size=55000, test_size=10000, mean=[0.2860], std=[0.3530],
        difficulty=2, targets={"baseline": 0.84, "good": 0.90, "great": 0.93},
        difficulty_note="Medium. Same size as MNIST but the classes look alike (shirts vs coats). MLPs stall near 88%; you need a CNN and several epochs for 92%+.",
    ),
    "cifar10": DatasetInfo(
        "cifar10", "CIFAR-10", "60k 32×32 color photos in 10 classes (planes, cars, birds, cats…).",
        "classification", "image", [3, 32, 32], 10,
        class_names=["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
        source="https://www.cs.toronto.edu/~kriz/cifar.html", train_size=45000, test_size=10000,
        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616], augment=True,
        difficulty=3, targets={"baseline": 0.40, "good": 0.75, "great": 0.90},
        difficulty_note="Hard. Real color photos: needs a deep CNN, data augmentation, 15+ epochs and ideally a GPU. Expect minutes to hours, not seconds.",
    ),
    "ag_news": DatasetInfo(
        "ag_news", "AG News", "127k news headlines + descriptions in 4 topics (World, Sports, Business, Sci/Tech), as word tokens.",
        "classification", "tokens", [64], 4, vocab_size=20002, seq_len=64,
        class_names=["World", "Sports", "Business", "Sci/Tech"],
        source="https://arxiv.org/abs/1509.01626", train_size=115000, test_size=7600, seq_len_range=[8, 256],
        difficulty=1, targets={"baseline": 0.85, "good": 0.90, "great": 0.92},
        difficulty_note="Easy. Topic words give it away, so even averaged word vectors reach about 90% in a minute. Bigger models add only a point or two.",
    ),
    "shakespeare": DatasetInfo(
        "shakespeare", "Tiny Shakespeare", "1.1M characters of Shakespeare. Character-level language modeling: predict the next character.",
        "lm", "tokens", [64], 65, vocab_size=65, seq_len=64,
        source="https://github.com/karpathy/char-rnn", train_size=14117, test_size=1742, seq_len_range=[8, 512],
        difficulty=3, targets={"baseline": 0.30, "good": 0.50, "great": 0.60},
        difficulty_note="Hard. Predicting the next character is never certain, so 55 to 60% is excellent. Needs an LSTM or transformer and many minutes of training; text quality keeps improving long after accuracy flattens.",
    ),
    "spirals": DatasetInfo(
        "spirals", "Spirals (2D)", "Three interleaved 2D spirals. Tiny and instant; linear models fail, MLPs succeed.",
        "classification", "vector", [2], 3, class_names=["A", "B", "C"], source="synthetic", train_size=2700, test_size=1000,
        difficulty=1, targets={"baseline": 0.45, "good": 0.95, "great": 0.99},
        difficulty_note="Easy and instant, but only for non-linear models: a linear model is stuck below 50%, a small MLP reaches 99%.",
    ),
}


@dataclass
class Data:
    info: DatasetInfo
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    vocab: list[str] | None = None

    def split(self, name):
        return getattr(self, f"{name}_x"), getattr(self, f"{name}_y")

    def prepare(self, x: torch.Tensor, device, *, augment: bool = False, gen: torch.Generator | None = None):
        """Convert a raw batch (uint8 images / long tokens / float vectors) to model input on ``device``."""
        info = self.info
        if info.input_kind == "image":
            if augment:
                x = _augment(x, gen)
            x = x.to(device, non_blocking=True).float().div_(255.0)
            mean = torch.tensor(info.mean, device=device).view(1, -1, 1, 1)
            std = torch.tensor(info.std, device=device).view(1, -1, 1, 1)
            return (x - mean) / std
        if info.input_kind == "tokens":
            return x.to(device, non_blocking=True).long()  # tokens may be stored as int16 to save memory
        return x.to(device, non_blocking=True)


def _augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random crop (pad 4) + horizontal flip, driven by ``gen`` for reproducibility."""
    n, c, h, w = x.shape
    padded = torch.nn.functional.pad(x, (4, 4, 4, 4))
    dx = torch.randint(0, 9, (n,), generator=gen)
    dy = torch.randint(0, 9, (n,), generator=gen)
    flip = torch.rand(n, generator=gen) < 0.5
    out = torch.empty_like(x)
    for i in range(n):
        img = padded[i, :, dy[i] : dy[i] + h, dx[i] : dx[i] + w]
        out[i] = img.flip(-1) if flip[i] else img
    return out


SHAKESPEARE_TRAIN_CHARS = 1003854  # 90% of the corpus


def with_seq_len(info: DatasetInfo, seq_len: int | None) -> DatasetInfo:
    """A copy of ``info`` for the requested sequence length (text datasets only)."""
    if not info.seq_len_range:
        return info
    lo, hi = info.seq_len_range
    L = int(seq_len or info.seq_len)
    if not lo <= L <= hi:
        raise ValueError(f"Sequence length for {info.name} must be between {lo} and {hi}.")
    out = replace(info, seq_len=L, input_shape=[L])
    if info.id == "shakespeare":
        n = (SHAKESPEARE_TRAIN_CHARS - 1) // L
        out.train_size = n - min(5000, max(1, n // 10))
    return out


_raw: dict[str, dict] = {}
_cache: dict[tuple[str, int | None], Data] = {}
_lock = threading.Lock()

# Text datasets are cached in a form that any sequence length can be cut from.
_CACHE_FILE = {"shakespeare": "shakespeare_v2.pt", "ag_news": "ag_news_v2.pt"}


def load(dataset_id: str, seq_len: int | None = None) -> Data:
    if dataset_id not in DATASETS:
        raise KeyError(f"Unknown dataset '{dataset_id}'")
    info = with_seq_len(DATASETS[dataset_id], seq_len)
    key = (dataset_id, info.seq_len)
    with _lock:
        if key not in _cache:
            # Keep at most a few sequence-length variants of a text dataset in memory.
            for old in [k for k in _cache if k[0] == dataset_id][2:]:
                del _cache[old]
            _cache[key] = _build(info, _load_raw(dataset_id))
        return _cache[key]


def _load_raw(dataset_id: str) -> dict:
    if dataset_id not in _raw:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cache = DATA_DIR / _CACHE_FILE.get(dataset_id, f"{dataset_id}.pt")
        if cache.exists():
            _raw[dataset_id] = torch.load(cache)
        else:
            _raw[dataset_id] = LOADERS[dataset_id]()
            torch.save(_raw[dataset_id], cache)
    return _raw[dataset_id]


def _chunks(seq: torch.Tensor, t: int):
    """Cut a token stream into non-overlapping windows; the target is the same window shifted by one."""
    n = (len(seq) - 1) // t
    return seq[: n * t].view(n, t).contiguous(), seq[1 : n * t + 1].view(n, t).contiguous()


def _build(info: DatasetInfo, raw: dict) -> Data:
    if info.id == "shakespeare":
        tx, ty = _chunks(raw["train_ids"], info.seq_len)
        test_x, test_y = _chunks(raw["test_ids"], info.seq_len)
    elif info.id == "ag_news":
        tx, ty = raw["train_x"][:, : info.seq_len], raw["train_y"]
        test_x, test_y = raw["test_x"][:, : info.seq_len].contiguous(), raw["test_y"]
    else:
        tx, ty, test_x, test_y = raw["train_x"], raw["train_y"], raw["test_x"], raw["test_y"]
    n = len(tx)
    n_val = min(5000, max(1, n // 10))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(VAL_SPLIT_SEED))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    info = replace(info, train_size=len(tr_idx), test_size=len(test_x))
    return Data(
        info,
        tx[tr_idx].contiguous(), ty[tr_idx].contiguous(),
        tx[val_idx].contiguous(), ty[val_idx].contiguous(),
        test_x, test_y,
        vocab=raw.get("vocab"),
    )


# ---------------------------------------------------------------------------
# Loaders - each returns {"train_x", "train_y", "test_x", "test_y"[, "vocab"]}
# ---------------------------------------------------------------------------
def _torchvision(cls_name):
    def load():
        import torchvision

        cls = getattr(torchvision.datasets, cls_name)
        root = DATA_DIR / "raw"
        out = {}
        for split, train in (("train", True), ("test", False)):
            ds = cls(root=str(root), train=train, download=True)
            x = torch.as_tensor(ds.data)
            x = x.unsqueeze(1) if x.ndim == 3 else x.permute(0, 3, 1, 2)
            out[f"{split}_x"] = x.contiguous().to(torch.uint8)
            out[f"{split}_y"] = torch.as_tensor(ds.targets, dtype=torch.long)
        return out

    return load


def _download(url) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def _shakespeare():
    text = _download("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").decode()
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cut = int(len(ids) * 0.9)
    return {"train_ids": ids[:cut].clone(), "test_ids": ids[cut:].clone(), "vocab": vocab}


_TOKEN = re.compile(r"[a-z0-9']+")


def _ag_news():
    info = DATASETS["ag_news"]
    base = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/"

    def rows(name):
        text = _download(base + name).decode("utf-8")
        for label, title, desc in csv.reader(io.StringIO(text)):
            yield int(label) - 1, _TOKEN.findall(f"{title} {desc}".replace("\\", " ").lower())

    train = list(rows("train.csv"))
    test = list(rows("test.csv"))
    counts = Counter(w for _, toks in train for w in toks)
    vocab = ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(info.vocab_size - 2)]
    stoi = {w: i for i, w in enumerate(vocab)}

    max_len = info.seq_len_range[1]

    def encode(data):
        # int16 keeps the full-length cache small (the vocabulary has about 20k entries).
        x = torch.zeros(len(data), max_len, dtype=torch.int16)
        y = torch.tensor([lbl for lbl, _ in data], dtype=torch.long)
        for i, (_, toks) in enumerate(data):
            ids = [stoi.get(w, 1) for w in toks[:max_len]]
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.int16)
        return x, y

    trx, tr_y = encode(train)
    tex, tey = encode(test)
    return {"train_x": trx, "train_y": tr_y, "test_x": tex, "test_y": tey, "vocab": vocab}


def _spirals():
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
    return {"train_x": x[:3000], "train_y": y[:3000], "test_x": x[3000:4000], "test_y": y[3000:4000]}


LOADERS = {
    "mnist": _torchvision("MNIST"),
    "fashion_mnist": _torchvision("FashionMNIST"),
    "cifar10": _torchvision("CIFAR10"),
    "shakespeare": _shakespeare,
    "ag_news": _ag_news,
    "spirals": _spirals,
}


def is_cached(dataset_id: str) -> bool:
    return dataset_id in _raw or (DATA_DIR / _CACHE_FILE.get(dataset_id, f"{dataset_id}.pt")).exists()


def get_info(dataset_id: str, seq_len: int | None = None) -> DatasetInfo:
    if dataset_id not in DATASETS:
        raise KeyError(f"Unknown dataset '{dataset_id}'")
    return with_seq_len(DATASETS[dataset_id], seq_len)


def info_for(ds: dict) -> DatasetInfo:
    """Dataset info for a run config's ``dataset`` section (honors its sequence length)."""
    return get_info(ds["id"], ds.get("seq_len"))


def load_for(ds: dict) -> Data:
    return load(ds["id"], ds.get("seq_len"))
