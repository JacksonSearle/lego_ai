"""Importing your own datasets.

Every source is first read into a small *table* (named columns of numbers, text or images):

* a CSV file
* a plain text file (becomes a character-level language-modeling dataset)
* a zip of images sorted into one folder per class
* a dataset from the Hugging Face Hub (see ``hub.py``)

The table is previewed in the Datasets tab, where the user says which column is the answer
and which are the inputs. ``build`` then turns it into tensors in the same form as the
built-in datasets and stores it under ``data/datasets/custom/<id>/``.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import re
import secrets
import shutil
import statistics
import time
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from . import datasets
from .datasets import CUSTOM_DIR, DatasetInfo

EXAMPLES = Path(__file__).resolve().parent / "examples"
MAX_ROWS = 60000
MAX_CLASSES = 200
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
TOKEN = datasets._TOKEN

# Uploaded tables waiting to be imported: token → table (only the last few are kept).
_staged: dict[str, dict] = {}


class ImportError_(ValueError):
    """Something about the file that the user can fix."""


# ---------------------------------------------------------------------------
# Reading sources into tables
# ---------------------------------------------------------------------------
def _is_number(v) -> bool:
    if isinstance(v, (int, float)):
        return True
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _table(columns: list[str], rows: list[list], source: str, split: list[str] | None = None, class_names: dict | None = None) -> dict:
    """Describe a table: work out each column's type and a few facts for the preview."""
    if not rows:
        raise ImportError_("The file has no rows of data.")
    rows = rows[:MAX_ROWS]
    cols = []
    for j, name in enumerate(columns):
        vals = [r[j] for r in rows if j < len(r)]
        present = [v for v in vals if v is not None and v != ""]
        if present and isinstance(present[0], (bytes, bytearray)):
            kind = "image"
        elif present and sum(_is_number(v) for v in present) >= 0.98 * len(present):
            kind = "number"
        else:
            kind = "text"
        uniq = len({str(v) for v in present[:5000]}) if kind != "image" else 0
        words = statistics.fmean(len(str(v).split()) for v in present[:500]) if kind == "text" and present else 0
        cols.append({
            "name": name, "type": kind, "unique": uniq, "missing": len(vals) - len(present),
            "avg_words": round(words, 1),
            # Few distinct values: this could be the answer column.
            "label_like": kind != "image" and 2 <= uniq <= min(MAX_CLASSES, max(2, len(rows) // 3)) and not (kind == "number" and any("." in str(v) for v in present[:200])),
            "class_names": (class_names or {}).get(name),
        })
    return {"columns": cols, "rows": rows, "source": source, "split": split if split and len(split) == len(rows) else None}


def read_csv(text: str, source: str = "CSV file") -> dict:
    text = text.lstrip("﻿")
    try:
        dialect = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise ImportError_("This does not look like a table: it needs a header row and at least one row of data.")
    width = max(len(r) for r in rows[:50])
    first = rows[0]
    # A header is a first row of names: no numbers in it.
    if len(first) == width and not any(_is_number(c) for c in first):
        header, body = [c.strip() or f"column {i + 1}" for i, c in enumerate(first)], rows[1:]
    else:
        header, body = [f"column {i + 1}" for i in range(width)], rows
    body = [[c.strip() for c in r] + [""] * (width - len(r)) for r in body if len(r) <= width]
    seen: dict[str, int] = {}
    for i, h in enumerate(header):  # make duplicate names distinct
        seen[h] = seen.get(h, 0) + 1
        if seen[h] > 1:
            header[i] = f"{h} ({seen[h]})"
    return _table(header, body, source)


def read_text(text: str, source: str = "text file") -> dict:
    if len(text) < 2000:
        raise ImportError_("A language-modeling dataset needs more text: at least a few thousand characters, ideally a few hundred thousand.")
    return {"columns": [], "rows": [], "source": source, "split": None, "text": text}


def read_zip(blob: bytes, source: str = "zip of images") -> dict:
    try:
        z = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        raise ImportError_("This is not a zip file.") from None
    rows, split = [], []
    for name in sorted(z.namelist()):
        parts = [p for p in name.split("/") if p]
        if name.endswith("/") or not name.lower().endswith(IMAGE_EXT) or any(p.startswith((".", "__MACOSX")) for p in parts):
            continue
        if len(parts) < 2:
            continue  # an image that is not inside a class folder
        which = "test" if any(p.lower() in ("test", "val", "valid", "validation") for p in parts[:-2]) else "train"
        rows.append([z.read(name), parts[-2]])
        split.append(which)
        if len(rows) >= MAX_ROWS:
            break
    if not rows:
        raise ImportError_("No images found. Put the pictures in one folder per class (for example cats/1.jpg, dogs/1.jpg) and zip the folders.")
    return _table(["image", "label"], rows, source, split if "test" in split else None)


def stage(table: dict) -> str:
    token = secrets.token_hex(8)
    _staged[token] = table
    for old in list(_staged)[:-3]:
        del _staged[old]
    return token


def staged(token: str) -> dict:
    if token not in _staged:
        raise ImportError_("That upload has expired. Choose the file again.")
    return _staged[token]


def _thumb(blob: bytes, size: int = 48) -> dict | None:
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(blob)).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    orig = list(img.size)
    img = img.resize((size, size))
    x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(size, size, 3).permute(2, 0, 1)
    return {"image": x.flatten().tolist(), "shape": [3, size, size], "size": orig}


def preview(token: str) -> dict:
    """What the Datasets tab shows before importing: columns, a few rows and a suggested setup."""
    t = staged(token)
    if "text" in t:
        text = t["text"]
        return {
            "token": token, "source": t["source"], "kind": "text", "n_rows": len(text), "columns": [], "rows": [],
            "sample": text[:600], "n_chars": len(text), "vocab": len(set(text)),
            "suggest": {"label": None, "inputs": []},
        }
    cols = t["columns"]
    names = [c["name"] for c in cols]
    rows = []
    for r in t["rows"][:8]:
        rows.append([_thumb(v) if isinstance(v, (bytes, bytearray)) else (str(v)[:120] if v is not None else "") for v in r])
    # Suggest the answer column: one called label/class/target..., else the last column with few values.
    likely = [c["name"] for c in cols if c["label_like"]]
    named = [n for n in likely if re.search(r"label|class|target|species|category|sentiment|^y$", n, re.I)]
    label = (named or likely[-1:] or [None])[0]
    rest = [c for c in cols if c["name"] != label]
    images = [c["name"] for c in rest if c["type"] == "image"]
    texts = sorted((c for c in rest if c["type"] == "text" and c["avg_words"] >= 3), key=lambda c: -c["avg_words"])
    numbers = [c["name"] for c in rest if c["type"] == "number" and not re.fullmatch(r"(?i)id|index|unnamed.*|", c["name"])]
    inputs = images[:1] or ([texts[0]["name"]] if texts else numbers)
    return {
        "token": token, "source": t["source"], "kind": "table", "n_rows": len(t["rows"]), "has_split": bool(t["split"]),
        "columns": [{k: v for k, v in c.items()} for c in cols], "rows": rows, "names": names,
        "suggest": {"label": label, "inputs": inputs},
    }


# ---------------------------------------------------------------------------
# Building a dataset from a table
# ---------------------------------------------------------------------------
def _slug(name: str) -> str:
    base = "u_" + (re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "dataset")[:40]
    slug, i = base, 2
    while slug in datasets.DATASETS:
        slug, i = f"{base}_{i}", i + 1
    return slug


def _split(n: int, marks: list[str] | None, frac: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices of the training and test rows: the source's own split if it has one, else a fixed random one."""
    if marks:
        test = torch.tensor([i for i, m in enumerate(marks) if m == "test"], dtype=torch.long)
        train = torch.tensor([i for i, m in enumerate(marks) if m != "test"], dtype=torch.long)
        if len(test) and len(train):
            return train, test
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    n_test = max(1, int(round(n * frac)))
    return perm[n_test:], perm[:n_test]


def _images(blobs: list[bytes], size: int, gray: bool) -> torch.Tensor:
    from PIL import Image

    out = torch.empty(len(blobs), 1 if gray else 3, size, size, dtype=torch.uint8)
    for i, b in enumerate(blobs):
        try:
            img = Image.open(io.BytesIO(b)).convert("L" if gray else "RGB")
        except Exception:  # noqa: BLE001
            raise ImportError_(f"Image number {i + 1} could not be opened.") from None
        s = size / min(img.size)  # shrink the short side to `size`, then crop the middle
        img = img.resize((max(size, round(img.width * s)), max(size, round(img.height * s))))
        left, top = (img.width - size) // 2, (img.height - size) // 2
        img = img.crop((left, top, left + size, top + size))
        x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
        out[i] = x.view(size, size, -1).permute(2, 0, 1)
    return out


def build(token: str, opts: dict) -> DatasetInfo:
    t = staged(token)
    name = (opts.get("name") or "").strip() or "My dataset"
    ds_id = _slug(name)
    frac = min(0.5, max(0.05, float(opts.get("test_fraction") or 0.15)))
    common = dict(source=t["source"], custom=True, difficulty=2, targets=None)

    if "text" in t:  # character-level language modeling
        text = t["text"]
        vocab = sorted(set(text))
        if len(vocab) > 3000:
            raise ImportError_(f"This text uses {len(vocab):,} different characters, which is too many for a character-level model.")
        stoi = {c: i for i, c in enumerate(vocab)}
        ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        cut = int(len(ids) * 0.9)
        raw = {"train_ids": ids[:cut].clone(), "test_ids": ids[cut:].clone(), "vocab": vocab}
        hi = int(min(512, max(8, (len(ids) - cut - 1) // 4)))
        L = min(64, hi)
        info = DatasetInfo(
            ds_id, name, opts.get("description") or f"{len(text):,} characters of your own text. Character-level language modeling: predict the next character.",
            "lm", "tokens", [L], len(vocab), vocab_size=len(vocab), seq_len=L, seq_len_range=[8, hi], tokenizer="char", train_tokens=cut,
            difficulty_note="Your own text. Like Tiny Shakespeare, next-character accuracy is never close to 100%; watch the loss and the generated samples.", **common,
        )
        return _store(info, raw)

    cols = {c["name"]: (j, c) for j, c in enumerate(t["columns"])}
    label = opts.get("label")
    inputs = [c for c in (opts.get("inputs") or []) if c in cols and c != label]
    if label not in cols:
        raise ImportError_("Choose which column holds the answer the model should learn to predict.")
    if not inputs:
        raise ImportError_("Choose at least one input column.")
    rows = [r for r in t["rows"] if r[cols[label][0]] not in (None, "")]
    marks = [m for m, r in zip(t["split"], t["rows"]) if r[cols[label][0]] not in (None, "")] if t["split"] else None
    lj, lcol = cols[label]
    values = [r[lj] for r in rows]
    names = lcol.get("class_names")
    if names and all(_is_number(v) and 0 <= int(float(v)) < len(names) for v in values[:2000]):
        y = torch.tensor([int(float(v)) for v in values], dtype=torch.long)
        class_names = [str(n) for n in names]
    else:
        keys = sorted({str(v) for v in values}, key=lambda s: (not _is_number(s), float(s) if _is_number(s) else 0, s))
        index = {k: i for i, k in enumerate(keys)}
        y = torch.tensor([index[str(v)] for v in values], dtype=torch.long)
        class_names = keys
    k = len(class_names)
    if k < 2:
        raise ImportError_(f"The answer column “{label}” has only one value, so there is nothing to learn.")
    if k > MAX_CLASSES:
        raise ImportError_(f"The answer column “{label}” has {k:,} different values. This app handles up to {MAX_CLASSES} classes: is it really a category?")
    if len(rows) < 20:
        raise ImportError_("A dataset needs at least 20 rows.")
    tr, te = _split(len(rows), marks, frac)
    types = {cols[c][1]["type"] for c in inputs}
    major = torch.bincount(y[tr], minlength=k).max().item() / len(tr)
    note = f"Your own dataset. Always guessing the most common class would score {major * 100:.0f}%, so anything above that is learned."

    if types == {"image"}:
        j = cols[inputs[0]][0]
        size = int(opts.get("image_size") or 32)
        if size not in (16, 28, 32, 48, 64):
            raise ImportError_("Image size must be 16, 28, 32, 48 or 64.")
        gray = bool(opts.get("grayscale"))
        x = _images([r[j] for r in rows], size, gray)
        f = x[tr].float().div(255)
        mean, std = f.mean(dim=(0, 2, 3)).tolist(), f.std(dim=(0, 2, 3)).clamp_min(1e-3).tolist()
        raw = {"train_x": x[tr].contiguous(), "train_y": y[tr], "test_x": x[te].contiguous(), "test_y": y[te]}
        info = DatasetInfo(
            ds_id, name, opts.get("description") or f"{len(rows):,} of your own images in {k} classes, {size}×{size} {'grayscale' if gray else 'color'}.",
            "classification", "image", [1 if gray else 3, size, size], k, class_names=class_names,
            mean=[round(m, 4) for m in mean], std=[round(s, 4) for s in std], augment=not gray, difficulty_note=note, **common,
        )
    elif types == {"text"} and len(inputs) == 1:
        j = cols[inputs[0]][0]
        toks = [TOKEN.findall(str(r[j]).replace("\\", " ").lower()) for r in rows]
        max_len = int(min(256, max(8, int(opts.get("max_len") or 64))))
        counts = Counter(w for i in tr.tolist() for w in toks[i])
        vocab = ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(20000)]
        stoi = {w: i for i, w in enumerate(vocab)}
        x = torch.zeros(len(rows), max_len, dtype=torch.int16)
        for i, ws in enumerate(toks):
            ids = [stoi.get(w, 1) for w in ws[:max_len]]
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.int16)
        raw = {"train_x": x[tr].contiguous(), "train_y": y[tr], "test_x": x[te].contiguous(), "test_y": y[te], "vocab": vocab}
        L = min(64, max_len)
        info = DatasetInfo(
            ds_id, name, opts.get("description") or f"{len(rows):,} of your own texts in {k} classes, as word tokens.",
            "classification", "tokens", [L], k, vocab_size=len(vocab), seq_len=L, class_names=class_names,
            seq_len_range=[8, max_len] if max_len > 8 else None, tokenizer="word", difficulty_note=note, **common,
        )
    elif types == {"number"}:
        js = [cols[c][0] for c in inputs]
        x = torch.tensor([[float(r[j]) if _is_number(r[j]) else float("nan") for j in js] for r in rows], dtype=torch.float32)
        mean = x[tr].nanmean(0)
        x = torch.where(x.isnan(), mean.expand_as(x), x)  # a missing number becomes the column's average
        std = x[tr].std(0).clamp_min(1e-6)
        x = (x - mean) / std
        raw = {"train_x": x[tr].contiguous(), "train_y": y[tr], "test_x": x[te].contiguous(), "test_y": y[te]}
        info = DatasetInfo(
            ds_id, name, opts.get("description") or f"{len(rows):,} rows of your own data: {len(inputs)} numbers in, one of {k} classes out.",
            "classification", "vector", [len(inputs)], k, class_names=class_names, difficulty_note=note, **common,
        )
        info.source = f"{t['source']} · inputs: {', '.join(inputs)}"
    else:
        raise ImportError_("Choose inputs of one kind: several number columns, or one text column, or one image column.")
    return _store(info, raw)


def _store(info: DatasetInfo, raw: dict) -> DatasetInfo:
    d = CUSTOM_DIR / info.id
    d.mkdir(parents=True, exist_ok=True)
    torch.save(raw, d / "data.pt")
    (d / "meta.json").write_text(json.dumps({**asdict(info), "created_at": time.time()}, indent=2))
    datasets.register(info)
    built = datasets.load(info.id)  # also checks that it loads
    info.train_size, info.test_size = built.info.train_size, built.info.test_size
    (d / "meta.json").write_text(json.dumps({**asdict(info), "created_at": time.time()}, indent=2))
    datasets.register(info)
    return info


def load_all():
    """Register every stored custom dataset (called when the server starts)."""
    if not CUSTOM_DIR.exists():
        return
    fields = DatasetInfo.__dataclass_fields__
    for meta in sorted(CUSTOM_DIR.glob("*/meta.json")):
        try:
            m = json.loads(meta.read_text())
            if (meta.parent / "data.pt").exists():
                datasets.register(DatasetInfo(**{k: v for k, v in m.items() if k in fields}))
        except (OSError, ValueError, TypeError):
            continue


def delete(dataset_id: str):
    info = datasets.DATASETS.get(dataset_id)
    if not info or not info.custom:
        raise ImportError_("Only datasets you imported can be deleted.")
    datasets.unregister(dataset_id)
    shutil.rmtree(CUSTOM_DIR / dataset_id, ignore_errors=True)


def from_upload(filename: str, content_b64: str) -> str:
    """Stage an uploaded file (sent as base64) and return its token."""
    blob = base64.b64decode(content_b64)
    low = filename.lower()
    if low.endswith(".zip"):
        return stage(read_zip(blob, filename))
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        text = blob.decode("latin-1")
    if low.endswith((".csv", ".tsv")):
        return stage(read_csv(text, filename))
    if low.endswith((".txt", ".md")):
        return stage(read_text(text, filename))
    raise ImportError_("Supported files: .csv or .tsv (a table), .txt (text to learn to continue), .zip (images in one folder per class).")


def example() -> str:
    """Stage the bundled worked example (the Iris flower measurements)."""
    return stage(read_csv((EXAMPLES / "iris.csv").read_text(), "iris.csv (worked example)"))
