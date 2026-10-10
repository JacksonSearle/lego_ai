"""User-defined blocks written in Python.

A custom block file is ordinary Python. Decorate an ``nn.Module`` subclass
(or a builder function) with ``@lego_block`` and it appears in the builder
palette like any built-in block::

    import torch.nn as nn
    from lego.sdk import Param, lego_block

    @lego_block(name="Gated MLP", color="#0ea5e9", params=[Param("hidden", "int", 256)])
    class GatedMLP(nn.Module):
        def __init__(self, in_shape, hidden=256):
            ...

The constructor always receives ``in_shape`` (input shape without the batch
dimension) first, then the declared params as keyword arguments. Add a ``ctx``
keyword argument to also receive dataset facts (num_classes, vocab_size, task).

Files live in ``data/custom_blocks``. Their source is copied into every run
config that uses them, so a saved run can be rebuilt even if the file changes.
"""
from __future__ import annotations

import hashlib
import inspect
import re
import traceback
import types
from pathlib import Path

import torch.nn as nn

from .registry import REGISTRY, BlockSpec, Param, register

CUSTOM_DIR = Path(__file__).resolve().parent.parent / "data" / "custom_blocks"
_MARK = "__lego_block__"


def lego_block(
    name: str,
    *,
    color: str = "#0ea5e9",
    description: str = "",
    params: list[Param] | None = None,
    expects: str = "any",
    paper: str | None = None,
    type: str | None = None,
    tags: list[str] | None = None,
    parts: list[dict] | None = None,
):
    """Mark a class or builder function as a Lego block."""

    def deco(obj):
        setattr(
            obj,
            _MARK,
            dict(name=name, color=color, description=description or (inspect.getdoc(obj) or "").strip(),
                 params=params or [], expects=expects, paper=paper, type=type, tags=tags or ["general"], parts=parts),
        )
        return obj

    return deco


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _make_builder(obj):
    """Adapt a class/function to the registry's build(in_shape, [ctx], **params) signature."""
    sig = inspect.signature(obj)
    wants_ctx = "ctx" in sig.parameters

    def remember(module, in_shape, params):
        # Lets the exporter write this block as ``ClassName(in_shape, **params)``.
        if isinstance(module, nn.Module) and inspect.isclass(obj) and "_lego_custom" not in module.__dict__:
            object.__setattr__(module, "_lego_custom", {"name": obj.__name__, "in_shape": tuple(in_shape), "params": params, "ctx": wants_ctx})
        return module

    if wants_ctx:
        def build(in_shape, ctx, **params):
            return remember(obj(in_shape, ctx=ctx, **params), in_shape, params)
    else:
        def build(in_shape, **params):
            return remember(obj(in_shape, **params), in_shape, params)
    return build


def exec_source(source: str, filename: str) -> list[BlockSpec]:
    """Execute a custom block file and return the block specs it defines (not yet registered)."""
    mod = types.ModuleType(f"lego_custom_{_slug(filename)}")
    mod.__file__ = filename
    code = compile(source, filename, "exec")
    exec(code, mod.__dict__)  # noqa: S102 - the user runs their own code locally, by design
    specs = []
    for attr, obj in vars(mod).items():
        meta = getattr(obj, _MARK, None)
        if meta is None or getattr(obj, "__module__", mod.__name__) != mod.__name__:
            continue
        for p in meta["params"]:
            if not isinstance(p, Param):
                raise TypeError(f"{attr}: params must be lego.sdk.Param objects")
        specs.append(
            BlockSpec(
                type=meta["type"] or f"custom:{_slug(attr)}",
                name=meta["name"],
                category="Custom",
                color=meta["color"],
                description=meta["description"] or "User-defined block.",
                build=_make_builder(obj),
                params=meta["params"],
                expects=meta["expects"],
                paper=meta["paper"],
                source="custom",
                source_file=filename,
                tags=meta.get("tags") or ["general"],
                parts=meta.get("parts"),
            )
        )
    if not specs:
        raise ValueError("No blocks found. Decorate an nn.Module class with @lego_block(...).")
    return specs


def _unregister_file(filename: str):
    for t in [t for t, s in REGISTRY.items() if s.source == "custom" and s.source_file == filename]:
        del REGISTRY[t]


def install(filename: str, source: str) -> list[BlockSpec]:
    """Validate + register a file's blocks, replacing what the file previously defined."""
    specs = exec_source(source, filename)
    for s in specs:
        existing = REGISTRY.get(s.type)
        if existing and (existing.source == "builtin" or existing.source_file != filename):
            raise ValueError(f"Block type '{s.type}' is already defined by {existing.source_file or 'a built-in block'}.")
    _unregister_file(filename)
    for s in specs:
        register(s, replace=True)
    return specs


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    if not name.endswith(".py"):
        name += ".py"
    if not re.fullmatch(r"[A-Za-z0-9_\-]+\.py", name):
        raise ValueError("File name may only contain letters, numbers, '_' and '-'.")
    return name


def save(filename: str, source: str) -> list[BlockSpec]:
    filename = _safe_filename(filename)
    specs = install(filename, source)
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    (CUSTOM_DIR / filename).write_text(source)
    return specs


def delete(filename: str):
    filename = _safe_filename(filename)
    _unregister_file(filename)
    (CUSTOM_DIR / filename).unlink(missing_ok=True)


def load_all() -> dict[str, str]:
    """Load every saved file at startup. Returns {filename: error} for files that failed."""
    errors = {}
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(CUSTOM_DIR.glob("*.py")):
        try:
            install(path.name, path.read_text())
        except Exception:  # noqa: BLE001
            errors[path.name] = traceback.format_exc(limit=3)
    return errors


def list_files() -> list[dict]:
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for path in sorted(CUSTOM_DIR.glob("*.py")):
        blocks = [s.type for s in REGISTRY.values() if s.source_file == path.name]
        out.append({"filename": path.name, "source": path.read_text(), "blocks": blocks})
    return out


def sources_for(block_types: set[str]) -> dict[str, dict]:
    """Source code of the custom files that define the given block types (for run configs)."""
    files = {}
    for t in block_types:
        spec = REGISTRY.get(t)
        if spec and spec.source == "custom" and spec.source_file not in files:
            src = (CUSTOM_DIR / spec.source_file).read_text()
            files[spec.source_file] = {"source": src, "sha256": hashlib.sha256(src.encode()).hexdigest()}
    return files


def ensure_from_config(custom_blocks: dict[str, dict]) -> list[str]:
    """Make sure the custom blocks embedded in a run config are registered.

    If a file of that name exists with different contents, the config's version
    is installed under a suffixed filename so the saved run rebuilds exactly.
    Returns notes describing what was installed.
    """
    notes = []
    for filename, entry in (custom_blocks or {}).items():
        src = entry["source"]
        sha = hashlib.sha256(src.encode()).hexdigest()
        path = CUSTOM_DIR / filename
        if path.exists() and hashlib.sha256(path.read_text().encode()).hexdigest() == sha:
            continue
        if not path.exists():
            save(filename, src)
            notes.append(f"Installed custom block file {filename} from the config.")
        else:
            notes.append(
                f"{filename} has changed since this run was saved; the run was built with the saved copy embedded in its config."
            )
    return notes


def build_registry_overlay(custom_blocks: dict[str, dict]) -> dict[str, BlockSpec]:
    """Specs from a config's embedded sources, used to rebuild a run exactly even if files changed."""
    overlay = {}
    for filename, entry in (custom_blocks or {}).items():
        for s in exec_source(entry["source"], filename):
            overlay[s.type] = s
    return overlay


EXAMPLE = '''"""Example custom blocks. Edit freely and press Save: they appear in the
builder palette under "Custom".

Rules:
  * Decorate an nn.Module class with @lego_block(...).
  * __init__ receives in_shape (no batch dim) first, then your params.
  * Optionally accept ctx= to see dataset facts (ctx.num_classes, ctx.vocab_size, ctx.task).
  * forward() takes and returns one tensor (batch first).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lego.sdk import Param, lego_block


@lego_block(
    name="SwiGLU Feed-Forward",
    color="#0ea5e9",
    description="Gated feed-forward layer used in LLaMA/PaLM: (xW ⊙ swish(xV))U, with a residual connection.",
    params=[
        Param("hidden", "int", 256, "Hidden size", "Width of the gated inner layer.", min=1),
        Param("dropout", "float", 0.0, "Dropout", "Dropout on the inner activations.", min=0, max=0.9, step=0.05),
    ],
    paper="https://arxiv.org/abs/2002.05202",
)
class SwiGLU(nn.Module):
    def __init__(self, in_shape, hidden=256, dropout=0.0):
        super().__init__()
        d = in_shape[-1]
        self.norm = nn.LayerNorm(d)
        self.w = nn.Linear(d, hidden)
        self.v = nn.Linear(d, hidden)
        self.u = nn.Linear(hidden, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        return x + self.u(self.drop(self.w(h) * F.silu(self.v(h))))


@lego_block(
    name="Squeeze-Excite",
    color="#0284c7",
    description="Channel attention for CNNs: global-pools each channel, passes it through a tiny MLP, and rescales the channels.",
    params=[Param("reduction", "int", 4, "Reduction", "Bottleneck ratio of the gating MLP.", min=1, max=64)],
    expects="image",
    paper="https://arxiv.org/abs/1709.01507",
)
class SqueezeExcite(nn.Module):
    def __init__(self, in_shape, reduction=4):
        super().__init__()
        if len(in_shape) != 3:
            raise ValueError("Squeeze-Excite needs an image (C×H×W)")
        c = in_shape[0]
        self.fc = nn.Sequential(nn.Linear(c, max(1, c // reduction)), nn.ReLU(), nn.Linear(max(1, c // reduction), c), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean(dim=(2, 3)))
        return x * w[:, :, None, None]
'''


FUSED_TEMPLATE = '''"""{name}: a fused block.

Made in the Builder by fusing {n} blocks into one. It builds exactly those blocks in order.
You can edit PARTS (same format as a model's block list), the name or the color, then Save.
"""
from lego.sdk import compose, lego_block

PARTS = {parts}


@lego_block(
    name={name!r},
    color={color!r},
    description={description!r},
    tags={tags!r},
    parts=PARTS,
)
def {func}(in_shape, ctx=None):
    return compose(in_shape, ctx, PARTS)
'''


def fuse(name: str, color: str, parts: list[dict]) -> BlockSpec:
    """Create (and save) a custom block that is the given blocks fused into one."""
    import json

    name = (name or "").strip()[:40] or "Fused block"
    if len(parts) < 1:
        raise ValueError("Select at least one block to fuse.")
    clean = []
    tags: set[str] = set()
    for b in parts:
        spec = REGISTRY.get(b.get("type"))
        if spec is None:
            raise ValueError(f"Unknown block type {b.get('type')!r}.")
        clean.append({"type": b["type"], "params": spec.resolve_params(b.get("params")), "repeat": max(1, int(b.get("repeat") or 1))})
        tags.update(spec.tags)
    slug = _slug(name) or "fused"
    func = "fused_" + slug
    filename = f"{func}.py"
    n = 2
    while (CUSTOM_DIR / filename).exists() or f"custom:{func}" in REGISTRY:
        func, filename = f"fused_{slug}_{n}", f"fused_{slug}_{n}.py"
        n += 1
    inner = " → ".join((REGISTRY[b["type"]].name + (f" ×{b['repeat']}" if b["repeat"] > 1 else "")) for b in clean)
    source = FUSED_TEMPLATE.format(
        name=name, n=sum(b["repeat"] for b in clean), color=color or "#38bdf8", func=func,
        parts=json.dumps(clean, indent=4).replace("true", "True").replace("false", "False").replace("null", "None"),
        description=f"Fused block: {inner}.", tags=sorted(tags - {"general"}) or ["general"],
    )
    return save(filename, source)[0]


def ensure_example():
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    marker = CUSTOM_DIR / ".example_installed"
    if not marker.exists():
        (CUSTOM_DIR / "examples.py").write_text(EXAMPLE)
        marker.write_text("1")
