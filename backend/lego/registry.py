"""Block registry.

Every block — built-in or user-defined — is described by a ``BlockSpec``. The
spec carries everything the UI needs to render the block (name, colour,
category, parameter descriptions/defaults) and a ``build`` callable that turns
an input shape + parameters into an ``nn.Module``.

Input dimensions are never parameters: the model builder walks the stack from
the input upward and hands each block the shape coming into it, so users only
choose things like "output features" or "hidden size".

Build signature::

    build(in_shape: tuple[int, ...], ctx: BuildContext, **params) -> nn.Module

``in_shape`` excludes the batch dimension. ``ctx`` exposes dataset facts
(number of classes, vocabulary size, task type) for blocks that need them.
"""
from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch.nn as nn

# ---------------------------------------------------------------------------
# Activation functions. Kept in one table so any block can expose an
# "activation" parameter and new activations show up everywhere at once.
# ---------------------------------------------------------------------------
ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
    "elu": nn.ELU,
    "mish": nn.Mish,
    "softplus": nn.Softplus,
    "identity": nn.Identity,
}


def make_activation(name: str) -> nn.Module:
    if name in (None, "", "none"):
        return nn.Identity()
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Options: {', '.join(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


@dataclass
class Param:
    """A user-tunable block parameter, rendered as a form field in the UI.

    type: "int" | "float" | "bool" | "choice" | "activation" | "str"
    ``activation`` is a choice whose options come from ``ACTIVATIONS``.
    """

    name: str
    type: str
    default: Any
    label: str = ""
    help: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[Any] | None = None

    def to_json(self) -> dict:
        d = asdict(self)
        d["label"] = self.label or self.name.replace("_", " ").capitalize()
        if self.type == "activation":
            d["choices"] = ["none", *ACTIVATIONS.keys()]
        return d

    def coerce(self, value: Any) -> Any:
        if value is None:
            return self.default
        if self.type == "int":
            v = int(value)
        elif self.type == "float":
            v = float(value)
        elif self.type == "bool":
            v = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
        else:
            v = value
        if self.type in ("int", "float"):
            if self.min is not None and v < self.min:
                raise ValueError(f"{self.label or self.name} must be ≥ {self.min}")
            if self.max is not None and v > self.max:
                raise ValueError(f"{self.label or self.name} must be ≤ {self.max}")
        if self.type == "choice" and self.choices and v not in self.choices:
            raise ValueError(f"{self.label or self.name} must be one of {self.choices}")
        return v


@dataclass
class BuildContext:
    task: str  # "classification" | "lm"
    num_classes: int
    vocab_size: int | None = None
    seq_len: int | None = None
    input_kind: str = "vector"  # "image" | "tokens" | "vector"


@dataclass
class BlockSpec:
    type: str
    name: str
    category: str
    color: str
    description: str
    build: Callable[..., nn.Module]
    params: list[Param] = field(default_factory=list)
    # What the block expects to receive, shown as a hint in the UI.
    expects: str = "any"
    paper: str | None = None
    source: str = "builtin"  # "builtin" | "custom"
    source_file: str | None = None
    # Which kinds of task the block is typically used for: "image", "text", "general".
    tags: list[str] = field(default_factory=lambda: ["general"])
    year: int | None = None

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "name": self.name,
            "category": self.category,
            "color": self.color,
            "description": self.description,
            "params": [p.to_json() for p in self.params],
            "expects": self.expects,
            "paper": self.paper,
            "source": self.source,
            "source_file": self.source_file,
            "tags": self.tags,
            "year": self.year,
        }

    def resolve_params(self, raw: dict | None) -> dict:
        raw = raw or {}
        return {p.name: p.coerce(raw.get(p.name)) for p in self.params}

    def instantiate(self, in_shape: tuple[int, ...], ctx: BuildContext, params: dict) -> nn.Module:
        kwargs = self.resolve_params(params)
        sig = inspect.signature(self.build)
        if "ctx" in sig.parameters:
            kwargs["ctx"] = ctx
        return self.build(tuple(in_shape), **kwargs)


REGISTRY: dict[str, BlockSpec] = {}

CATEGORY_ORDER = [
    "Dense",
    "Convolution",
    "Pooling",
    "Recurrent",
    "Sequence",
    "Attention",
    "Normalization",
    "Utility",
    "Head",
    "Custom",
]


def register(spec: BlockSpec, *, replace: bool = False) -> BlockSpec:
    if spec.type in REGISTRY and not replace and REGISTRY[spec.type].source == "builtin":
        raise ValueError(f"Block type '{spec.type}' is already a built-in block")
    REGISTRY[spec.type] = spec
    return spec


def block(
    type: str,
    name: str,
    *,
    category: str,
    color: str,
    description: str,
    params: list[Param] | None = None,
    expects: str = "any",
    paper: str | None = None,
):
    """Decorator for built-in block builders."""

    def deco(fn):
        register(
            BlockSpec(
                type=type,
                name=name,
                category=category,
                color=color,
                description=description,
                build=fn,
                params=params or [],
                expects=expects,
                paper=paper,
            )
        )
        return fn

    return deco


def get_spec(type: str) -> BlockSpec:
    if type not in REGISTRY:
        raise KeyError(f"Unknown block type '{type}'. Is it a custom block that was deleted?")
    return REGISTRY[type]
