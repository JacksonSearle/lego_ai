"""Turn an architecture (list of blocks) into a PyTorch model.

Architecture JSON::

    {"blocks": [{"id": "b1", "type": "conv2d", "params": {...}, "repeat": 2}, ...]}

Blocks are listed from input (index 0) to output. Shapes are inferred by running
a tiny dummy batch through each block as it is built, so blocks never need
their input size spelled out and custom blocks get shape checking for free.

``LegoNet`` keeps one entry per block (a Sequential of its repeats) keyed by the
block id, which leaves room for graph features such as skip connections
between arbitrary blocks later on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .registry import BuildContext, get_spec


@dataclass
class BlockReport:
    id: str
    type: str
    in_shape: list[int] | None = None
    out_shape: list[int] | None = None
    params: int = 0
    error: str | None = None


@dataclass
class InspectResult:
    ok: bool
    blocks: list[BlockReport] = field(default_factory=list)
    input_shape: list[int] = field(default_factory=list)
    output_shape: list[int] | None = None
    expected_output: list[int] = field(default_factory=list)
    total_params: int = 0
    error: str | None = None

    def to_json(self):
        return {
            "ok": self.ok,
            "blocks": [b.__dict__ for b in self.blocks],
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "expected_output": self.expected_output,
            "total_params": self.total_params,
            "error": self.error,
        }


class LegoNet(nn.Module):
    def __init__(self, ids: list[str], modules: list[nn.Module]):
        super().__init__()
        self.order = list(ids)
        self.blocks = nn.ModuleDict({_key(i): m for i, m in zip(ids, modules)})

    def forward(self, x):
        for i in self.order:
            x = self.blocks[_key(i)](x)
        return x


def _key(block_id: str) -> str:
    # ModuleDict keys cannot contain "."
    return "b_" + str(block_id).replace(".", "_")


def _dummy(shape, kind):
    if kind == "tokens":
        return torch.zeros(2, *shape, dtype=torch.long)
    return torch.randn(2, *shape)


def expected_output(ctx: BuildContext) -> list[int]:
    if ctx.task == "lm":
        return [ctx.seq_len, ctx.vocab_size]
    return [ctx.num_classes]


def build(arch: dict, input_shape, ctx: BuildContext, *, strict: bool = True, overlay: dict | None = None):
    """Build the model. Returns (model, InspectResult).

    With ``strict`` a shape error raises ValueError; otherwise the report
    records it and building stops at the failing block. ``overlay`` maps block
    types to specs that take precedence over the registry (used to rebuild a run
    from the custom-block source embedded in its config).
    """
    report = InspectResult(ok=True, input_shape=list(input_shape), expected_output=expected_output(ctx))
    shape = tuple(input_shape)
    kind = ctx.input_kind
    ids, modules = [], []
    failed = False
    for raw in arch.get("blocks", []):
        br = BlockReport(id=str(raw.get("id")), type=raw.get("type"), in_shape=list(shape))
        report.blocks.append(br)
        if failed:
            br.in_shape = None
            continue
        try:
            spec = (overlay or {}).get(raw["type"]) or get_spec(raw["type"])
            repeat = max(1, int(raw.get("repeat", 1) or 1))
            reps = []
            for _ in range(repeat):
                if kind == "tokens" and raw["type"] != "embedding" and spec.expects != "any" and spec.expects != "tokens":
                    raise ValueError(f"{spec.name} cannot read raw token ids. Add a Token Embedding first.")
                m = spec.instantiate(shape, ctx, raw.get("params") or {})
                with torch.no_grad():
                    was = m.training
                    m.eval()
                    y = m(_dummy(shape, kind))
                    m.train(was)
                if not isinstance(y, torch.Tensor):
                    raise ValueError(f"{spec.name} must return a single tensor, got {type(y).__name__}.")
                shape = tuple(y.shape[1:])
                kind = "tokens" if y.dtype == torch.long else "float"
                reps.append(m)
            module = reps[0] if len(reps) == 1 else nn.Sequential(*reps)
            br.out_shape = list(shape)
            br.params = sum(p.numel() for p in module.parameters())
            ids.append(br.id)
            modules.append(module)
        except Exception as e:  # noqa: BLE001  - any build error is reported to the user
            br.error = str(e) or type(e).__name__
            report.ok = False
            report.error = f"Block {len(report.blocks)} ({raw.get('type')}): {br.error}"
            failed = True
            if strict:
                raise ValueError(report.error) from e

    report.output_shape = None if failed else list(shape)
    report.total_params = sum(b.params for b in report.blocks)
    if not failed:
        if kind == "tokens":
            report.ok = False
            report.error = "The network outputs token ids. Add blocks (start with a Token Embedding)."
        elif list(shape) != report.expected_output:
            report.ok = False
            report.error = (
                f"Output shape {_fmt(shape)} does not match the dataset's target {_fmt(report.expected_output)}. "
                "Finish with an Output Head block."
            )
        if not report.ok and strict:
            raise ValueError(report.error)
    return LegoNet(ids, modules), report


def inspect(arch: dict, input_shape, ctx: BuildContext, overlay: dict | None = None) -> InspectResult:
    from .rng import isolated  # avoid disturbing the random streams of running jobs

    with isolated():
        torch.manual_seed(0)
        _, report = build(arch, input_shape, ctx, strict=False, overlay=overlay)
    return report


def _fmt(shape):
    return "×".join(str(s) for s in shape) if shape else "scalar"


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def activation_bytes_per_sample(model: nn.Module, input_shape, kind) -> int:
    """Rough activation memory per example: sum of every leaf module's output."""
    total = 0

    def hook(_m, _i, out):
        nonlocal total
        if isinstance(out, torch.Tensor):
            total += out.numel() * out.element_size()
        elif isinstance(out, (tuple, list)):
            for o in out:
                if isinstance(o, torch.Tensor):
                    total += o.numel() * o.element_size()

    handles = [m.register_forward_hook(hook) for m in model.modules() if not list(m.children())]
    try:
        with torch.no_grad():
            model.eval()
            device = next((p.device for p in model.parameters()), torch.device("cpu"))
            model(_dummy(input_shape, kind)[:1].to(device))
            model.train()
    finally:
        for h in handles:
            h.remove()
    return total

