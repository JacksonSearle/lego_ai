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
    # Measured while building, per training example: bytes of activations the layers produce,
    # and multiply-adds for one forward pass. Used for memory and time estimates.
    act_bytes_per_sample: int = 0
    flops_per_sample: int = 0

    def to_json(self):
        return {
            "ok": self.ok,
            "blocks": [b.__dict__ for b in self.blocks],
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "expected_output": self.expected_output,
            "total_params": self.total_params,
            "error": self.error,
            "act_bytes_per_sample": self.act_bytes_per_sample,
            "flops_per_sample": self.flops_per_sample,
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


def _dummy(shape, kind, gen: torch.Generator | None = None):
    """A tiny batch used to discover a block's output shape.

    With ``gen`` the values come from a private generator. Without it they are drawn from
    the global random stream, which is how the first version of this app worked: models
    saved back then are rebuilt that way so their weights come out identical.
    """
    if kind == "tokens":
        return torch.zeros(2, *shape, dtype=torch.long)
    return torch.randn(2, *shape, generator=gen) if gen is not None else torch.randn(2, *shape)


def _count(module: nn.Module, output, stats: dict):
    """Add one layer's activation memory and multiply-adds (both per example) to ``stats``."""
    outs = [o for o in (output if isinstance(output, (tuple, list)) else [output]) if isinstance(o, torch.Tensor)]
    if not outs:
        return
    batch = outs[0].shape[0] if outs[0].ndim else 1
    stats["act"] += sum(o.numel() * o.element_size() for o in outs) // max(1, batch)
    out = outs[0]
    per = out.numel() // max(1, batch)
    if isinstance(module, nn.Linear):
        stats["flops"] += per * module.in_features
    elif isinstance(module, nn.Conv2d):
        k = module.kernel_size[0] * module.kernel_size[1]
        stats["flops"] += per * k * module.in_channels // module.groups
    elif isinstance(module, nn.MultiheadAttention):
        t, d = out.shape[1], module.embed_dim
        stats["flops"] += 4 * t * d * d + 2 * t * t * d
    elif isinstance(module, nn.RNNBase):
        gates = {"LSTM": 4, "GRU": 3}.get(module.mode, 1)
        t = out.shape[1] if module.batch_first else out.shape[0]
        dirs = 2 if module.bidirectional else 1
        h = module.hidden_size
        first = gates * h * (module.input_size + h)
        rest = gates * h * (h * dirs + h) * (module.num_layers - 1)
        stats["flops"] += t * dirs * (first + rest)
    else:
        stats["flops"] += per  # normalization, activations, pooling: about one operation per value


def _probe(m: nn.Module, x: torch.Tensor, stats: dict | None):
    """Run the dummy batch through a block (in eval mode), measuring its layers on the way."""
    handles = []
    if stats is not None:
        for sub in m.modules():
            if isinstance(sub, (nn.MultiheadAttention, nn.RNNBase)) or not any(True for _ in sub.children()):
                handles.append(sub.register_forward_hook(lambda mod, _i, out: _count(mod, out, stats)))
    try:
        with torch.no_grad():
            was = m.training
            m.eval()
            y = m(x)
            m.train(was)
    finally:
        for h in handles:
            h.remove()
    return y


def compose(in_shape, ctx: BuildContext, parts: list[dict], overlay: dict | None = None) -> nn.Module:
    """Build a list of blocks into one module (used by fused blocks).

    ``parts`` has the same form as an architecture's block list. Shapes are inferred part by
    part exactly as in a full model, so a fused block works wherever its first part would.
    """
    shape = tuple(in_shape)
    kind = ctx.current_kind or ("tokens" if ctx.input_kind == "tokens" and len(shape) == 1 else "float")
    gen = torch.Generator().manual_seed(0)
    mods = []
    for raw in parts:
        spec = (overlay or {}).get(raw["type"]) or get_spec(raw["type"])
        for _ in range(max(1, int(raw.get("repeat", 1) or 1))):
            ctx.current_kind = kind
            m = spec.instantiate(shape, ctx, raw.get("params") or {})
            y = _probe(m, _dummy(shape, kind, gen), None)
            if not isinstance(y, torch.Tensor):
                raise ValueError(f"{spec.name} must return a single tensor.")
            shape = tuple(y.shape[1:])
            kind = "tokens" if y.dtype == torch.long else "float"
            mods.append(m)
    return mods[0] if len(mods) == 1 else nn.Sequential(*mods)


def expected_output(ctx: BuildContext) -> list[int]:
    if ctx.task == "lm":
        return [ctx.seq_len, ctx.vocab_size]
    return [ctx.num_classes]


def build(arch: dict, input_shape, ctx: BuildContext, *, strict: bool = True, overlay: dict | None = None,
          legacy_stream: bool = False):
    """Build the model. Returns (model, InspectResult).

    With ``strict`` a shape error raises ValueError; otherwise the report
    records it and building stops at the failing block. ``overlay`` maps block
    types to specs that take precedence over the registry (used to rebuild a run
    from the custom-block source embedded in its config). ``legacy_stream`` rebuilds
    a model the way version 1 did, with shape-check batches drawn from the same random
    stream as the weights, so that old runs still reproduce exactly.
    """
    report = InspectResult(ok=True, input_shape=list(input_shape), expected_output=expected_output(ctx))
    shape = tuple(input_shape)
    kind = "tokens" if ctx.input_kind == "tokens" else "float"
    ids, modules = [], []
    failed = False
    stats = {"act": 0, "flops": 0}
    gen = None if legacy_stream else torch.Generator().manual_seed(0)
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
                ctx.current_kind = kind
                m = spec.instantiate(shape, ctx, raw.get("params") or {})
                y = _probe(m, _dummy(shape, kind, gen), stats)
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
    report.act_bytes_per_sample = stats["act"]
    report.flops_per_sample = stats["flops"]
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

