"""Example custom blocks. Edit freely and press Save — they appear in the
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

@lego_block(
    name="Jackson-Custom",
    color="#a83232",
    description="Herp a derp dumb description",
    params=[Param("reduction", "int", 4, "Reduction", "Bottleneck ratio of the gating MLP.", min=1, max=64)],
    paper="https://arxiv.org/abs/1709.01507",
)
class JacksonCustom(nn.Module):
    def __init__(self, in_shape, reduction=4):
        super().__init__()
        c = in_shape[0]
        self.fc = nn.Sequential(nn.Linear(c, max(1, c // reduction)), nn.ReLU(), nn.Linear(max(1, c // reduction), c), nn.Sigmoid())

    def forward(self, x):
        return -x