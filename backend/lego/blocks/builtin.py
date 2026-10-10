"""Built-in blocks.

Shape conventions (batch dimension omitted):
    image     (C, H, W)
    sequence  (T, D)
    tokens    (T,)   integer token ids
    vector    (D,)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import BuildContext, Param, block, make_activation


class ShapeError(ValueError):
    """Raised when a block cannot accept the shape flowing into it."""


def _need_image(in_shape, name):
    if len(in_shape) != 3:
        raise ShapeError(f"{name} needs an image (C×H×W) but receives {_fmt(in_shape)}.")


def _need_sequence(in_shape, name, hint=""):
    if len(in_shape) == 3:
        raise ShapeError(
            f"{name} needs a sequence (T×D) but receives an image {_fmt(in_shape)}. "
            "Add 'Image → Sequence' or 'Patch Embedding' first."
        )
    if len(in_shape) != 2:
        raise ShapeError(f"{name} needs a sequence (T×D) but receives {_fmt(in_shape)}.{hint}")


def _fmt(shape):
    return "×".join(str(s) for s in shape) if shape else "scalar"


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------
class LinearAct(nn.Module):
    def __init__(self, d_in, d_out, bias, activation):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        self.act = make_activation(activation)

    def forward(self, x):
        return self.act(self.linear(x))


@block(
    "linear",
    "Linear",
    category="Dense",
    color="#3b82f6",
    description="Fully-connected layer y = act(Wx + b). Applied to the last dimension, so it works on vectors and on every step of a sequence.",
    expects="vector or sequence",
    params=[
        Param("out_features", "int", 128, "Output features", "Width of the layer (number of neurons).", min=1),
        Param("activation", "activation", "relu", "Activation", "Non-linearity applied after the layer. 'none' gives a purely linear layer."),
        Param("bias", "bool", True, "Bias", "Learn an additive bias term."),
    ],
    paper="https://psycnet.apa.org/record/1959-09865-001",
)
def build_linear(in_shape, out_features, activation, bias):
    if len(in_shape) == 3:
        raise ShapeError(f"Linear receives an image {_fmt(in_shape)}. Add a Flatten block first.")
    return LinearAct(in_shape[-1], out_features, bias, activation)


class MLPBlock(nn.Module):
    def __init__(self, d_in, hidden, d_out, activation, dropout, residual):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), make_activation(activation), nn.Dropout(dropout), nn.Linear(hidden, d_out)
        )
        self.residual = residual

    def forward(self, x):
        y = self.net(x)
        return x + y if self.residual else y


@block(
    "mlp",
    "MLP Block",
    category="Dense",
    color="#1d4ed8",
    description="Two-layer perceptron: Linear → activation → dropout → Linear. Optionally residual (adds the input back), as in transformer feed-forward layers.",
    expects="vector or sequence",
    params=[
        Param("hidden", "int", 256, "Hidden size", "Width of the inner layer.", min=1),
        Param("out_features", "int", 0, "Output features", "Output width. 0 keeps the same width as the input (needed for residual).", min=0),
        Param("activation", "activation", "relu", "Activation", "Non-linearity between the two layers."),
        Param("dropout", "float", 0.0, "Dropout", "Probability of zeroing hidden units during training.", min=0, max=0.95, step=0.05),
        Param("residual", "bool", False, "Residual", "Add the block's input to its output (requires output = input width)."),
    ],
    paper="https://www.nature.com/articles/323533a0",
)
def build_mlp(in_shape, hidden, out_features, activation, dropout, residual):
    if len(in_shape) == 3:
        raise ShapeError(f"MLP Block receives an image {_fmt(in_shape)}. Add a Flatten block first.")
    d_in = in_shape[-1]
    d_out = out_features or d_in
    if residual and d_out != d_in:
        raise ShapeError(f"Residual MLP needs output features = input features ({d_in}); set output features to 0.")
    return MLPBlock(d_in, hidden, d_out, activation, dropout, residual)


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------
def _pad(kernel_size, padding):
    return kernel_size // 2 if padding == "same" else 0


@block(
    "conv2d",
    "Conv2D",
    category="Convolution",
    color="#f97316",
    description="2D convolution: slides learned filters over the image, sharing weights across positions. Optionally followed by batch norm and an activation.",
    expects="image",
    params=[
        Param("out_channels", "int", 32, "Filters", "Number of output channels (feature maps).", min=1),
        Param("kernel_size", "int", 3, "Kernel size", "Height/width of each filter.", min=1, max=15),
        Param("stride", "int", 1, "Stride", "Step between filter positions. 2 halves the resolution.", min=1, max=8),
        Param("padding", "choice", "same", "Padding", "'same' pads so stride-1 convs keep the size; 'valid' uses no padding.", choices=["same", "valid"]),
        Param("batch_norm", "bool", False, "Batch norm", "Normalize activations per channel (Ioffe & Szegedy, 2015)."),
        Param("activation", "activation", "relu", "Activation", "Non-linearity after the convolution."),
    ],
    paper="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf",
)
def build_conv2d(in_shape, out_channels, kernel_size, stride, padding, batch_norm, activation):
    _need_image(in_shape, "Conv2D")
    layers = [nn.Conv2d(in_shape[0], out_channels, kernel_size, stride, _pad(kernel_size, padding), bias=not batch_norm)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(make_activation(activation))
    return nn.Sequential(*layers)


@block(
    "depthwise_conv",
    "Depthwise-Separable Conv",
    category="Convolution",
    color="#fdba74",
    description="MobileNet-style conv: a per-channel spatial filter followed by a 1×1 channel mixer. Far fewer parameters than a full conv.",
    expects="image",
    params=[
        Param("out_channels", "int", 64, "Filters", "Output channels of the 1×1 pointwise conv.", min=1),
        Param("kernel_size", "int", 3, "Kernel size", "Size of the depthwise filter.", min=1, max=15),
        Param("stride", "int", 1, "Stride", "Stride of the depthwise conv.", min=1, max=8),
        Param("activation", "activation", "relu", "Activation", "Non-linearity after each conv."),
    ],
    paper="https://arxiv.org/abs/1704.04861",
)
def build_depthwise(in_shape, out_channels, kernel_size, stride, activation):
    _need_image(in_shape, "Depthwise-Separable Conv")
    c = in_shape[0]
    return nn.Sequential(
        nn.Conv2d(c, c, kernel_size, stride, kernel_size // 2, groups=c, bias=False),
        nn.BatchNorm2d(c),
        make_activation(activation),
        nn.Conv2d(c, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        make_activation(activation),
    )


class ResidualConv(nn.Module):
    def __init__(self, c_in, c_out, stride, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act = make_activation(activation)
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + self.shortcut(x))


@block(
    "resblock",
    "Residual Conv Block",
    category="Convolution",
    color="#c2410c",
    description="ResNet basic block: two 3×3 convs with batch norm, plus a skip connection that adds the input back. Makes very deep CNNs trainable.",
    expects="image",
    params=[
        Param("out_channels", "int", 0, "Filters", "Output channels. 0 keeps the input channel count.", min=0),
        Param("stride", "int", 1, "Stride", "2 halves the resolution (the skip path is projected to match).", min=1, max=4),
        Param("activation", "activation", "relu", "Activation", "Non-linearity used inside the block."),
    ],
    paper="https://arxiv.org/abs/1512.03385",
)
def build_resblock(in_shape, out_channels, stride, activation):
    _need_image(in_shape, "Residual Conv Block")
    return ResidualConv(in_shape[0], out_channels or in_shape[0], stride, activation)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------
@block(
    "maxpool2d",
    "Max Pool",
    category="Pooling",
    color="#f59e0b",
    description="Downsamples an image by taking the maximum in each window.",
    expects="image",
    params=[
        Param("kernel_size", "int", 2, "Window", "Size of the pooling window.", min=1, max=8),
        Param("stride", "int", 0, "Stride", "0 means same as the window size.", min=0, max=8),
    ],
)
def build_maxpool(in_shape, kernel_size, stride):
    _need_image(in_shape, "Max Pool")
    if in_shape[1] < kernel_size or in_shape[2] < kernel_size:
        raise ShapeError(f"Image {_fmt(in_shape)} is smaller than the pooling window {kernel_size}.")
    return nn.MaxPool2d(kernel_size, stride or kernel_size)


@block(
    "avgpool2d",
    "Average Pool",
    category="Pooling",
    color="#fbbf24",
    description="Downsamples an image by averaging each window.",
    expects="image",
    params=[
        Param("kernel_size", "int", 2, "Window", "Size of the pooling window.", min=1, max=8),
        Param("stride", "int", 0, "Stride", "0 means same as the window size.", min=0, max=8),
    ],
)
def build_avgpool(in_shape, kernel_size, stride):
    _need_image(in_shape, "Average Pool")
    if in_shape[1] < kernel_size or in_shape[2] < kernel_size:
        raise ShapeError(f"Image {_fmt(in_shape)} is smaller than the pooling window {kernel_size}.")
    return nn.AvgPool2d(kernel_size, stride or kernel_size)


class GlobalAvgPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3))


@block(
    "global_avgpool",
    "Global Average Pool",
    category="Pooling",
    color="#d97706",
    description="Averages each channel over all positions: C×H×W → C. The usual bridge from conv features to a classifier.",
    expects="image",
    paper="https://arxiv.org/abs/1312.4400",
)
def build_gap(in_shape):
    _need_image(in_shape, "Global Average Pool")
    return GlobalAvgPool()


class SeqPool(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == "mean":
            return x.mean(dim=1)
        if self.mode == "max":
            return x.max(dim=1).values
        if self.mode == "first":
            return x[:, 0]
        return x[:, -1]


@block(
    "seq_pool",
    "Sequence Pool",
    category="Pooling",
    color="#0891b2",
    description="Collapses a sequence T×D into a single vector D (mean, max, first or last step).",
    expects="sequence",
    params=[Param("mode", "choice", "mean", "Mode", "How to combine the time steps.", choices=["mean", "max", "last", "first"])],
)
def build_seq_pool(in_shape, mode):
    _need_sequence(in_shape, "Sequence Pool")
    return SeqPool(mode)


# ---------------------------------------------------------------------------
# Recurrent
# ---------------------------------------------------------------------------
class Recurrent(nn.Module):
    def __init__(self, kind, d_in, hidden, layers, bidirectional, return_sequences, dropout, nonlinearity="tanh"):
        super().__init__()
        cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        kw = {"nonlinearity": nonlinearity} if kind == "rnn" else {}
        self.rnn = cls(
            d_in, hidden, num_layers=layers, batch_first=True, bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0, **kw,
        )
        self.kind = kind
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional

    def forward(self, x):
        out, h = self.rnn(x)
        if self.return_sequences:
            return out
        if self.kind == "lstm":
            h = h[0]
        if self.bidirectional:
            return torch.cat([h[-2], h[-1]], dim=-1)
        return h[-1]


def _recurrent_params(default_hidden=128):
    return [
        Param("hidden", "int", default_hidden, "Hidden size", "Size of the recurrent state.", min=1),
        Param("layers", "int", 1, "Stacked layers", "Number of recurrent layers stacked inside this block.", min=1, max=8),
        Param("bidirectional", "bool", False, "Bidirectional", "Also read the sequence backwards and concatenate. Not allowed for language modelling (it would see the future)."),
        Param("return_sequences", "bool", True, "Return sequence", "Output every time step (T×H). Turn off to output only the final state (H)."),
        Param("dropout", "float", 0.0, "Dropout", "Dropout between stacked layers (only when layers > 1).", min=0, max=0.9, step=0.05),
    ]


def _build_recurrent(kind, label, in_shape, ctx, hidden, layers, bidirectional, return_sequences, dropout, **kw):
    _need_sequence(in_shape, label)
    if bidirectional and ctx.task == "lm":
        raise ShapeError("Bidirectional recurrent layers would see future characters in language modelling.")
    return Recurrent(kind, in_shape[-1], hidden, layers, bidirectional, return_sequences, dropout, **kw)


@block(
    "rnn",
    "RNN (Elman)",
    category="Recurrent",
    color="#86efac",
    description="Vanilla recurrent network: h_t = tanh(W x_t + U h_{t-1}). Simple but struggles with long-range memory (vanishing gradients).",
    expects="sequence",
    params=[*_recurrent_params(), Param("nonlinearity", "choice", "tanh", "Nonlinearity", "Activation used in the recurrence.", choices=["tanh", "relu"])],
    paper="https://doi.org/10.1207/s15516709cog1402_1",
)
def build_rnn(in_shape, ctx, nonlinearity, **p):
    return _build_recurrent("rnn", "RNN", in_shape, ctx, nonlinearity=nonlinearity, **p)


@block(
    "lstm",
    "LSTM",
    category="Recurrent",
    color="#16a34a",
    description="Long Short-Term Memory: gated recurrent cell with a separate memory lane, so information and gradients survive many steps.",
    expects="sequence",
    params=_recurrent_params(),
    paper="https://www.bioinf.jku.at/publications/older/2604.pdf",
)
def build_lstm(in_shape, ctx, **p):
    return _build_recurrent("lstm", "LSTM", in_shape, ctx, **p)


@block(
    "gru",
    "GRU",
    category="Recurrent",
    color="#059669",
    description="Gated Recurrent Unit: a lighter LSTM variant with two gates and no separate cell state.",
    expects="sequence",
    params=_recurrent_params(),
    paper="https://arxiv.org/abs/1406.1078",
)
def build_gru(in_shape, ctx, **p):
    return _build_recurrent("gru", "GRU", in_shape, ctx, **p)


# ---------------------------------------------------------------------------
# Sequence / embeddings
# ---------------------------------------------------------------------------
@block(
    "embedding",
    "Token Embedding",
    category="Sequence",
    color="#ec4899",
    description="Looks up a learned vector for each token id: T → T×D. Required first block for text datasets.",
    expects="tokens",
    params=[Param("dim", "int", 128, "Embedding dim", "Length of each token's vector.", min=1)],
    paper="https://arxiv.org/abs/1301.3781",
)
def build_embedding(in_shape, ctx, dim):
    if ctx.vocab_size is None:
        raise ShapeError("Token Embedding only works on text datasets (this dataset has no vocabulary).")
    if len(in_shape) != 1:
        raise ShapeError(f"Token Embedding needs token ids (T) but receives {_fmt(in_shape)}.")
    return nn.Embedding(ctx.vocab_size, dim)


class ImageRows(nn.Module):
    def forward(self, x):
        b, c, h, w = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, h, c * w)


@block(
    "image_to_sequence",
    "Image → Sequence",
    category="Sequence",
    color="#be185d",
    description="Reads an image row by row: C×H×W → H×(C·W). Lets RNNs and transformers process images as sequences.",
    expects="image",
)
def build_image_rows(in_shape):
    _need_image(in_shape, "Image → Sequence")
    return ImageRows()


class PatchEmbed(nn.Module):
    def __init__(self, c, patch, dim):
        super().__init__()
        self.proj = nn.Conv2d(c, dim, patch, patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


@block(
    "patch_embed",
    "Patch Embedding",
    category="Sequence",
    color="#db2777",
    description="Vision Transformer front-end: cuts the image into P×P patches and linearly projects each one, giving a sequence of patch tokens.",
    expects="image",
    params=[
        Param("patch_size", "int", 4, "Patch size", "Height/width of each patch. Must divide the image size.", min=1, max=32),
        Param("dim", "int", 64, "Embedding dim", "Size of each patch token.", min=1),
    ],
    paper="https://arxiv.org/abs/2010.11929",
)
def build_patch_embed(in_shape, patch_size, dim):
    _need_image(in_shape, "Patch Embedding")
    if in_shape[1] % patch_size or in_shape[2] % patch_size:
        raise ShapeError(f"Patch size {patch_size} does not divide the image size {in_shape[1]}×{in_shape[2]}.")
    return PatchEmbed(in_shape[0], patch_size, dim)


class PositionalEncoding(nn.Module):
    def __init__(self, t, d, kind, dropout):
        super().__init__()
        if kind == "learned":
            self.pe = nn.Parameter(torch.zeros(1, t, d))
            nn.init.normal_(self.pe, std=0.02)
        else:
            pos = torch.arange(t).unsqueeze(1)
            div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
            pe = torch.zeros(1, t, d)
            pe[0, :, 0::2] = torch.sin(pos * div)
            pe[0, :, 1::2] = torch.cos(pos * div)[:, : d // 2]
            self.register_buffer("pe", pe)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.shape[1]])


@block(
    "positional_encoding",
    "Positional Encoding",
    category="Sequence",
    color="#f472b6",
    description="Adds position information to each step so attention (which is order-blind) knows where tokens are.",
    expects="sequence",
    params=[
        Param("kind", "choice", "learned", "Type", "'learned' trains a vector per position; 'sinusoidal' uses fixed sine waves (Vaswani et al.).", choices=["learned", "sinusoidal"]),
        Param("dropout", "float", 0.0, "Dropout", "Dropout applied after adding positions.", min=0, max=0.9, step=0.05),
    ],
    paper="https://arxiv.org/abs/1706.03762",
)
def build_posenc(in_shape, kind, dropout):
    _need_sequence(in_shape, "Positional Encoding")
    return PositionalEncoding(in_shape[0], in_shape[1], kind, dropout)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
def _is_causal(causal, ctx):
    return ctx.task == "lm" if causal == "auto" else causal == "yes"


class TransformerBlock(nn.Module):
    def __init__(self, d, heads, ff_mult, dropout, causal, activation, norm_first):
        super().__init__()
        self.norm_first = norm_first
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * ff_mult), make_activation(activation), nn.Dropout(dropout), nn.Linear(d * ff_mult, d))
        self.drop = nn.Dropout(dropout)
        self.causal = causal

    def _attn(self, x):
        mask = None
        if self.causal:
            t = x.shape[1]
            mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), 1)
        return self.attn(x, x, x, attn_mask=mask, need_weights=False)[0]

    def forward(self, x):
        if self.norm_first:
            x = x + self.drop(self._attn(self.ln1(x)))
            return x + self.drop(self.ff(self.ln2(x)))
        x = self.ln1(x + self.drop(self._attn(x)))
        return self.ln2(x + self.drop(self.ff(x)))


@block(
    "transformer",
    "Transformer Block",
    category="Attention",
    color="#7c3aed",
    description="Multi-head self-attention + feed-forward MLP, each wrapped in a residual connection and layer norm. The building block of GPT, BERT and ViT.",
    expects="sequence",
    params=[
        Param("heads", "int", 4, "Attention heads", "Number of parallel attention heads. Must divide the model width.", min=1, max=32),
        Param("ff_mult", "int", 4, "FF expansion", "Feed-forward hidden size as a multiple of the model width.", min=1, max=8),
        Param("dropout", "float", 0.1, "Dropout", "Dropout on attention and feed-forward outputs.", min=0, max=0.9, step=0.05),
        Param("causal", "choice", "auto", "Causal mask", "Prevent each position from attending to later ones. 'auto' = on for language modelling.", choices=["auto", "yes", "no"]),
        Param("activation", "activation", "gelu", "Activation", "Non-linearity in the feed-forward layer."),
        Param("norm_first", "bool", True, "Pre-norm", "Apply LayerNorm before each sub-layer (more stable) instead of after (original 2017 design)."),
    ],
    paper="https://arxiv.org/abs/1706.03762",
)
def build_transformer(in_shape, ctx, heads, ff_mult, dropout, causal, activation, norm_first):
    _need_sequence(in_shape, "Transformer Block", " Add a Token Embedding or Patch Embedding first.")
    d = in_shape[-1]
    if d % heads:
        raise ShapeError(f"Model width {d} is not divisible by {heads} heads.")
    return TransformerBlock(d, heads, ff_mult, dropout, _is_causal(causal, ctx), activation, norm_first)


class SelfAttention(nn.Module):
    def __init__(self, d, heads, dropout, causal):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.causal = causal

    def forward(self, x):
        mask = None
        if self.causal:
            t = x.shape[1]
            mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), 1)
        h = self.ln(x)
        return x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]


@block(
    "self_attention",
    "Self-Attention",
    category="Attention",
    color="#a78bfa",
    description="Just the attention sub-layer (pre-norm, residual): every position gathers information from every other position.",
    expects="sequence",
    params=[
        Param("heads", "int", 4, "Attention heads", "Number of parallel heads. Must divide the model width.", min=1, max=32),
        Param("dropout", "float", 0.0, "Dropout", "Dropout on attention weights.", min=0, max=0.9, step=0.05),
        Param("causal", "choice", "auto", "Causal mask", "'auto' = on for language modelling.", choices=["auto", "yes", "no"]),
    ],
    paper="https://arxiv.org/abs/1409.0473",
)
def build_self_attention(in_shape, ctx, heads, dropout, causal):
    _need_sequence(in_shape, "Self-Attention")
    if in_shape[-1] % heads:
        raise ShapeError(f"Model width {in_shape[-1]} is not divisible by {heads} heads.")
    return SelfAttention(in_shape[-1], heads, dropout, _is_causal(causal, ctx))


class MixerBlock(nn.Module):
    def __init__(self, t, d, token_hidden, channel_hidden, activation, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.token_mlp = nn.Sequential(nn.Linear(t, token_hidden), make_activation(activation), nn.Dropout(dropout), nn.Linear(token_hidden, t))
        self.ln2 = nn.LayerNorm(d)
        self.channel_mlp = nn.Sequential(nn.Linear(d, channel_hidden), make_activation(activation), nn.Dropout(dropout), nn.Linear(channel_hidden, d))

    def forward(self, x):
        x = x + self.token_mlp(self.ln1(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mlp(self.ln2(x))


@block(
    "mixer",
    "MLP-Mixer Block",
    category="Attention",
    color="#c084fc",
    description="Attention-free alternative: one MLP mixes information across tokens, another across channels. Not causal, so for classification only.",
    expects="sequence",
    params=[
        Param("token_hidden", "int", 64, "Token-mixing hidden", "Hidden size of the MLP that mixes across positions.", min=1),
        Param("channel_hidden", "int", 128, "Channel-mixing hidden", "Hidden size of the MLP that mixes across features.", min=1),
        Param("activation", "activation", "gelu", "Activation", "Non-linearity in both MLPs."),
        Param("dropout", "float", 0.0, "Dropout", "Dropout inside both MLPs.", min=0, max=0.9, step=0.05),
    ],
    paper="https://arxiv.org/abs/2105.01601",
)
def build_mixer(in_shape, ctx, token_hidden, channel_hidden, activation, dropout):
    _need_sequence(in_shape, "MLP-Mixer Block")
    if ctx.task == "lm":
        raise ShapeError("MLP-Mixer mixes across all positions, so it would see future characters in language modelling.")
    return MixerBlock(in_shape[0], in_shape[1], token_hidden, channel_hidden, activation, dropout)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
class SeqBatchNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)

    def forward(self, x):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


@block(
    "batchnorm",
    "Batch Norm",
    category="Normalization",
    color="#14b8a6",
    description="Normalizes each feature/channel using batch statistics, then rescales. Speeds up and stabilises training.",
    paper="https://arxiv.org/abs/1502.03167",
)
def build_batchnorm(in_shape):
    if len(in_shape) == 3:
        return nn.BatchNorm2d(in_shape[0])
    if len(in_shape) == 2:
        return SeqBatchNorm(in_shape[-1])
    return nn.BatchNorm1d(in_shape[0])


@block(
    "layernorm",
    "Layer Norm",
    category="Normalization",
    color="#0f766e",
    description="Normalizes across the features of each example (independent of the batch). Standard in transformers and RNNs.",
    paper="https://arxiv.org/abs/1607.06450",
)
def build_layernorm(in_shape):
    if len(in_shape) == 3:
        return nn.GroupNorm(1, in_shape[0])
    return nn.LayerNorm(in_shape[-1])


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
@block(
    "flatten",
    "Flatten",
    category="Utility",
    color="#64748b",
    description="Unrolls everything into one long vector (e.g. 1×28×28 → 784).",
)
def build_flatten(in_shape):
    return nn.Flatten()


@block(
    "dropout",
    "Dropout",
    category="Utility",
    color="#94a3b8",
    description="Randomly zeros a fraction of values during training to reduce overfitting.",
    params=[Param("p", "float", 0.5, "Drop probability", "Fraction of values zeroed during training.", min=0, max=0.95, step=0.05)],
    paper="https://jmlr.org/papers/v15/srivastava14a.html",
)
def build_dropout(in_shape, p):
    return nn.Dropout(p)


@block(
    "activation",
    "Activation",
    category="Utility",
    color="#eab308",
    description="A standalone non-linearity.",
    params=[Param("fn", "activation", "relu", "Function", "Which non-linearity to apply.")],
)
def build_activation(in_shape, fn):
    return make_activation(fn)


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------
class ClassifierHead(nn.Module):
    def __init__(self, reduce, d_in, d_out, dropout):
        super().__init__()
        self.reduce = reduce
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x):
        if self.reduce == "flatten":
            x = x.flatten(1)
        elif self.reduce == "mean_seq":
            x = x.mean(dim=1)
        elif self.reduce == "mean_img":
            x = x.mean(dim=(2, 3))
        return self.fc(self.drop(x))


@block(
    "classifier",
    "Output Head",
    category="Head",
    color="#ef4444",
    description="Final linear layer sized automatically to the dataset's output (number of classes, or vocabulary size per step for language models).",
    params=[
        Param("pool", "choice", "auto", "Reduce", "How to turn image/sequence features into one vector first. 'auto' = flatten images, average sequences.", choices=["auto", "flatten", "mean"]),
        Param("dropout", "float", 0.0, "Dropout", "Dropout before the final layer.", min=0, max=0.95, step=0.05),
    ],
)
def build_classifier(in_shape, ctx, pool, dropout):
    if ctx.task == "lm":
        _need_sequence(in_shape, "Output Head (language model)", " Language models predict a token at every step.")
        return ClassifierHead("none", in_shape[-1], ctx.vocab_size, dropout)
    if len(in_shape) == 1:
        return ClassifierHead("none", in_shape[0], ctx.num_classes, dropout)
    if len(in_shape) == 3:
        if pool == "mean":
            return ClassifierHead("mean_img", in_shape[0], ctx.num_classes, dropout)
        return ClassifierHead("flatten", math.prod(in_shape), ctx.num_classes, dropout)
    if pool == "flatten":
        return ClassifierHead("flatten", math.prod(in_shape), ctx.num_classes, dropout)
    return ClassifierHead("mean_seq", in_shape[-1], ctx.num_classes, dropout)


# ---------------------------------------------------------------------------
# Task tags and year of introduction (used by the palette filter and the guidebook).
# ---------------------------------------------------------------------------
from ..registry import REGISTRY  # noqa: E402

_META = {
    "linear": (["general"], 1958), "mlp": (["general"], 1986), "activation": (["general"], 1986),
    "dropout": (["general"], 2012), "flatten": (["general"], None), "classifier": (["general"], None),
    "batchnorm": (["general", "image"], 2015), "layernorm": (["general", "text"], 2016),
    "conv2d": (["image"], 1989), "depthwise_conv": (["image"], 2017), "resblock": (["image"], 2015),
    "maxpool2d": (["image"], 1989), "avgpool2d": (["image"], 1989), "global_avgpool": (["image"], 2013),
    "patch_embed": (["image"], 2020), "image_to_sequence": (["image"], 1990), "mixer": (["image"], 2021),
    "embedding": (["text"], 2003), "seq_pool": (["text"], None), "rnn": (["text"], 1990), "lstm": (["text"], 1997),
    "gru": (["text"], 2014), "self_attention": (["text", "image"], 2014),
    "positional_encoding": (["text", "image"], 2017), "transformer": (["text", "image"], 2017),
}
for _type, (_tags, _year) in _META.items():
    REGISTRY[_type].tags = _tags
    REGISTRY[_type].year = _year
