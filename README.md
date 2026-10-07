# Lego AI

Build neural networks out of coloured blocks, train them with live charts, tune them with Optuna,
and compare them on a leaderboard. Every run is saved as a JSON config that retrains to the
**same numbers, bit for bit**.

```bash
./run.sh            # builds the frontend, serves everything on http://localhost:8000
```

Requires [uv](https://docs.astral.sh/uv/) and Node 20+. Datasets download on first use into
`backend/data/datasets/` (CIFAR-10 is ~170 MB and takes a few minutes).

Development (hot reload on both sides):

```bash
cd backend && uv run uvicorn lego.server:app --reload --port 8000
cd frontend && npm run dev     # http://localhost:5173, proxies /api to :8000
```

## Tabs

| Tab | What it does |
| --- | --- |
| **Build** | Drag blocks from the palette onto a stack that flows **up** from the input (bottom, with its shape) to the output (top, with the dataset's target shape). Shapes are inferred after every edit and shown on the arrows. Click a block for its parameters, with help text and defaults. Use **Repeat** to stack a block N times (shown as ×N beside it). Templates give a proven starting point per dataset, or you can start from scratch. Undo/redo, import/export. |
| **Train** | Hyperparameters (optimizer, LR, scheduler, warm-up, weight decay, dropout override, label smoothing, grad clipping, loss, augmentation, seed, device). **Check what fits in memory** tests a range of batch sizes. Pressing Train checks yours first and suggests alternatives if it won't fit. Turn on **Optuna** to give ranges instead of values. Live loss/accuracy/LR charts. Resume any checkpoint, or load a previous run's config to retrain from scratch. |
| **Leaderboard** | All runs ranked by validation accuracy (or test acc, loss, params, time), filterable by dataset. Click a run for its curves, architecture, hyperparameters and environment, and to open it in the builder, **retrain from scratch** (marked ✓ reproduced when it matches exactly), resume, download the config, or download the weights. |
| **Custom Blocks** | Write blocks in Python (CodeMirror editor). Save, and they show up in the palette like built-ins. |
| **History of AI** | Plays by itself: trains 12 landmark architectures from 1805 to 2021 on the same MNIST budget, with a timeline, live charts, a separate demo leaderboard, and a paragraph and paper link for each. |

## Blocks

Each colour family is one kind of layer:

- **Dense** (blues): Linear, MLP Block (optionally residual)
- **Convolution** (oranges): Conv2D (+BN), Depthwise-Separable Conv, Residual Conv Block
- **Pooling** (ambers/cyan): Max Pool, Average Pool, Global Average Pool, Sequence Pool
- **Recurrent** (greens): RNN (Elman), LSTM, GRU
- **Sequence** (pinks): Token Embedding, Image → Sequence, Patch Embedding, Positional Encoding
- **Attention** (purples): Transformer Block (causal mask is automatic for language modelling), Self-Attention, MLP-Mixer Block
- **Normalization** (teals): Batch Norm, Layer Norm
- **Utility** (greys/yellow): Flatten, Dropout, Activation
- **Head** (red): Output Head, which sizes itself to the dataset (classes, or vocab per step)

Blocks never ask for their input size, because it comes from the block below. Any block with an
activation can use relu, gelu, silu, tanh, sigmoid, leaky_relu, elu, mish, softplus or identity.

## Datasets

MNIST, Fashion-MNIST, CIFAR-10, AG News (word-level topic classification), Tiny Shakespeare
(character-level language modelling), and 2-D Spirals (synthetic, instant). For each, the
validation split is a fixed 5k (or 10%) slice of the training set, the same for every run, and
the official test set is scored once at the end.

## Reproducibility

A run config (`*.lego.json`) records:

- the dataset and optional training subset
- the architecture, as a block list
- every hyperparameter and the seed
- the full **source code of any custom blocks** used, with its sha256
- the environment: Python, torch and numpy versions, platform, thread count

Training uses one seed for weight init, shuffling, augmentation and dropout. Batches come from
seeded in-memory permutations (no DataLoader workers), deterministic algorithms are on, and
checkpoints save every RNG state. As a result:

- retraining a config gives identical metrics (tested: same test loss to the last bit)
- resuming from a checkpoint gives the same result as never stopping
- other jobs running at the same time, and the builder's live shape checks, don't disturb a run,
  because each run draws from its own random stream (`lego/rng.py`)

Bitwise reproducibility holds on **CPU** with the same torch version and thread count. MPS/CUDA
are faster but some kernels are non-deterministic. If a config's recorded environment differs
from the current one, the app warns you.

## Custom blocks

```python
import torch.nn as nn
from lego.sdk import Param, lego_block, make_activation

@lego_block(name="Gated MLP", color="#0ea5e9",
            params=[Param("hidden", "int", 256, "Hidden size", "Width of the gate.", min=1)])
class GatedMLP(nn.Module):
    def __init__(self, in_shape, hidden=256):        # in_shape excludes the batch dim
        super().__init__()
        d = in_shape[-1]
        self.a, self.b, self.out = nn.Linear(d, hidden), nn.Linear(d, hidden), nn.Linear(hidden, d)
    def forward(self, x):
        return x + self.out(self.a(x) * self.b(x).sigmoid())
```

Add a `ctx=None` argument to read `ctx.num_classes`, `ctx.vocab_size`, `ctx.task`. Raise
`ValueError` for unsupported shapes, and the message appears on the block. Files live in
`backend/data/custom_blocks/`. `examples.py` (SwiGLU, Squeeze-Excite) is installed on first run.

## Extending

- **New built-in block**: add a function decorated with `@block(...)` in
  `backend/lego/blocks/builtin.py`. The palette, parameter panel, shape inference and saved
  configs all pick it up automatically.
- **New activation**: add it to `ACTIVATIONS` in `backend/lego/registry.py`. It appears in every
  activation dropdown.
- **Skip connections between arbitrary blocks**: `LegoNet` (`backend/lego/model.py`) keys each
  block by id, so a block field like `"skip_from": "<id>"` could add the stored output of an
  earlier block in `forward`.
- **New hyperparameter**: add an entry to `HYPERPARAMS` in `backend/lego/config.py`, with a
  `tune` range to make it Optuna-searchable, and use it in `train.py`.
- **New dataset**: add a `DatasetInfo` and a loader in `backend/lego/datasets.py`, plus templates
  in `templates.py`.

## Layout

```
backend/lego/
  registry.py     block/param specs, activations
  blocks/builtin.py  built-in blocks
  model.py        architecture JSON → nn.Module, shape inference
  custom.py, sdk.py  user Python blocks
  datasets.py     dataset loading/caching
  config.py       run-config schema, hyperparameter specs, environment capture
  train.py        training loop, checkpoints, resume, reproduction check
  rng.py          per-run random streams
  batchsize.py    memory probe
  tuner.py        Optuna studies
  history.py      History-of-AI stages
  runs.py, jobs.py   storage, background jobs + event streams
  server.py       FastAPI app
frontend/src/
  pages/          Builder, Train, Leaderboard, CustomBlocks, History
  components/     Brick, ParamField, LineChart, LiveCharts, ArchView
  store.tsx       app state, undo/redo, live job stream
```
