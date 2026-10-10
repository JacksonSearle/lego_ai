# Lego AI

Build neural networks out of colored blocks, train them with live charts, tune them with Optuna,
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

## Learning mode

On first launch you choose between two modes:

- **I'm new to AI** (guided). You start in the **Lab**, the simplest possible model: MNIST digits →
  Flatten → one hidden layer → Output Head. You pick the layer's width from five presets, from
  *too small* (2 neurons) to *too big* (4,096). Training starts on its own, and the page shows
  the accuracy, the model's guesses on real test digits, and an MNIST leaderboard. You earn XP for
  everything you try, and features unlock level by level:

  | Level | Unlocks |
  | --- | --- |
  | 1 | Lab |
  | 2 | stacking hidden layers (repeat) |
  | 3 | the Builder: block panel, drag & drop, repeat, Dropout |
  | 4 | tensor shapes lesson + the "repair the broken model" challenge |
  | 5 | Train page (epochs, learning rate), Leaderboard, activation choice |
  | 6 | Skill Tree, more datasets, History of AI |
  | 7 | optimizer, batch size + memory checker |
  | 8 | JSON import/export, resume, templates |
  | 9 | schedules, regularization, stability |
  | 10 | Optuna |
  | 11 | Custom Python blocks |
  | 12 | seed / device / max steps |

  From level 6, each level gives skill points to spend in the **Skill Tree**. It has an Image
  branch (Conv → Pooling → ResNet → Depthwise → ViT → MLP-Mixer), a Text branch (Embeddings →
  RNN → LSTM/GRU → Attention → Transformer) and a General branch, each in roughly historical
  order. Each level comes with quests (the panel in the bottom right, with *Show me* buttons that
  highlight what to click). There are 31 achievements, and avatars unlock as you level up.
- **Unlock everything** (expert). Every block, page and setting is available, all achievements
  are granted, there's no XP bar, and a one-minute spotlight tour introduces each page.

You can switch modes or reset progress from the avatar menu. Progress is stored in
`backend/data/progress.json`.

The **Guidebook** tab is always available. It holds 48 lessons: concepts (neural nets, MNIST,
training, layers, tensor shapes, overfitting), every page and action (loading a model from JSON,
resuming, reproducing, the memory checker, Optuna, a line-by-line custom-block walkthrough),
training knobs (epochs, learning rate, batch size, optimizers, schedules, regularization, seeds)
and every block. Block lessons include an animated diagram of what the block does, a walkthrough of
each parameter, the matching PyTorch code annotated line by line, the original paper and a quiz.
Every page has a **?** button linking to its lesson. The lesson content lives in
`frontend/src/learn/` (`curriculum.ts` for levels, quests, achievements and skills; `lessons.tsx`;
`diagrams/`).

## Tabs

| Tab | What it does |
| --- | --- |
| **Build** | Drag blocks from the palette onto a stack that flows **up** from the input (bottom, with its shape) to the output (top, with the dataset's target shape). Shapes are inferred after every edit and shown on the arrows. Click a block for its parameters, with help text and defaults. Use **Repeat** to stack a block N times (shown as ×N beside it). Templates give a proven starting point per dataset, or you can start from scratch. Undo/redo, import/export. |
| **Train** | Hyperparameters (optimizer, LR, scheduler, warm-up, weight decay, dropout override, label smoothing, grad clipping, loss, augmentation, seed, device). **Check what fits in memory** tests a range of batch sizes. Pressing Train checks yours first and suggests alternatives if it won't fit. Turn on **Optuna** to give ranges instead of values. Live loss/accuracy/LR charts. Resume any checkpoint, or load a previous run's config to retrain from scratch. |
| **Leaderboard** | All runs ranked by validation accuracy (or test acc, loss, params, time), filterable by dataset. Click a run for its curves, architecture, hyperparameters and environment, and to open it in the builder, **retrain from scratch** (marked ✓ reproduced when it matches exactly), resume, download the config, or download the weights. |
| **Custom Blocks** | Write blocks in Python (CodeMirror editor). Save, and they show up in the palette like built-ins. |
| **History of AI** | Plays by itself: trains 12 landmark architectures from 1805 to 2021 on the same MNIST budget, with a timeline, live charts, a separate demo leaderboard, and a paragraph and paper link for each. |

## Blocks

Each color family is one kind of layer:

- **Dense** (blues): Linear, MLP Block (optionally residual)
- **Convolution** (oranges): Conv2D (+BN), Depthwise-Separable Conv, Residual Conv Block
- **Pooling** (ambers/cyan): Max Pool, Average Pool, Global Average Pool, Sequence Pool
- **Recurrent** (greens): RNN (Elman), LSTM, GRU
- **Sequence** (pinks): Token Embedding, Image → Sequence, Patch Embedding, Positional Encoding
- **Attention** (purples): Transformer Block (causal mask is automatic for language modeling), Self-Attention, MLP-Mixer Block
- **Normalization** (teals): Batch Norm, Layer Norm
- **Utility** (grays/yellow): Flatten, Dropout, Activation
- **Head** (red): Output Head, which sizes itself to the dataset (classes, or vocab per step)

Blocks never ask for their input size, because it comes from the block below. Any block with an
activation can use relu, gelu, silu, tanh, sigmoid, leaky_relu, elu, mish, softplus or identity.

## Datasets

MNIST, Fashion-MNIST, CIFAR-10, AG News (word-level topic classification), Tiny Shakespeare
(character-level language modeling), and 2-D Spirals (synthetic, instant). For each, the
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
