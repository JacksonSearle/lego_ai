import type { ReactElement, ReactNode } from 'react'
import type { Feature } from './curriculum'
import { BLOCK_DIAGRAMS } from './diagrams/blocks'
import {
  BatchNoise, Depth, Epochs, LearningRate, MatMul, MnistPeek, NeuralNet, OptunaSearch, Optimizers, Overfitting, Regularization,
  Schedules, Shapes, TrainingLoop,
} from './diagrams/concepts'

export type Section = 'start' | 'app' | 'train' | 'general' | 'image' | 'text'

export const SECTIONS: { id: Section; title: string; icon: string }[] = [
  { id: 'start', title: 'Start here', icon: '🚀' },
  { id: 'app', title: 'Using the app', icon: '🧭' },
  { id: 'train', title: 'Training knobs', icon: '🎛️' },
  { id: 'general', title: 'Blocks · General', icon: '🧱' },
  { id: 'image', title: 'Blocks · Image', icon: '🖼️' },
  { id: 'text', title: 'Blocks · Text', icon: '📝' },
]

export interface CodeLine { code: string; note?: string }

export interface Lesson {
  id: string
  title: string
  section: Section
  summary: string
  tags?: ('image' | 'text' | 'general')[]
  year?: number
  diagram?: () => ReactElement
  diagrams?: (() => ReactElement)[]
  body: ReactNode
  block?: string // block type: parameters are listed from its spec
  paramNotes?: Record<string, string>
  code?: CodeLine[]
  paper?: { title: string; url: string; authors: string; year: number }
  quiz?: { q: string; options: string[]; answer: number; explain: string }
  /** When the lesson becomes readable in guided mode. Block lessons follow their block's unlock. */
  unlock?: { level?: number; feature?: Feature; skill?: string }
  action?: 'challenge' | 'addBlock' | { tab: string; label: string }
}

const P = ({ children }: { children: ReactNode }) => <p>{children}</p>
const Tip = ({ children }: { children: ReactNode }) => <div className="lesson-tip">💡 {children}</div>

export const LESSONS: Lesson[] = [
  // =========================================================================
  // Start here
  // =========================================================================
  {
    id: 'start.welcome', section: 'start', title: 'Welcome to Lego AI', summary: 'What you’ll learn and how this app works.', unlock: { level: 1 },
    body: <>
      <P>Lego AI teaches you how modern AI models are built — by building them. A neural network is a stack of <b>blocks</b>, each doing one simple job. You snap blocks together, press train, and watch the model learn.</P>
      <P>You’ll start with the simplest possible model and a famous dataset of handwritten digits. As you try things you earn <b>experience (XP)</b>. Each new level unlocks something new — more blocks, more controls, new datasets — along with a lesson here in the <b>Guidebook</b> explaining it.</P>
      <P>The quest panel (bottom left) always shows what to try next. If you’re ever lost, press <b>Show me</b> on a quest.</P>
      <Tip>Every lesson you read earns XP, and the guidebook keeps everything you’ve unlocked so you can come back any time.</Tip>
    </>,
  },
  {
    id: 'start.nn', section: 'start', title: 'What is a neural network?', summary: 'Neurons, weights, and learning from examples.', unlock: { level: 1 }, diagram: NeuralNet,
    body: <>
      <P>Traditional programs follow rules a person wrote. A neural network instead <b>learns</b> its rules from examples. Show it thousands of pictures of handwritten digits along with the right answers, and it gradually adjusts itself until it gets them right.</P>
      <P>Inside, it’s just numbers. Each <b>neuron</b> takes some numbers in, multiplies each by a <b>weight</b>, adds them up, and passes the result on. A network has thousands to billions of weights. <b>Training</b> means finding weight values that make the final answer correct.</P>
      <P>Neurons are organised in <b>layers</b>. The input layer holds the data (e.g. pixel brightness). The output layer holds the answer (e.g. one score per digit). Layers in between are “hidden” — they find useful patterns on their own.</P>
    </>,
    quiz: { q: 'What does training change inside a neural network?', options: ['The input data', 'The weights', 'The number of layers', 'The answers in the dataset'], answer: 1, explain: 'Training adjusts the weights. The architecture (layers) is what you design; the data stays fixed.' },
  },
  {
    id: 'start.mnist', section: 'start', title: 'Meet MNIST', summary: '70,000 handwritten digits — the “hello world” of AI.', unlock: { level: 1 }, diagram: MnistPeek, tags: ['image'],
    body: <>
      <P><b>MNIST</b> is a collection of 70,000 small pictures of handwritten digits (0–9), collected from US Census workers and high-school students in the 1990s. Each picture is <b>28 × 28 pixels</b> in greyscale, and each comes with a label saying which digit it is.</P>
      <P>The task: look at a picture and say which digit it is. That’s a <b>classification</b> problem with <b>10 classes</b>. Humans get about 99.8% right. A good model can get over 99%.</P>
      <P>The data is split in three:</P>
      <ul>
        <li><b>Training set</b> (55,000) — the examples the model learns from.</li>
        <li><b>Validation set</b> (5,000) — held back to check progress on pictures it has never trained on. This is the “accuracy” you’ll see most.</li>
        <li><b>Test set</b> (10,000) — scored once at the very end, as a final exam.</li>
      </ul>
      <Tip>Why hold data back? A model could simply memorise the training pictures. Only accuracy on <i>unseen</i> pictures tells you whether it really learned to read digits.</Tip>
    </>,
    paper: { title: 'Gradient-Based Learning Applied to Document Recognition', url: 'http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf', authors: 'Y. LeCun, L. Bottou, Y. Bengio, P. Haffner', year: 1998 },
    quiz: { q: 'An MNIST image is 28×28 pixels. How many numbers is that?', options: ['28', '56', '784', '10'], answer: 2, explain: '28 rows × 28 columns = 784 brightness values.' },
  },
  {
    id: 'start.training', section: 'start', title: 'How training works', summary: 'Guess, check, blame, adjust — repeated thousands of times.', unlock: { level: 1 }, diagram: TrainingLoop,
    body: <>
      <P>Training is a loop. The model takes a small <b>batch</b> of examples (say 128 digits) and:</P>
      <ol>
        <li><b>Guesses</b> — runs the pictures through its layers (the <i>forward pass</i>).</li>
        <li><b>Checks</b> — compares guesses with the right answers. The <b>loss</b> is one number measuring how wrong it was. Lower is better.</li>
        <li><b>Blames</b> — works out how much each weight contributed to the error (<i>backpropagation</i>, which computes the <i>gradient</i>).</li>
        <li><b>Adjusts</b> — nudges every weight a little in the direction that reduces the loss (the <i>optimizer step</i>).</li>
      </ol>
      <P>Do this thousands of times and the loss falls while accuracy rises. That’s the curve you’ll watch when you press Train.</P>
    </>,
    quiz: { q: 'What does the loss measure?', options: ['How big the model is', 'How wrong the model’s guesses are', 'How long training takes', 'How many examples there are'], answer: 1, explain: 'The loss is a single number for how wrong the guesses are. Training tries to make it smaller.' },
  },
  {
    id: 'start.layers', section: 'start', title: 'Layers & depth', summary: 'Why stacking layers helps, and what “deep” means.', unlock: { level: 2 }, diagram: Depth,
    body: <>
      <P>A <b>hidden layer</b> transforms its input into a new set of numbers — new “features”. With one hidden layer, the model can only build fairly simple features directly from pixels.</P>
      <P>Stack several and each layer builds on the last: the first may respond to small strokes, the next to loops and corners, the next to whole digit shapes. A network with many layers is called <b>deep</b> — that’s where “deep learning” gets its name.</P>
      <P>Depth isn’t free: deeper networks are slower and can be harder to train. Later you’ll meet tricks (activations, normalisation, residual connections) that make very deep networks work.</P>
      <Tip>In the Lab, “number of hidden layers” simply repeats the same block several times. In the Builder you can repeat any block with its <b>Repeat</b> control.</Tip>
    </>,
  },
  {
    id: 'start.shapes', section: 'start', title: 'Tensors & shapes', summary: 'Why blocks have to fit together — and how to fix it when they don’t.', unlock: { feature: 'lesson.shapes' }, diagrams: [Shapes, MatMul],
    body: <>
      <P>Everything flowing through a network is a <b>tensor</b>: a grid of numbers with a <b>shape</b>. A single number has shape (); a list of 784 numbers has shape <code>784</code>; an MNIST image has shape <code>1×28×28</code> (1 colour channel, 28 rows, 28 columns). A batch of 128 images adds one more dimension in front: <code>128×1×28×28</code>.</P>
      <P>Each block expects a certain shape and produces a new one. A Linear layer wants a flat list; a convolution wants an image; a transformer wants a sequence. <b>The output shape of each block must match what the next block expects</b> — just like Lego studs only fit the right holes.</P>
      <P>Why so strict? Inside a Linear layer, the input is multiplied by a weight matrix. Matrix multiplication only works when the inner sizes agree — there must be exactly one weight for every input number. A mismatch isn’t a style problem; the maths is impossible.</P>
      <P>In the Builder the shape after every block is shown on the arrows between them, and a block that can’t accept its input turns red and tells you why. The usual fixes:</P>
      <ul>
        <li><b>Image going into a Linear layer?</b> Add a <b>Flatten</b> block before it.</li>
        <li><b>Wrong output size?</b> The top of the model must output one number per class (10 for MNIST). End with an <b>Output Head</b>, which sizes itself automatically.</li>
        <li><b>Text tokens going into anything but an embedding?</b> Start with a <b>Token Embedding</b>.</li>
      </ul>
      <Tip>Ready? Press <b>Start the challenge</b> below. You’ll get a broken model to repair.</Tip>
    </>,
    quiz: { q: 'A Linear layer with 784 inputs receives a tensor of shape 1×28×28. What should you add before it?', options: ['Dropout', 'Another Linear layer', 'Flatten', 'An Output Head'], answer: 2, explain: 'Flatten turns 1×28×28 into 784 — same numbers, flat shape — so the Linear layer can use them.' },
    action: 'challenge',
  },
  {
    id: 'start.overfitting', section: 'start', title: 'Overfitting & validation', summary: 'When a model memorises instead of learning.', unlock: { level: 5 }, diagram: Overfitting,
    body: <>
      <P>A big enough model can memorise its training examples perfectly — including their quirks — without learning the general rule. It then scores brilliantly on training data and badly on new data. That’s <b>overfitting</b>.</P>
      <P>That’s why we watch <b>validation</b> accuracy and loss, measured on held-out examples. When training loss keeps improving but validation loss starts getting worse, the model has begun to memorise. The leaderboard ranks models by their <b>best</b> validation accuracy for this reason.</P>
      <P>Remedies: more data, fewer epochs, a smaller model, or <b>regularisation</b> such as Dropout and weight decay (you’ll unlock these).</P>
    </>,
    quiz: { q: 'Training loss keeps falling but validation loss starts rising. What’s happening?', options: ['Underfitting', 'Overfitting', 'The learning rate is too small', 'Nothing — that’s ideal'], answer: 1, explain: 'The model is memorising the training set: classic overfitting.' },
  },

  // =========================================================================
  // Using the app
  // =========================================================================
  {
    id: 'app.lab', section: 'app', title: 'The Lab', summary: 'Your first model in one click.', unlock: { level: 1 }, action: { tab: 'lab', label: 'Open the Lab' },
    body: <>
      <P>The Lab is the simplest possible model: the image is flattened into 784 numbers, passed through <b>one hidden Linear layer</b>, and then an <b>Output Head</b> gives a score for each digit.</P>
      <P>You choose one thing: how wide the hidden layer is, from <i>too small</i> (2 neurons) to <i>too big</i> (4,096). Picking a size starts training immediately. Within seconds you’ll see the accuracy, the learning curve, some test digits with the model’s guesses, and where your model ranks on the MNIST leaderboard.</P>
      <P>At level 2 you can also choose how many hidden layers to stack.</P>
    </>,
  },
  {
    id: 'app.builder', section: 'app', title: 'The Builder', summary: 'Drag, drop, stack and configure blocks.', unlock: { feature: 'builder' }, action: { tab: 'build', label: 'Open the Builder' },
    body: <>
      <P>The Builder is where you design models. Data flows <b>upward</b>: the input (with its shape) is at the bottom, the output the dataset needs is at the top.</P>
      <ul>
        <li><b>Add blocks</b> — drag them from the panel on the left into the stack (a yellow line shows where they’ll land), or click one to drop it in.</li>
        <li><b>Rearrange</b> — drag a block within the stack. Drag it to the red bin at the bottom to delete it.</li>
        <li><b>Configure</b> — click a block. The side panel explains every setting and shows its default.</li>
        <li><b>Repeat</b> — in the side panel, stack a block several times. The count appears beside it (×3).</li>
        <li><b>Check shapes</b> — the shape after each block is written on the arrows. Red blocks explain what’s wrong.</li>
      </ul>
      <P>Undo with ⌘Z, delete with ⌫, duplicate with ⌘D. Until the Train page unlocks, a “Train & test” panel on the right trains your design with one click.</P>
    </>,
  },
  {
    id: 'app.train', section: 'app', title: 'The Train page', summary: 'Hyperparameters, live charts, resuming.', unlock: { feature: 'train.page' }, action: { tab: 'train', label: 'Open Train' },
    body: <>
      <P>The Train page controls <i>how</i> your model learns. The left column holds settings (called <b>hyperparameters</b> because they aren’t learned — you choose them). More appear as you level up: epochs and learning rate first, then optimizer and batch size, schedules, regularisation and finally Optuna.</P>
      <P>Press <b>Train</b> and the right column shows live charts. Blue is the training set, orange the validation set. Loss should go down and accuracy up. When training finishes, the model is scored on the test set and added to the leaderboard.</P>
    </>,
  },
  {
    id: 'app.leaderboard', section: 'app', title: 'The Leaderboard', summary: 'Compare every model you’ve trained.', unlock: { feature: 'leaderboard.page' }, action: { tab: 'leaderboard', label: 'Open Leaderboard' },
    body: <>
      <P>Every run is saved and ranked by its best validation accuracy (you can sort by test accuracy, loss, size or time). Filter by dataset at the top. The coloured chips show each model’s blocks at a glance.</P>
      <P>Click a run to see its curves, architecture, settings and the computer it ran on. From there you can open it in the Builder, retrain it, resume it, or download its config and weights.</P>
    </>,
  },
  {
    id: 'app.guidebook', section: 'app', title: 'The Guidebook, skill tree & achievements', summary: 'Where everything you unlock is explained.', unlock: { level: 1 }, action: { tab: 'guide', label: 'Open Guidebook' },
    body: <>
      <P>The Guidebook holds a lesson for every concept, page and block. Locked lessons show what unlocks them. The <b>Skill Tree</b> tab is where you spend skill points (two per level from level 6) to unlock new blocks. Image blocks and text blocks form two branches, roughly in the order they were invented. The <b>Achievements</b> tab lists badges and what earns them.</P>
      <P>Each page has a <b>?</b> button that opens its guide.</P>
    </>,
  },
  {
    id: 'app.skilltree', section: 'app', title: 'Choosing a path', summary: 'Image or text? Old or new?', unlock: { feature: 'skilltree' }, action: { tab: 'guide', label: 'Open the Skill Tree' },
    body: <>
      <P>From level 6 you earn <b>2 skill points per level</b>. Spend them in the Skill Tree to unlock new blocks. Each branch follows history, from older ideas to newer ones, so you can feel why each new invention mattered:</P>
      <ul>
        <li><b>🖼️ Image</b>: Convolution (1989) → Pooling → Residual blocks (2015) → Depthwise conv (2017) → Vision Transformer (2020) → MLP-Mixer (2021). Convolution also unlocks the CIFAR-10 colour photo dataset.</li>
        <li><b>📝 Text</b>: Embeddings → RNN (1990) → LSTM & GRU (1997/2014) → Attention (2014) → Transformer (2017). Embeddings unlock the AG News and Tiny Shakespeare datasets.</li>
        <li><b>🧱 General</b>: Dropout, activations, MLP blocks, normalisation — useful everywhere.</li>
      </ul>
      <P>Unlocking a skill also unlocks its lesson, with an animation, its PyTorch code and the original paper.</P>
    </>,
  },
  {
    id: 'app.history', section: 'app', title: 'History of AI', summary: 'Watch 200 years of models race on MNIST.', unlock: { feature: 'history.page' }, action: { tab: 'history', label: 'Open History' },
    body: <>
      <P>The History page trains twelve landmark models — from 1805’s least squares to 2021’s MLP-Mixer — on the same digits with the same budget, so you can see what each idea added. It runs by itself; click any year on the timeline to see that model’s blocks, its learning curve and the paper that introduced it. It has its own leaderboard, separate from yours.</P>
    </>,
  },
  {
    id: 'app.json', section: 'app', title: 'Saving & loading models (JSON)', summary: 'Export a model, share it, load it back.', unlock: { feature: 'config.io' },
    body: <>
      <P>Every model can be saved as a small text file in <b>JSON</b> format (<code>.lego.json</code>). It records the dataset, every block and its settings, every training setting and the random seed — and, for runs, the software versions used. Anyone with the file can rebuild exactly the same model.</P>
      <ul>
        <li><b>Export</b>: in the Builder toolbar press <b>Export</b>. On the Leaderboard, open a run and press <b>Download config</b>.</li>
        <li><b>Load</b>: in the Builder press <b>Import</b> and choose a <code>.lego.json</code> file. It replaces your current model (undo with ⌘Z). If it uses custom blocks you don’t have, their code is installed from the file.</li>
        <li>Or on the Train page choose <b>Start from a previous run → Load config</b>.</li>
      </ul>
      <P>Here’s what a config looks like:</P>
      <pre className="json">{`{
  "format": "lego-ai.run",
  "name": "Simple CNN",
  "dataset": { "id": "mnist" },
  "architecture": { "blocks": [
    { "type": "conv2d", "params": { "out_channels": 32 }, "repeat": 1 },
    { "type": "maxpool2d", "params": {}, "repeat": 1 },
    { "type": "flatten", "params": {}, "repeat": 1 },
    { "type": "classifier", "params": {}, "repeat": 1 }
  ]},
  "training": { "epochs": 5, "lr": 0.002, "seed": 42, ... }
}`}</pre>
    </>,
  },
  {
    id: 'app.resume', section: 'app', title: 'Resuming training', summary: 'Stop now, continue later — from exactly where you were.', unlock: { feature: 'train.resume' },
    body: <>
      <P>At the end of every epoch, a <b>checkpoint</b> is saved: the weights, the optimizer’s memory, the learning-rate schedule and every random-number generator’s state. That means training can continue later exactly as if it had never stopped.</P>
      <ul>
        <li>On the <b>Train</b> page: <b>Start from a previous run</b> → pick a run → <b>Resume checkpoint to N epochs</b>.</li>
        <li>On the <b>Leaderboard</b>: open a run → <b>Resume from checkpoint</b>.</li>
      </ul>
      <P>Use it to give a promising model more epochs, or to continue a run you stopped. If you stop mid-epoch, it resumes from the last completed epoch.</P>
    </>,
  },
  {
    id: 'app.reproduce', section: 'app', title: 'Reproducing a run', summary: 'Same config, same seed, same result — bit for bit.', unlock: { feature: 'config.io' },
    body: <>
      <P>Science needs results others can repeat. Open a run on the Leaderboard and press <b>Retrain from scratch</b>: the identical config is trained again from new random weights — the same random weights, because the seed is the same. When it finishes, it’s compared with the original. If every number matches you get a <b>✓ reproduced</b> badge.</P>
      <P>This works exactly on the CPU with the same PyTorch version. GPUs use some operations that aren’t perfectly repeatable, so tiny differences are possible there.</P>
    </>,
  },
  {
    id: 'app.batch', section: 'app', title: 'Will it fit? The memory checker', summary: 'Find a batch size your computer can handle.', unlock: { feature: 'train.batch' },
    body: <>
      <P>Bigger batches need more memory: every example’s intermediate results are kept until the backward pass. Too big and your computer runs out.</P>
      <P>On the Train page press <b>Check what fits in memory</b>. Lego AI estimates the memory for batch sizes from 8 up to 8192, actually tries the ones that look safe, and shows the memory used, time per step and time per epoch. Click a row to use it. When you press Train, your batch size is checked first; if it won’t fit you’ll be shown alternatives.</P>
    </>,
  },
  {
    id: 'app.optuna', section: 'app', title: 'Hyperparameter search with Optuna', summary: 'Let the computer find good settings.', unlock: { feature: 'train.optuna' }, diagram: OptunaSearch,
    body: <>
      <P>Finding a good learning rate, dropout and weight decay by hand is slow. <b>Optuna</b> automates it: you give each setting a <i>range</i> instead of a value, and it runs many short trainings (“trials”), learning which regions work best.</P>
      <ol>
        <li>On the Train page switch on <b>Hyperparameter search (Optuna)</b>.</li>
        <li>Tick the settings to search and set their ranges. Use <b>log</b> for things that vary over orders of magnitude, like learning rate (0.0001 to 0.01).</li>
        <li>Choose the number of trials and epochs per trial, then press <b>Run Optuna</b>.</li>
        <li>Watch the trials table fill in. The best is starred — press <b>Use best params</b> to copy them into the form.</li>
      </ol>
      <P>Each trial is saved as a normal run, so the best one can be reproduced or resumed.</P>
    </>,
    paper: { title: 'Optuna: A Next-generation Hyperparameter Optimization Framework', url: 'https://arxiv.org/abs/1907.10902', authors: 'T. Akiba et al.', year: 2019 },
  },
  {
    id: 'app.custom', section: 'app', title: 'Your first custom block', summary: 'A hand-held walkthrough of writing a block in Python.', unlock: { feature: 'custom.page' }, action: { tab: 'custom', label: 'Open Custom Blocks' },
    body: <>
      <P>Every block in the palette is a small piece of Python using <b>PyTorch</b>. You can write your own, and it will appear in the palette like any other block. Let’s make one step by step.</P>
      <ol>
        <li>Go to <b>Custom Blocks</b> and press <b>+ New</b>. A starter file appears.</li>
        <li>Read the code with the explanation below — it’s a Linear layer plus an activation, a block you already know.</li>
        <li>Change something: the <code>name</code>, the <code>color</code>, or the default <code>width</code>.</li>
        <li>Press <b>Check</b> to test it, then <b>Save & load</b>.</li>
        <li>Go to the Builder. Your block is in the <b>Custom</b> group of the palette. Drag it into a model and train.</li>
      </ol>
      <Tip>If something is wrong, Check shows the Python error. The most common one is a shape mismatch: use <code>in_shape[-1]</code> for the input size.</Tip>
    </>,
    code: [
      { code: 'import torch.nn as nn', note: 'PyTorch’s neural-network toolbox: layers like nn.Linear live here.' },
      { code: 'from lego.sdk import Param, lego_block, make_activation', note: 'Lego AI helpers: the decorator that registers a block, and Param to describe its settings.' },
      { code: '' },
      { code: '@lego_block(', note: 'This decorator turns the class below into a palette block.' },
      { code: '    name="My Block",', note: 'The name shown on the brick.' },
      { code: '    color="#22d3ee",', note: 'The brick’s colour (any hex colour).' },
      { code: '    params=[Param("width", "int", 128, "Width", "Number of output features.", min=1)],', note: 'Each Param becomes a field in the side panel: name, type, default, label, help text.' },
      { code: ')' },
      { code: 'class MyBlock(nn.Module):', note: 'Every PyTorch layer is a class that extends nn.Module.' },
      { code: '    def __init__(self, in_shape, width=128):', note: 'Called once to build the block. in_shape is the shape of the input (without the batch), e.g. (784,). width comes from the side panel.' },
      { code: '        super().__init__()', note: 'Required boilerplate for nn.Module.' },
      { code: '        self.fc = nn.Linear(in_shape[-1], width)', note: 'A Linear layer from the input size (last dimension of in_shape) to width outputs. Its weights are learned.' },
      { code: '        self.act = make_activation("gelu")', note: 'An activation function (see the Activation lesson).' },
      { code: '' },
      { code: '    def forward(self, x):', note: 'Called for every batch: x is the input tensor, e.g. shape (128, 784).' },
      { code: '        return self.act(self.fc(x))', note: 'Linear, then activation. The output shape (128, width) is detected automatically.' },
    ],
  },

  // =========================================================================
  // Training knobs
  // =========================================================================
  {
    id: 'train.epochs', section: 'train', title: 'Epochs', summary: 'How many times the model sees the data.', unlock: { feature: 'train.epochs' }, diagram: Epochs,
    body: <>
      <P>One <b>epoch</b> is one full pass over the training set. With 55,000 digits and batches of 128, that’s about 430 steps. In the Lab every model trained for exactly 1 epoch.</P>
      <P>More epochs give the model more practice, so accuracy usually climbs — quickly at first, then slowly. Eventually more epochs can hurt: the model starts memorising (overfitting). Watch the orange validation curve: when it stops improving, more epochs won’t help.</P>
      <Tip>Try 5 epochs on a model you already trained for 1, and compare on the leaderboard.</Tip>
    </>,
    quiz: { q: 'With 1,000 training examples and a batch size of 100, how many steps is one epoch?', options: ['1', '10', '100', '1,000'], answer: 1, explain: '1,000 examples ÷ 100 per batch = 10 steps per epoch.' },
  },
  {
    id: 'train.lr', section: 'train', title: 'Learning rate', summary: 'How big a step to take each update.', unlock: { feature: 'train.lr' }, diagram: LearningRate,
    body: <>
      <P>After each batch, every weight moves a little in the direction that lowers the loss. The <b>learning rate</b> sets how big that step is. It is usually the single most important setting.</P>
      <ul>
        <li><b>Too small</b> (e.g. 0.00001): every step is safe, but learning crawls. After an epoch, accuracy may barely move.</li>
        <li><b>Just right</b> (often 0.001 with Adam/AdamW): fast, steady progress.</li>
        <li><b>Too large</b> (e.g. 0.1 or more with Adam): steps overshoot the valley; the loss jumps around or explodes to “NaN” (not a number).</li>
      </ul>
      <P>Try the extremes yourself — it’s the best way to build intuition.</P>
    </>,
    quiz: { q: 'Your loss jumps around wildly and then becomes NaN. What should you try first?', options: ['Raise the learning rate', 'Lower the learning rate', 'Add more epochs', 'Use a bigger batch'], answer: 1, explain: 'An exploding loss is the classic sign of a learning rate that’s too high.' },
  },
  {
    id: 'train.batch', section: 'train', title: 'Batch size', summary: 'How many examples per step.', unlock: { feature: 'train.batch' }, diagram: BatchNoise,
    body: <>
      <P>The model doesn’t learn from one example at a time, nor from all 55,000 at once — it uses a <b>batch</b>. The gradient from a batch is an estimate of the “true” downhill direction.</P>
      <ul>
        <li><b>Small batches</b> (16–64): noisy estimates, many cheap steps. The noise can even help find better solutions.</li>
        <li><b>Large batches</b> (512+): smooth estimates, fewer but more expensive steps, and much more memory. Often needs a larger learning rate.</li>
      </ul>
      <P>128 is a good default. Use the memory checker to see what your computer can handle.</P>
    </>,
  },
  {
    id: 'train.optimizers', section: 'train', title: 'Optimizers', summary: 'SGD, momentum, Adam, AdamW.', unlock: { feature: 'train.optimizer' }, diagram: Optimizers,
    body: <>
      <P>The <b>optimizer</b> decides exactly how to use the gradient to update the weights.</P>
      <ul>
        <li><b>SGD</b> (stochastic gradient descent): step straight downhill. Simple, but zig-zags in narrow valleys.</li>
        <li><b>SGD + momentum</b>: keeps a running velocity, like a heavy ball rolling. Smooths out zig-zags. (Set with the Momentum setting.)</li>
        <li><b>Adam</b> (2014): adapts the step size for every weight separately based on its history. Works well out of the box.</li>
        <li><b>AdamW</b> (2017): Adam with weight decay done properly. The default here.</li>
        <li><b>RMSprop</b>, <b>Adagrad</b>: earlier adaptive methods; mostly of historical interest.</li>
      </ul>
      <Tip>SGD usually needs a larger learning rate (0.01–0.1) than Adam (0.0003–0.003).</Tip>
    </>,
    paper: { title: 'Adam: A Method for Stochastic Optimization', url: 'https://arxiv.org/abs/1412.6980', authors: 'D. Kingma, J. Ba', year: 2014 },
  },
  {
    id: 'train.schedulers', section: 'train', title: 'Learning-rate schedules', summary: 'Change the step size as training goes on.', unlock: { feature: 'train.schedule' }, diagram: Schedules,
    body: <>
      <P>A good learning rate early in training is often too large later on. A <b>schedule</b> changes it over time:</P>
      <ul>
        <li><b>cosine</b> (default): glides smoothly from the full rate down to zero.</li>
        <li><b>onecycle</b>: ramps up for the first 30%, then down. Often trains fast.</li>
        <li><b>step</b>: multiplies by γ every N epochs.</li>
        <li><b>exponential</b>: multiplies by γ every epoch.</li>
        <li><b>plateau</b>: cuts the rate when validation loss stops improving.</li>
      </ul>
      <P><b>Warm-up</b> starts from a tiny rate and ramps up over the first epoch(s). Transformers in particular need it.</P>
    </>,
    paper: { title: 'SGDR: Stochastic Gradient Descent with Warm Restarts', url: 'https://arxiv.org/abs/1608.03983', authors: 'I. Loshchilov, F. Hutter', year: 2016 },
  },
  {
    id: 'train.regularization', section: 'train', title: 'Regularisation', summary: 'Weight decay, dropout, label smoothing, augmentation.', unlock: { feature: 'train.regularization' }, diagram: Regularization,
    body: <>
      <P>Regularisation makes memorising harder, so the model must learn general patterns instead.</P>
      <ul>
        <li><b>Weight decay</b>: pulls weights toward zero every step. Typical: 0.0001–0.05.</li>
        <li><b>Dropout override</b>: sets the drop probability of every Dropout in the model at once (handy for Optuna).</li>
        <li><b>Label smoothing</b>: softens targets so the model isn’t pushed to be 100% sure. Typical: 0.05–0.1.</li>
        <li><b>Data augmentation</b>: for colour images (CIFAR-10), randomly shifts and mirrors each picture, so the model sees endless variations.</li>
      </ul>
      <P>Gradient clipping (under Stability) caps how big a single update can get — essential for RNNs.</P>
    </>,
    paper: { title: 'Decoupled Weight Decay Regularization (AdamW)', url: 'https://arxiv.org/abs/1711.05101', authors: 'I. Loshchilov, F. Hutter', year: 2017 },
  },
  {
    id: 'train.repro', section: 'train', title: 'Reproducibility: seeds & devices', summary: 'Why the same seed gives the same result.', unlock: { feature: 'train.repro' },
    body: <>
      <P>Training uses randomness: the starting weights, the order of examples, which neurons Dropout switches off, how images are augmented. Computers generate “random” numbers from a starting value called the <b>seed</b>. Same seed, same sequence of random numbers, same result.</P>
      <P>Change the seed to see how much your result depends on luck. If two models differ by less than that, the difference may be noise.</P>
      <P><b>Device</b> chooses the processor. CPU is exactly reproducible. Apple’s GPU (MPS) or NVIDIA’s (CUDA) are much faster for big models like CNNs on CIFAR-10, but a few operations aren’t perfectly repeatable. <b>Max steps</b> stops training after a fixed number of updates — handy for quick experiments.</P>
    </>,
  },
]

// =========================================================================
// Block lessons
// =========================================================================
interface BlockLessonDef extends Omit<Lesson, 'id' | 'section'> { block: string; section: Section }

const BLOCK_LESSONS: BlockLessonDef[] = [
  {
    block: 'linear', section: 'general', title: 'Linear (fully connected) layer', summary: 'Every output looks at every input.', tags: ['general'], year: 1958,
    body: <>
      <P>The Linear layer is the original neural-network building block. Each of its outputs (neurons) computes a weighted sum of <i>all</i> its inputs, plus a bias: <code>y = W·x + b</code>. With 784 inputs and 128 outputs, that’s 784 × 128 + 128 = <b>100,480</b> learnable numbers.</P>
      <P>By itself it can only draw straight lines (it’s “linear”), so it’s usually followed by an activation function such as ReLU. Rosenblatt’s 1958 perceptron was a single such unit.</P>
    </>,
    paramNotes: { out_features: 'How many neurons — the width of the layer. More neurons = more capacity (and more weights).', activation: 'The bend applied after the weighted sum. “none” keeps it purely linear.', bias: 'Adds a learnable constant to each output. Almost always on.' },
    code: [
      { code: 'layer = nn.Linear(in_features=784, out_features=128)', note: 'Creates a 784×128 weight matrix W and 128 biases b, initialised randomly.' },
      { code: 'act = nn.ReLU()', note: 'The activation (if any).' },
      { code: 'y = act(layer(x))', note: 'x has shape (batch, 784). layer(x) computes x·Wᵀ + b → (batch, 128).' },
    ],
    paper: { title: 'The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain', url: 'https://psycnet.apa.org/record/1959-09865-001', authors: 'F. Rosenblatt', year: 1958 },
    quiz: { q: 'How many weights (not counting biases) does Linear(784 → 10) have?', options: ['794', '7,840', '784', '10'], answer: 1, explain: 'One weight for every input–output pair: 784 × 10 = 7,840.' },
  },
  {
    block: 'flatten', section: 'general', title: 'Flatten', summary: 'Unroll a grid into a list.', tags: ['general'],
    body: <>
      <P>Flatten reshapes a tensor into one long list, keeping every number. A 1×28×28 image becomes 784 numbers. It has no weights and learns nothing — it just changes the shape so a Linear layer can accept it.</P>
      <P>The catch: after flattening, the model no longer “knows” which pixels were neighbours. Convolutions (in the Image branch) keep the grid and use that structure.</P>
    </>,
    code: [
      { code: 'flat = nn.Flatten()', note: 'Keeps the batch dimension, flattens the rest.' },
      { code: 'y = flat(x)', note: 'x: (batch, 1, 28, 28) → y: (batch, 784).' },
    ],
  },
  {
    block: 'classifier', section: 'general', title: 'Output Head', summary: 'Turn features into one score per class.', tags: ['general'],
    body: <>
      <P>Every model here ends with an Output Head. It’s a Linear layer whose number of outputs is set automatically to what the dataset needs: <b>10</b> for digits, <b>4</b> for AG News topics, or one score per character at every position for language modelling.</P>
      <P>If its input is still an image or a sequence, it first reduces it to a single vector (by flattening or averaging). During training its scores go into the <b>cross-entropy loss</b>, which applies softmax and measures how much probability went to the right answer.</P>
    </>,
    paramNotes: { pool: 'How to squash an image or sequence into one vector first. “auto” flattens images and averages sequences.', dropout: 'Optional dropout just before the final layer.' },
    code: [
      { code: 'head = nn.Linear(features, num_classes)', note: 'num_classes comes from the dataset (10 for MNIST).' },
      { code: 'logits = head(x)', note: 'Raw scores, shape (batch, 10).' },
      { code: 'probs = logits.softmax(dim=-1)', note: 'Probabilities that add to 1 (used for showing predictions).' },
      { code: 'loss = F.cross_entropy(logits, labels)', note: 'Training compares the scores with the true labels.' },
    ],
  },
  {
    block: 'activation', section: 'general', title: 'Activation functions', summary: 'The bend that makes depth useful.', tags: ['general'], year: 1986,
    body: <>
      <P>Stack two Linear layers and the result is… still just one Linear layer (multiplying two matrices gives one matrix). To get something more powerful you need a <b>non-linearity</b> between layers: an activation function.</P>
      <P><b>Sigmoid</b> and <b>tanh</b> powered the networks of the 1980s and 90s, but they flatten out for large inputs, which slows learning. <b>ReLU</b> (max(0, x)) is simple and doesn’t saturate for positive values; it helped kick off the deep-learning boom in 2012. <b>GELU</b> and <b>SiLU</b> are smooth cousins used in transformers.</P>
      <P>Now that this is unlocked, every block with an activation lets you choose it, and there’s a standalone Activation block too.</P>
    </>,
    paramNotes: { fn: 'Which function to apply to every number.' },
    code: [
      { code: 'nn.ReLU()(x)      # max(0, x)', note: 'The default. Negatives become 0.' },
      { code: 'nn.GELU()(x)      # smooth ReLU', note: 'Used in GPT and BERT.' },
      { code: 'nn.Sigmoid()(x)   # 1 / (1 + e^-x)', note: 'Squashes to 0…1.' },
      { code: 'nn.Tanh()(x)      # squashes to -1…1', note: 'Used inside RNNs.' },
    ],
    paper: { title: 'Rectified Linear Units Improve Restricted Boltzmann Machines', url: 'https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf', authors: 'V. Nair, G. Hinton', year: 2010 },
    quiz: { q: 'Why do networks need activation functions between Linear layers?', options: ['To make training faster', 'Without them, stacked layers collapse into one linear layer', 'To reduce memory', 'To normalise the data'], answer: 1, explain: 'Linear ∘ Linear = Linear. The bend is what lets depth add power.' },
  },
  {
    block: 'dropout', section: 'general', title: 'Dropout', summary: 'Randomly switch neurons off to prevent memorising.', tags: ['general'], year: 2012,
    body: <>
      <P>Dropout is a regulariser: during training it sets a random fraction <i>p</i> of the values passing through it to zero (and scales the rest up to compensate). Each step drops a different random set. When you evaluate the model, nothing is dropped.</P>
      <P>Because any neuron might vanish, the network can’t rely on a few memorised features; it learns redundant, robust ones. Typical values: 0.1–0.5. Place it after hidden layers, not right before the input.</P>
    </>,
    paramNotes: { p: 'The fraction of values zeroed during training. 0.5 = half.' },
    code: [
      { code: 'drop = nn.Dropout(p=0.5)', note: 'p = probability of zeroing each value.' },
      { code: 'y = drop(x)', note: 'In training: about half of x becomes 0, the rest is multiplied by 2.' },
      { code: 'model.eval()  # dropout now does nothing', note: 'Evaluation mode switches dropout off.' },
    ],
    paper: { title: 'Dropout: A Simple Way to Prevent Neural Networks from Overfitting', url: 'https://jmlr.org/papers/v15/srivastava14a.html', authors: 'N. Srivastava, G. Hinton, et al.', year: 2014 },
  },
  {
    block: 'mlp', section: 'general', title: 'MLP Block', summary: 'Linear → activation → Linear, optionally residual.', tags: ['general'], year: 1986,
    body: <>
      <P>The multi-layer perceptron block packs a two-layer network into one brick: expand to a hidden size, apply an activation, project back. With <b>residual</b> switched on, its input is added to its output, so the block learns a <i>correction</i> — exactly the feed-forward part of a transformer.</P>
    </>,
    paramNotes: { hidden: 'The width of the inner layer — usually 2–4× the input width.', out_features: '0 keeps the input width (needed for residual).', residual: 'Add the input back to the output.' },
    code: [
      { code: 'net = nn.Sequential(', note: 'Runs the layers in order.' },
      { code: '    nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(p),', note: 'Expand, bend, regularise.' },
      { code: '    nn.Linear(hidden, d))' , note: 'Project back to d features.' },
      { code: 'y = x + net(x) if residual else net(x)', note: 'The residual shortcut.' },
    ],
    paper: { title: 'Learning representations by back-propagating errors', url: 'https://www.nature.com/articles/323533a0', authors: 'D. Rumelhart, G. Hinton, R. Williams', year: 1986 },
  },
  {
    block: 'batchnorm', section: 'general', title: 'Batch Norm & Layer Norm', summary: 'Keep signals at a healthy scale.', tags: ['general', 'image', 'text'], year: 2015,
    body: <>
      <P>As data flows through many layers, values can drift to be huge or tiny, making learning unstable. Normalisation layers re-centre them to mean 0 and spread 1, then let the model learn its preferred scale and shift.</P>
      <P><b>Batch Norm</b> (2015) normalises each feature across the examples in a batch — a big reason CNNs became easy to train. <b>Layer Norm</b> (2016) normalises across the features of each single example, so it works with any batch size and with sequences; it’s used in every transformer.</P>
    </>,
    code: [
      { code: 'nn.BatchNorm2d(channels)(x)', note: 'For images: statistics per channel over batch × height × width.' },
      { code: 'nn.LayerNorm(d)(x)', note: 'For vectors/sequences: statistics over the last dimension of each example.' },
    ],
    paper: { title: 'Batch Normalization: Accelerating Deep Network Training', url: 'https://arxiv.org/abs/1502.03167', authors: 'S. Ioffe, C. Szegedy', year: 2015 },
  },
  {
    block: 'conv2d', section: 'image', title: 'Convolution (Conv2D)', summary: 'Small filters that slide over the image.', tags: ['image'], year: 1989,
    body: <>
      <P>Flattening throws away the fact that neighbouring pixels belong together. A <b>convolution</b> keeps the image as a grid and slides small filters (e.g. 3×3) across it. Each filter is a little pattern detector — an edge, a curve, a texture — and because the same filter is reused at every position, a stroke is recognised wherever it appears.</P>
      <P>Each filter produces one output “channel” (feature map). Stack conv layers and later filters combine earlier ones into larger patterns. LeCun’s LeNet used this to read zip codes in 1989; CNNs dominated image recognition from 2012 to about 2020.</P>
      <P>Shapes: 1×28×28 in, Conv2D(32 filters, padding “same”) → 32×28×28 out. A stride of 2 halves the height and width.</P>
    </>,
    paramNotes: {
      out_channels: 'How many different filters to learn — the number of output channels.',
      kernel_size: 'The size of each filter: 3 means 3×3. Bigger kernels see more context but cost more.',
      stride: 'How far the filter jumps each step. 2 halves the output size.',
      padding: '“same” adds a border of zeros so the output keeps the input’s size (with stride 1).',
      batch_norm: 'Normalise each output channel (see Batch Norm).',
      activation: 'Non-linearity applied to every output value.',
    },
    code: [
      { code: 'conv = nn.Conv2d(in_channels=1, out_channels=32,', note: 'Learns 32 filters, each 1×3×3 (one per input channel).' },
      { code: '                 kernel_size=3, stride=1, padding=1)', note: 'padding=1 keeps 28×28 → 28×28.' },
      { code: 'y = torch.relu(conv(x))', note: 'x: (batch, 1, 28, 28) → y: (batch, 32, 28, 28).' },
      { code: '# weights: 32 × 1 × 3 × 3 = 288 (+32 biases)', note: 'Far fewer than a Linear layer on the same image!' },
    ],
    paper: { title: 'Backpropagation Applied to Handwritten Zip Code Recognition', url: 'https://doi.org/10.1162/neco.1989.1.4.541', authors: 'Y. LeCun et al.', year: 1989 },
    quiz: { q: 'Why does a conv layer need far fewer weights than a Linear layer on the same image?', options: ['It uses smaller numbers', 'The same small filter is reused at every position', 'It skips half the pixels', 'It has no biases'], answer: 1, explain: 'Weight sharing: one 3×3 filter is slid over the whole image.' },
  },
  {
    block: 'pooling', section: 'image', title: 'Pooling', summary: 'Shrink feature maps while keeping what matters.', tags: ['image'], year: 1989,
    body: <>
      <P>After a conv layer finds features, <b>pooling</b> summarises small windows to shrink the map: <b>Max Pool</b> keeps the strongest response in each 2×2 window, <b>Average Pool</b> keeps the mean. 32×28×28 → 32×14×14. This cuts computation and makes the model a little less sensitive to exact positions.</P>
      <P><b>Global Average Pool</b> averages each entire channel to a single number (C×H×W → C). It replaced big flattened Linear layers at the end of modern CNNs.</P>
    </>,
    paramNotes: { kernel_size: 'Window size (2 = 2×2).', stride: 'Step between windows; 0 = same as the window (no overlap).' },
    code: [
      { code: 'nn.MaxPool2d(2)(x)', note: '(B, C, 28, 28) → (B, C, 14, 14), keeping each window’s max.' },
      { code: 'nn.AvgPool2d(2)(x)', note: 'Same, but averaging.' },
      { code: 'x.mean(dim=(2, 3))', note: 'Global average pool: (B, C, H, W) → (B, C).' },
    ],
    paper: { title: 'Network In Network (global average pooling)', url: 'https://arxiv.org/abs/1312.4400', authors: 'M. Lin, Q. Chen, S. Yan', year: 2013 },
  },
  {
    block: 'resblock', section: 'image', title: 'Residual block', summary: 'Learn a correction and add the input back.', tags: ['image'], year: 2015,
    body: <>
      <P>Before 2015, making a CNN deeper than ~20 layers often made it <i>worse</i>: signals and gradients faded as they passed through so many layers. He et al. fixed this with a simple trick: add the block’s input to its output. The block only has to learn the <i>difference</i> F(x), and the shortcut gives gradients a highway back to early layers.</P>
      <P>ResNets with 152 layers won ImageNet in 2015. Shortcuts like this are now in almost every deep model, including transformers.</P>
    </>,
    paramNotes: { out_channels: '0 keeps the channel count. If it changes, the shortcut uses a 1×1 conv to match.', stride: '2 halves the resolution (the shortcut is downsampled to match).' },
    code: [
      { code: 'y = relu(bn1(conv1(x)))', note: 'First 3×3 conv + batch norm + ReLU.' },
      { code: 'y = bn2(conv2(y))', note: 'Second 3×3 conv + batch norm.' },
      { code: 'out = relu(y + shortcut(x))', note: 'Add the input back. shortcut is identity, or a 1×1 conv if shapes differ.' },
    ],
    paper: { title: 'Deep Residual Learning for Image Recognition', url: 'https://arxiv.org/abs/1512.03385', authors: 'K. He, X. Zhang, S. Ren, J. Sun', year: 2015 },
  },
  {
    block: 'depthwise_conv', section: 'image', title: 'Depthwise-separable convolution', summary: 'Cheap convolutions for phones.', tags: ['image'], year: 2017,
    body: <>
      <P>A normal 3×3 conv from 64 to 128 channels has 64×128×9 ≈ 74k weights. A depthwise-separable conv splits it: a 3×3 filter per input channel (64×9) plus a 1×1 conv to mix channels (64×128) — about 8.8k weights. MobileNet used this to run image recognition on phones.</P>
    </>,
    code: [
      { code: 'nn.Conv2d(c, c, 3, padding=1, groups=c)', note: 'groups=c: each channel gets its own filter (depthwise).' },
      { code: 'nn.Conv2d(c, out_c, kernel_size=1)', note: 'The 1×1 “pointwise” conv mixes channels.' },
    ],
    paper: { title: 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications', url: 'https://arxiv.org/abs/1704.04861', authors: 'A. Howard et al.', year: 2017 },
  },
  {
    block: 'patch_embed', section: 'image', title: 'Patch Embedding (Vision Transformer)', summary: 'An image is worth 16×16 words.', tags: ['image'], year: 2020,
    body: <>
      <P>Transformers revolutionised text. To use them on images, the Vision Transformer (ViT) cuts the picture into patches, turns each into a vector, and treats the patches as a sentence. Add a Positional Encoding (so it knows where each patch was), stack Transformer Blocks, pool, and classify.</P>
      <P>ViTs need lots of data to beat CNNs — on small datasets with short training, a CNN usually wins. Try both on MNIST and see.</P>
      <P>Typical ViT stack: Patch Embedding → Positional Encoding → Transformer Block ×4 → Layer Norm → Sequence Pool → Output Head.</P>
    </>,
    paramNotes: { patch_size: 'Side length of each square patch; must divide the image size (28 → 4 or 7).', dim: 'Length of each patch vector (the transformer’s width).' },
    code: [
      { code: 'proj = nn.Conv2d(c, dim, kernel_size=p, stride=p)', note: 'A conv with stride = kernel = patch size embeds each patch independently.' },
      { code: 'tokens = proj(x).flatten(2).transpose(1, 2)', note: '(B, dim, 7, 7) → (B, 49, dim): a sequence of 49 patch tokens.' },
    ],
    paper: { title: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale', url: 'https://arxiv.org/abs/2010.11929', authors: 'A. Dosovitskiy et al.', year: 2020 },
  },
  {
    block: 'mixer', section: 'image', title: 'MLP-Mixer', summary: 'Attention-free mixing with plain MLPs.', tags: ['image'], year: 2021,
    body: <>
      <P>In 2021 Google asked: is attention even necessary? MLP-Mixer replaces it with two simple MLPs applied alternately — one across patches, one across features — and gets surprisingly close to ViT. It shows that much of a transformer’s power comes from the overall recipe: patches, residual connections, normalisation and lots of data.</P>
      <P>Use it after a Patch Embedding, in place of Transformer Blocks.</P>
    </>,
    paramNotes: { token_hidden: 'Hidden size of the MLP that mixes across patches.', channel_hidden: 'Hidden size of the MLP that mixes across features.' },
    code: [
      { code: 'x = x + token_mlp(ln1(x).transpose(1, 2)).transpose(1, 2)', note: 'Mix across patches (transpose so patches are the last dim).' },
      { code: 'x = x + channel_mlp(ln2(x))', note: 'Mix across features for each patch.' },
    ],
    paper: { title: 'MLP-Mixer: An all-MLP Architecture for Vision', url: 'https://arxiv.org/abs/2105.01601', authors: 'I. Tolstikhin et al.', year: 2021 },
  },
  {
    block: 'image_to_sequence', section: 'image', title: 'Image → Sequence', summary: 'Read an image row by row.', tags: ['image', 'text'], year: 1990,
    body: <>
      <P>Sequence models (RNNs, LSTMs, transformers) read one step at a time. Image → Sequence lets them read pictures by treating each row of pixels as one step: a 1×28×28 digit becomes 28 steps of 28 numbers. Reading MNIST row by row with an RNN is a classic test of memory — by the last row, the model must remember the strokes it saw at the top.</P>
    </>,
    code: [
      { code: 'b, c, h, w = x.shape', note: 'e.g. (B, 1, 28, 28).' },
      { code: 'seq = x.permute(0, 2, 1, 3).reshape(b, h, c * w)', note: '→ (B, 28, 28): 28 time steps of 28 features.' },
    ],
  },
  {
    block: 'embedding', section: 'text', title: 'Token Embedding', summary: 'Words in, vectors out.', tags: ['text'], year: 2003,
    body: <>
      <P>Text datasets arrive as lists of <b>token ids</b> — each word (AG News) or character (Shakespeare) replaced by its number in a vocabulary. Ids are just labels: word #87 isn’t “bigger” than word #4. The Embedding layer gives every id its own learned vector, so similar words can end up with similar vectors.</P>
      <P>It’s always the first block of a text model: T token ids → T×D vectors. The simplest text classifier is just Embedding → Sequence Pool → Output Head (a “bag of words” — it ignores word order but works surprisingly well).</P>
    </>,
    paramNotes: { dim: 'Length of each word vector. 64–256 is typical for small models.' },
    code: [
      { code: 'emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=128)', note: 'A table with one learned row per token.' },
      { code: 'vectors = emb(token_ids)', note: '(B, 64) ids → (B, 64, 128) vectors.' },
    ],
    paper: { title: 'A Neural Probabilistic Language Model', url: 'https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf', authors: 'Y. Bengio et al.', year: 2003 },
  },
  {
    block: 'seq_pool', section: 'text', title: 'Sequence Pool', summary: 'Many vectors → one.', tags: ['text', 'image'],
    body: <>
      <P>A classifier needs one vector per example, but text models produce one vector per token. Sequence Pool combines them: <b>mean</b> (average — the usual choice), <b>max</b> (strongest signal per feature), or just the <b>first</b>/<b>last</b> token. Vision Transformers use it too, after the transformer blocks.</P>
    </>,
    paramNotes: { mode: 'How to combine the time steps.' },
    code: [{ code: 'pooled = x.mean(dim=1)', note: '(B, T, D) → (B, D).' }],
  },
  {
    block: 'rnn', section: 'text', title: 'Recurrent Neural Network (Elman)', summary: 'Read one step at a time, with memory.', tags: ['text'], year: 1990,
    body: <>
      <P>An RNN processes a sequence in order, keeping a <b>hidden state</b> — a vector that summarises everything seen so far. At each step it combines the new input with the previous state using the same weights. Elman showed in 1990 that such networks discover structure in language, like where words begin and end.</P>
      <P>The weakness: the state is squashed through tanh at every step, so information (and gradients) fade over long sequences — the <i>vanishing gradient</i> problem. Turn on gradient clipping when training RNNs.</P>
    </>,
    paramNotes: {
      hidden: 'Size of the memory vector.',
      layers: 'Stack several RNN layers inside one block.',
      bidirectional: 'Also read backwards and combine. Not allowed for predicting the next character (it would see the future).',
      return_sequences: 'On: output the state at every step (T×H) — needed for language modelling. Off: only the final state (H) — handy for classification.',
      nonlinearity: 'tanh (classic) or relu.',
    },
    code: [
      { code: 'rnn = nn.RNN(input_size=128, hidden_size=256, batch_first=True)', note: 'One cell, reused at every time step.' },
      { code: 'outputs, h_last = rnn(x)', note: 'x: (B, T, 128) → outputs: (B, T, 256), h_last: final state.' },
      { code: '# inside: h_t = tanh(W·x_t + U·h_{t-1} + b)', note: 'New memory from the current input and the old memory.' },
    ],
    paper: { title: 'Finding Structure in Time', url: 'https://doi.org/10.1207/s15516709cog1402_1', authors: 'J. Elman', year: 1990 },
  },
  {
    block: 'lstm', section: 'text', title: 'LSTM & GRU', summary: 'Gated memory that lasts.', tags: ['text'], year: 1997,
    body: <>
      <P>The <b>LSTM</b> (Long Short-Term Memory) fixes the RNN’s fading memory with a separate <b>cell state</b> that flows along with only gentle, controlled changes, plus <b>gates</b> that decide what to forget, what to write and what to output. It can remember things for hundreds of steps and powered speech recognition and translation for two decades.</P>
      <P>The <b>GRU</b> (2014) is a simplified version with two gates and no separate cell — often just as good and a bit faster. Both are in the palette.</P>
    </>,
    paramNotes: {
      hidden: 'Size of the memory.', layers: 'Stacked LSTM layers inside this block.', bidirectional: 'Read in both directions (classification only).',
      return_sequences: 'Output every step (on) or only the final state (off).', dropout: 'Dropout between stacked layers.',
    },
    code: [
      { code: 'lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True)', note: 'Two stacked LSTM layers.' },
      { code: 'outputs, (h, c) = lstm(x)', note: 'h = hidden state, c = cell state (the conveyor belt).' },
      { code: '# f = σ(...)  forget gate;  i = σ(...)  input gate', note: 'Gates are sigmoids: 0 = closed, 1 = open.' },
      { code: '# c_t = f * c_{t-1} + i * tanh(...)', note: 'Keep some old memory, add some new.' },
    ],
    paper: { title: 'Long Short-Term Memory', url: 'https://www.bioinf.jku.at/publications/older/2604.pdf', authors: 'S. Hochreiter, J. Schmidhuber', year: 1997 },
  },
  {
    block: 'self_attention', section: 'text', title: 'Self-Attention', summary: 'Every token looks at every other token.', tags: ['text', 'image'], year: 2014,
    body: <>
      <P>RNNs pass information one step at a time, so connecting the first word with the fiftieth takes fifty hops. <b>Attention</b> connects them directly: each token computes a <i>query</i>, every token offers a <i>key</i> and a <i>value</i>; the query–key match decides how much of each value to mix in. It first appeared in machine translation in 2014 and became the core of the transformer.</P>
      <P><b>Multi-head</b> attention runs several attentions in parallel, so different heads can track different relationships (grammar, meaning, position). A <b>causal mask</b> stops tokens looking ahead — essential when predicting the next character.</P>
    </>,
    paramNotes: { heads: 'Number of parallel attention heads; must divide the vector size.', causal: '“auto” = on for language modelling, off otherwise.' },
    code: [
      { code: 'attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)', note: 'Splits d into 4 heads.' },
      { code: 'h = layer_norm(x)', note: 'Pre-norm.' },
      { code: 'y = x + attn(h, h, h, attn_mask=causal_mask)[0]', note: 'Queries, keys and values all come from h (self-attention), plus a residual.' },
      { code: '# weights = softmax(Q·Kᵀ / √d) ; out = weights · V', note: 'The attention formula.' },
    ],
    paper: { title: 'Neural Machine Translation by Jointly Learning to Align and Translate', url: 'https://arxiv.org/abs/1409.0473', authors: 'D. Bahdanau, K. Cho, Y. Bengio', year: 2014 },
  },
  {
    block: 'positional_encoding', section: 'text', title: 'Positional Encoding', summary: 'Tell attention where each token is.', tags: ['text', 'image'], year: 2017,
    body: <>
      <P>Attention treats its input as a <i>set</i> — shuffle the tokens and it gives the same answer. Word order matters, so we add a position signal to each token’s vector: either fixed sine waves (the original transformer) or a learned vector per position. Always put it right after the Token or Patch Embedding.</P>
    </>,
    paramNotes: { kind: '“learned” trains a vector per position; “sinusoidal” uses fixed waves.', dropout: 'Dropout after adding positions.' },
    code: [
      { code: 'self.pe = nn.Parameter(torch.zeros(1, T, d))', note: 'Learned: one vector per position.' },
      { code: 'y = x + self.pe[:, :x.shape[1]]', note: 'Simply added to the token vectors.' },
    ],
    paper: { title: 'Attention Is All You Need', url: 'https://arxiv.org/abs/1706.03762', authors: 'A. Vaswani et al.', year: 2017 },
  },
  {
    block: 'transformer', section: 'text', title: 'Transformer Block', summary: 'Attention + MLP + shortcuts: the engine of GPT.', tags: ['text', 'image'], year: 2017,
    body: <>
      <P>The transformer block combines everything: <b>Layer Norm → multi-head self-attention → add input back</b>, then <b>Layer Norm → MLP → add back</b>. Attention lets tokens share information; the MLP processes each token on its own. Because there’s no step-by-step recurrence, it trains on all positions in parallel — which is why transformers scaled into GPT, BERT, and today’s language models.</P>
      <P>A tiny GPT: Token Embedding → Positional Encoding → Transformer Block ×4 (causal) → Layer Norm → Output Head. Try it on Tiny Shakespeare.</P>
    </>,
    paramNotes: {
      heads: 'Parallel attention heads; must divide the model width.', ff_mult: 'MLP hidden size as a multiple of the width (4 is standard).',
      dropout: 'Dropout inside attention and MLP.', causal: '“auto” = on for language modelling.', activation: 'MLP non-linearity (GELU is standard).',
      norm_first: 'Pre-norm (modern, more stable) vs post-norm (original 2017).',
    },
    code: [
      { code: 'x = x + dropout(attn(ln1(x)))', note: 'Sub-layer 1: self-attention with a residual.' },
      { code: 'x = x + dropout(ff(ln2(x)))', note: 'Sub-layer 2: feed-forward MLP with a residual.' },
      { code: '# ff = Linear(d, 4d) → GELU → Linear(4d, d)', note: 'The MLP expands 4× then contracts.' },
    ],
    paper: { title: 'Attention Is All You Need', url: 'https://arxiv.org/abs/1706.03762', authors: 'A. Vaswani et al.', year: 2017 },
    quiz: { q: 'What stops a GPT-style model from cheating by looking at the next character?', options: ['Dropout', 'A causal mask', 'Layer Norm', 'Positional encoding'], answer: 1, explain: 'The causal mask hides future positions from attention.' },
  },
]

// Blocks covered by another block's lesson.
export const LESSON_FOR_BLOCK: Record<string, string> = {
  maxpool2d: 'block.pooling', avgpool2d: 'block.pooling', global_avgpool: 'block.pooling', layernorm: 'block.batchnorm', gru: 'block.lstm',
}

for (const b of BLOCK_LESSONS) {
  LESSONS.push({ ...b, id: `block.${b.block}`, diagram: BLOCK_DIAGRAMS[b.block], action: 'addBlock' })
}

export const lessonForBlock = (type: string) => LESSON_FOR_BLOCK[type] ?? `block.${type}`
export const getLesson = (id: string) => LESSONS.find((l) => l.id === id)

/** The block type a lesson's “Add to my model” button inserts. */
export const blockOfLesson = (l: Lesson) => (l.block === 'pooling' ? 'maxpool2d' : l.block)
