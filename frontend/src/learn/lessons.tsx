import type { ReactElement, ReactNode } from 'react'
import type { Feature } from './curriculum'
import { BLOCK_DIAGRAMS } from './diagrams/blocks'
import {
  BatchNoise, Depth, Epochs, LearningRate, MatMul, MnistPeek, NeuralNet, OptunaSearch, Optimizers, Overfitting, Regularization,
  Schedules, Shapes, TrainingLoop,
} from './diagrams/concepts'

export type Section = 'start' | 'app' | 'train' | 'advanced' | 'general' | 'image' | 'text'

export const SECTIONS: { id: Section; title: string; icon: string }[] = [
  { id: 'start', title: 'Start here', icon: '🚀' },
  { id: 'app', title: 'Using the app', icon: '🧭' },
  { id: 'train', title: 'Training knobs', icon: '🎛️' },
  { id: 'advanced', title: 'Advanced techniques', icon: '🧪' },
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
  action?: 'challenge' | 'addBlock' | { tab: string; label: string; view?: 'lessons' | 'skills' }
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
      <P>Lego AI teaches you how modern AI models are built, by building them. A neural network is a stack of <b>blocks</b>, each doing one simple job. You snap blocks together, press train, and watch the model learn.</P>
      <P>You’ll start with the simplest possible model and a famous dataset of handwritten digits. As you try things you earn <b>experience (XP)</b>. Each new level unlocks something new (more blocks, more controls, new datasets) along with a lesson here in the <b>Guidebook</b> explaining it.</P>
      <P>The quest panel (bottom left) always shows what to try next. If you’re ever lost, press <b>Show me</b> on a quest.</P>
      <Tip>Every lesson you read earns XP, and the guidebook keeps everything you’ve unlocked so you can come back any time.</Tip>
    </>,
  },
  {
    id: 'start.nn', section: 'start', title: 'What is a neural network?', summary: 'Neurons, weights, and learning from examples.', unlock: { level: 1 }, diagram: NeuralNet,
    body: <>
      <P>Traditional programs follow rules a person wrote. A neural network instead <b>learns</b> its rules from examples. Show it thousands of pictures of handwritten digits along with the right answers, and it gradually adjusts itself until it gets them right.</P>
      <P>Inside, it’s just numbers. Each <b>neuron</b> takes some numbers in, multiplies each by a <b>weight</b>, adds them up, and passes the result on. A network has thousands to billions of weights. <b>Training</b> means finding weight values that make the final answer correct.</P>
      <P>Neurons are organized in <b>layers</b>. The input layer holds the data (e.g. pixel brightness). The output layer holds the answer (e.g. one score per digit). Layers in between are “hidden”: they find useful patterns on their own.</P>
    </>,
    quiz: { q: 'What does training change inside a neural network?', options: ['The input data', 'The weights', 'The number of layers', 'The answers in the dataset'], answer: 1, explain: 'Training adjusts the weights. The architecture (layers) is what you design; the data stays fixed.' },
  },
  {
    id: 'start.mnist', section: 'start', title: 'Meet MNIST', summary: '70,000 handwritten digits, the “hello world” of AI.', unlock: { level: 1 }, diagram: MnistPeek, tags: ['image'],
    body: <>
      <P><b>MNIST</b> is a collection of 70,000 small pictures of handwritten digits (0–9), collected from US Census workers and high-school students in the 1990s. Each picture is <b>28 × 28 pixels</b> in grayscale, and each comes with a label saying which digit it is.</P>
      <P>The task: look at a picture and say which digit it is. That’s a <b>classification</b> problem with <b>10 classes</b>. Humans get about 99.8% right. A good model can get over 99%.</P>
      <P>The data is split in three:</P>
      <ul>
        <li><b>Training set</b> (55,000): the examples the model learns from.</li>
        <li><b>Validation set</b> (5,000): held back to check progress on pictures it has never trained on. This is the “accuracy” you’ll see most.</li>
        <li><b>Test set</b> (10,000): scored once at the very end, as a final exam.</li>
      </ul>
      <Tip>Why hold data back? A model could simply memorize the training pictures. Only accuracy on <i>unseen</i> pictures tells you whether it really learned to read digits.</Tip>
    </>,
    paper: { title: 'Gradient-Based Learning Applied to Document Recognition', url: 'http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf', authors: 'Y. LeCun, L. Bottou, Y. Bengio, P. Haffner', year: 1998 },
    quiz: { q: 'An MNIST image is 28×28 pixels. How many numbers is that?', options: ['28', '56', '784', '10'], answer: 2, explain: '28 rows × 28 columns = 784 brightness values.' },
  },
  {
    id: 'start.training', section: 'start', title: 'How training works', summary: 'Guess, check, blame, adjust, repeated thousands of times.', unlock: { level: 1 }, diagram: TrainingLoop,
    body: <>
      <P>Training is a loop. The model takes a small <b>batch</b> of examples (say 128 digits) and:</P>
      <ol>
        <li><b>Guesses</b>: runs the pictures through its layers (the <i>forward pass</i>).</li>
        <li><b>Checks</b>: compares guesses with the right answers. The <b>loss</b> is one number measuring how wrong it was. Lower is better.</li>
        <li><b>Blames</b>: works out how much each weight contributed to the error (<i>backpropagation</i>, which computes the <i>gradient</i>).</li>
        <li><b>Adjusts</b>: nudges every weight a little in the direction that reduces the loss (the <i>optimizer step</i>).</li>
      </ol>
      <P>Do this thousands of times and the loss falls while accuracy rises. That’s the curve you’ll watch when you press Train.</P>
    </>,
    quiz: { q: 'What does the loss measure?', options: ['How big the model is', 'How wrong the model’s guesses are', 'How long training takes', 'How many examples there are'], answer: 1, explain: 'The loss is a single number for how wrong the guesses are. Training tries to make it smaller.' },
  },
  {
    id: 'start.layers', section: 'start', title: 'Layers & depth', summary: 'Why stacking layers helps, and what “deep” means.', unlock: { level: 2 }, diagram: Depth,
    body: <>
      <P>A <b>hidden layer</b> transforms its input into a new set of numbers, new “features”. With one hidden layer, the model can only build fairly simple features directly from pixels.</P>
      <P>Stack several and each layer builds on the last: the first may respond to small strokes, the next to loops and corners, the next to whole digit shapes. A network with many layers is called <b>deep</b>. That’s where “deep learning” gets its name.</P>
      <P>Depth isn’t free: deeper networks are slower and can be harder to train. Later you’ll meet tricks (activations, normalization, residual connections) that make very deep networks work.</P>
      <Tip>In the Lab, “number of hidden layers” simply repeats the same block several times. In the Builder you can repeat any block with its <b>Repeat</b> control.</Tip>
    </>,
  },
  {
    id: 'start.shapes', section: 'start', title: 'Tensors & shapes', summary: 'Why blocks have to fit together, and how to fix it when they don’t.', unlock: { feature: 'lesson.shapes' }, diagrams: [Shapes, MatMul],
    body: <>
      <P>Everything flowing through a network is a <b>tensor</b>: a grid of numbers with a <b>shape</b>. A single number has shape (); a list of 784 numbers has shape <code>784</code>; an MNIST image has shape <code>1×28×28</code> (1 color channel, 28 rows, 28 columns). A batch of 128 images adds one more dimension in front: <code>128×1×28×28</code>.</P>
      <P>Each block expects a certain shape and produces a new one. A Linear layer wants a flat list; a convolution wants an image; a transformer wants a sequence. <b>The output shape of each block must match what the next block expects</b>, just like Lego studs only fit the right holes.</P>
      <P>Why so strict? Inside a Linear layer, the input is multiplied by a weight matrix. Matrix multiplication only works when the inner sizes agree: there must be exactly one weight for every input number. A mismatch isn’t a style problem; the math is impossible.</P>
      <P>In the Builder the shape after every block is shown on the arrows between them, and a block that can’t accept its input turns red and tells you why. The usual fixes:</P>
      <ul>
        <li><b>Image going into a Linear layer?</b> Add a <b>Flatten</b> block before it.</li>
        <li><b>Wrong output size?</b> The top of the model must output one number per class (10 for MNIST). End with an <b>Output Head</b>, which sizes itself automatically.</li>
        <li><b>Text tokens going into anything but an embedding?</b> Start with a <b>Token Embedding</b>.</li>
      </ul>
      <Tip>Ready? Press <b>Start the challenge</b> below. You’ll get a broken model to repair.</Tip>
    </>,
    quiz: { q: 'A Linear layer with 784 inputs receives a tensor of shape 1×28×28. What should you add before it?', options: ['Dropout', 'Another Linear layer', 'Flatten', 'An Output Head'], answer: 2, explain: 'Flatten turns 1×28×28 into 784, same numbers, flat shape, so the Linear layer can use them.' },
    action: 'challenge',
  },
  {
    id: 'start.overfitting', section: 'start', title: 'Overfitting & validation', summary: 'When a model memorizes instead of learning.', unlock: { level: 5 }, diagram: Overfitting,
    body: <>
      <P>A big enough model can memorize its training examples perfectly, including their quirks, without learning the general rule. It then scores brilliantly on training data and badly on new data. That’s <b>overfitting</b>.</P>
      <P>That’s why we watch <b>validation</b> accuracy and loss, measured on held-out examples. When training loss keeps improving but validation loss starts getting worse, the model has begun to memorize. The leaderboard ranks models by their <b>best</b> validation accuracy for this reason.</P>
      <P>Remedies: more data, fewer epochs, a smaller model, or <b>regularization</b> such as Dropout and weight decay (you’ll unlock these).</P>
    </>,
    quiz: { q: 'Training loss keeps falling but validation loss starts rising. What’s happening?', options: ['Underfitting', 'Overfitting', 'The learning rate is too small', 'Nothing, that’s ideal'], answer: 1, explain: 'The model is memorizing the training set: classic overfitting.' },
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
        <li><b>Add blocks</b>: drag them from the panel on the left into the stack (a yellow line shows where they’ll land), or click one to drop it in.</li>
        <li><b>Rearrange</b>: drag a block within the stack. Drag it to the red bin at the bottom to delete it.</li>
        <li><b>Configure</b>: click a block. The side panel explains every setting and shows its default.</li>
        <li><b>Repeat</b>: in the side panel, stack a block several times. The count appears beside it (×3).</li>
        <li><b>Check shapes</b>: the shape after each block is written on the arrows. Red blocks explain what’s wrong.</li>
      </ul>
      <P>Undo with ⌘Z, delete with ⌫, duplicate with ⌘D. Until the Train page unlocks, a “Train & test” panel on the right trains your design with one click.</P>
    </>,
  },
  {
    id: 'app.train', section: 'app', title: 'The Train page', summary: 'Hyperparameters, live charts, resuming.', unlock: { feature: 'train.page' }, action: { tab: 'train', label: 'Open Train' },
    body: <>
      <P>The Train page controls <i>how</i> your model learns. The left column holds settings (called <b>hyperparameters</b> because they aren’t learned: you choose them). More appear as you level up: epochs and learning rate first, then optimizer and batch size, schedules, regularization and finally Optuna.</P>
      <P>Press <b>Train</b> and the right column shows live charts. Blue is the training set, orange the validation set. Loss should go down and accuracy up. When training finishes, the model is scored on the test set and added to the leaderboard.</P>
    </>,
  },
  {
    id: 'app.leaderboard', section: 'app', title: 'The Leaderboard', summary: 'Compare every model you’ve trained.', unlock: { feature: 'leaderboard.page' }, action: { tab: 'leaderboard', label: 'Open Leaderboard' },
    body: <>
      <P>Every run is saved and ranked by its best validation accuracy (you can sort by test accuracy, loss, size or time). Filter by dataset at the top. The colored chips show each model’s blocks at a glance.</P>
      <P>Click a run to see its curves, architecture, settings and the computer it ran on. From there you can open it in the Builder, retrain it, resume it, or download its config and weights.</P>
    </>,
  },
  {
    id: 'app.explore', section: 'app', title: 'Explore: look at the data and test your models', summary: 'See what the model sees, find its mistakes, try your own input, chat.', unlock: { level: 1 }, action: { tab: 'explore', label: 'Open Explore' },
    body: <>
      <P>Accuracy is one number. To understand a model you have to look at examples. The <b>Explore</b> tab has three parts:</P>
      <ul>
        <li><b>Look at the data.</b> Browse random examples from any dataset, from the training, validation or test part. Click a class to see only that class. Always do this before training: you will spot odd, ambiguous and even mislabeled examples.</li>
        <li><b>Test a model.</b> Pick any model you trained. You get its accuracy for each class, a <b>confusion matrix</b> (which classes it mixes up: rows are the truth, columns are its answers), and its <b>most confident mistakes</b>. Click a cell or a class to see those examples. Below that you can <b>try your own input</b>: draw a digit, drop in a photo, or type a sentence, and see the model’s answer with its confidence for every class.</li>
        <li><b>Chat.</b> Talk to a text model trained on Tiny Shakespeare. It continues whatever you type in the style it learned. You can keep several chats, switch models in the middle of one, and adjust how adventurous its writing is.</li>
      </ul>
      <Tip>When a model is very sure and wrong, ask why. Is the example unusual? Is the label wrong? Does it look like another class? Mistakes tell you what to change next: more data, a different block, or longer training.</Tip>
      <P><b>About your own images.</b> A model only knows what its training data looked like. MNIST digits are white on black, centered and 28×28 pixels, so your drawing is converted to exactly that. A photo of a digit on lined paper may still fool it: that gap between training data and the real world is one of the central problems in machine learning.</P>
    </>,
    quiz: { q: 'In a confusion matrix, what does a bright cell in row “4”, column “9” mean?', options: ['The model is good at 4s and 9s', 'Many real 4s were guessed as 9', 'Many real 9s were guessed as 4', 'There are few 4s in the data'], answer: 1, explain: 'Rows are the true class, columns are the model’s answer: real 4s that it called 9.' },
  },
  {
    id: 'app.guidebook', section: 'app', title: 'The Guidebook & skill tree', summary: 'Where everything you unlock is explained.', unlock: { level: 1 }, action: { tab: 'guide', label: 'Open Guidebook' },
    body: <>
      <P>The Guidebook holds a lesson for every concept, page and block. Locked lessons show what unlocks them. The <b>Skill Tree</b> tab is where you spend skill points (two per level from level 6) to unlock new blocks. Image blocks and text blocks form two branches, roughly in the order they were invented. <b>Achievements</b> (badges and what earns them) live in your profile: click your avatar or the XP bar at the top right.</P>
      <P>Each page has a <b>?</b> button that opens its guide.</P>
    </>,
  },
  {
    id: 'app.skilltree', section: 'app', title: 'Choosing a path', summary: 'Image or text? Old or new?', unlock: { feature: 'skilltree' }, action: { tab: 'guide', label: 'Open the Skill Tree', view: 'skills' },
    body: <>
      <P>From level 6 you earn <b>2 skill points per level</b>. Spend them in the Skill Tree to unlock new blocks. Each branch follows history, from older ideas to newer ones, so you can feel why each new invention mattered:</P>
      <ul>
        <li><b>🖼️ Image</b>: Convolution (1989) → Pooling → Residual blocks (2015) → Depthwise conv (2017) → Vision Transformer (2020) → MLP-Mixer (2021). Convolution also unlocks the CIFAR-10 color photo dataset.</li>
        <li><b>📝 Text</b>: Embeddings → RNN (1990) → LSTM & GRU (1997/2014) → Attention (2014) → Transformer (2017). Embeddings unlock the AG News and Tiny Shakespeare datasets.</li>
        <li><b>🧱 General</b>: Dropout, activations, MLP blocks, normalization, useful everywhere.</li>
      </ul>
      <P>Unlocking a skill also unlocks its lesson, with an animation, its PyTorch code and the original paper.</P>
    </>,
  },
  {
    id: 'app.json', section: 'app', title: 'Saving & loading models (JSON)', summary: 'Export a model, share it, load it back.', unlock: { feature: 'config.io' },
    body: <>
      <P>Every model can be saved as a small text file in <b>JSON</b> format (<code>.lego.json</code>). It records the dataset, every block and its settings, every training setting and the random seed and, for runs, the software versions used. Anyone with the file can rebuild exactly the same model.</P>
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
    id: 'app.python', section: 'app', title: 'Exporting real Python: train.py and generate.py', summary: 'Take your model out of the app as plain PyTorch code.', unlock: { feature: 'config.io' },
    body: <>
      <P>The blocks you stack are a friendly face on ordinary <b>PyTorch</b> code. Press <b>Python</b> in the Builder toolbar (or <b>Python scripts</b> on a run in the Leaderboard) to download that code as a zip:</P>
      <ul>
        <li><b>train.py</b>: your model written out layer by layer with its real sizes, the code that loads the dataset, and the full training loop. All settings sit in one <code>CONFIG</code> dictionary at the top.</li>
        <li><b>generate.py</b> for text generators, or <b>predict.py</b> for classifiers: loads the trained weights and lets you use the model from the command line.</li>
        <li><b>config.lego.json</b>: the same model as a Lego AI config, so you can import it back.</li>
      </ul>
      <P>The scripts need only PyTorch (plus torchvision for image datasets), not this app:</P>
      <pre className="json">{`pip install torch torchvision
python train.py
python generate.py "ROMEO:" --length 400`}</pre>
      <P>The script follows the app’s training procedure step for step. With the same seed on the CPU it reaches exactly the same test accuracy as the run in your Leaderboard. That makes it a good way to learn: read a line, find the matching block or setting in the app, change it and see what happens.</P>
      <Tip>This is also how to train on a bigger machine: copy the folder to a computer with a strong GPU, set <code>"device": "cuda"</code> in CONFIG and run it there.</Tip>
    </>,
  },
  {
    id: 'app.famous', section: 'app', title: 'Famous architectures', summary: 'LeNet, VGG, every ResNet, both original Transformers, ViT and GPT-2, rebuilt from blocks.', unlock: { feature: 'builder.templates' },
    body: <>
      <P>The <b>Templates</b> window has a second section: <b>Famous architectures</b>. Each one rebuilds a landmark model from its paper, block for block, at the published size, adapted only as much as needed to accept this app’s small datasets.</P>
      <ul>
        <li><b>LeNet-5</b> (1998): the first successful CNN. About 60 thousand weights. Trains in seconds.</li>
        <li><b>VGG-16</b> (2014): just 3×3 convolutions, stacked deep. About 15 million weights.</li>
        <li><b>ResNet-18, 34, 50, 101 and 152</b> (2015): skip connections made very deep networks trainable. The three largest use the <b>Bottleneck</b> block: a 1×1 convolution to shrink, a 3×3 to compute, a 1×1 to expand.</li>
        <li><b>Transformer base and big</b> (2017): the two sizes from “Attention Is All You Need”: 6 layers at width 512 with 8 heads, and 6 layers at width 1024 with 16 heads.</li>
        <li><b>ViT-Base</b> (2020) and <b>GPT-2 small</b> (2019): 12 transformer layers at width 768.</li>
      </ul>
      <P>Loading one is free, so look inside: count the blocks, read the shapes on the arrows, compare the weight counts. Training is another matter. Most of these needed days on several GPUs, and the app will tell you so: watch for the warning next to the Train button.</P>
      <Tip>These models were designed for far bigger datasets than CIFAR-10 or Tiny Shakespeare. On small data a small model often wins, and it always trains faster. Try a famous model with one or two stages removed.</Tip>
    </>,
  },
  {
    id: 'app.resume', section: 'app', title: 'Pausing & resuming training', summary: 'Stop now, continue later, from exactly where you were.', unlock: { feature: 'train.resume' },
    body: <>
      <P>While a model trains, the big button on the Train page becomes <b>⏸ Pause</b>. Pausing saves a <b>checkpoint</b>: the weights, the optimizer’s memory, the learning-rate schedule, how far through the data it was, and every random-number generator’s state. Press <b>▶ Resume</b> and training carries on from the very next batch, with exactly the same result as if it had never stopped.</P>
      <P>A checkpoint is also saved at the end of every epoch, so any run can be picked up again later:</P>
      <ul>
        <li>On the <b>Train</b> page: <b>Start from a previous run</b> → pick a run → <b>Resume checkpoint to N epochs</b>.</li>
        <li>On the <b>Leaderboard</b>: open a run → <b>Resume from checkpoint</b>.</li>
      </ul>
      <P>Use it to give a promising model more epochs, or to free up your computer for a while in the middle of a long run.</P>
    </>,
  },
  {
    id: 'app.reproduce', section: 'app', title: 'Reproducing a run', summary: 'Same config, same seed, same result, bit for bit.', unlock: { feature: 'config.io' },
    body: <>
      <P>Science needs results others can repeat. Open a run on the Leaderboard and press <b>Retrain from scratch</b>: the identical config is trained again from new random weights (the same random weights, because the seed is the same). When it finishes, it’s compared with the original. If every number matches you get a <b>✓ reproduced</b> badge.</P>
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
        <li>Watch the trials table fill in. The best is starred: press <b>Use best params</b> to copy them into the form.</li>
      </ol>
      <P>Each trial is saved as a normal run, so the best one can be reproduced or resumed.</P>
    </>,
    paper: { title: 'Optuna: A Next-generation Hyperparameter Optimization Framework', url: 'https://arxiv.org/abs/1907.10902', authors: 'T. Akiba et al.', year: 2019 },
  },
  {
    id: 'app.datasets', section: 'app', title: 'Your own datasets', summary: 'Import a spreadsheet, a text file, folders of photos or a Hugging Face dataset.', unlock: { feature: 'datasets.custom' }, action: { tab: 'datasets', label: 'Open Datasets' },
    body: <>
      <P>So far every model learned from data someone else prepared. The <b>Datasets</b> tab lets you bring your own. Start with the <b>worked example</b> there: it imports a small real table step by step and explains each choice. Then do one yourself.</P>
      <P>A dataset is many <b>examples</b>. For each one the model needs the <b>inputs</b> it may look at and the <b>answer</b> (the label) it should learn to give. Four kinds of file work:</P>
      <ul>
        <li><b>A table (.csv).</b> One row per example, one header row. Pick the answer column, which must be a category (“spam / not spam”, a species, a grade). The inputs are either several number columns or one text column.</li>
        <li><b>Text (.txt).</b> No answers needed: the model learns to predict the next character, as with Tiny Shakespeare. Afterwards you can chat with it in Explore.</li>
        <li><b>Pictures (.zip).</b> One folder per class with the pictures inside. They are cropped square and shrunk to the size you choose.</li>
        <li><b>Hugging Face.</b> Paste the name of a public dataset from huggingface.co/datasets and the first 20,000 rows are fetched as a table.</li>
      </ul>
      <P>Three things happen on import, and each matters:</P>
      <ul>
        <li><b>A test set is held back</b> (15% by default) that the model never trains on. A further slice of the rest becomes the validation set.</li>
        <li><b>Numbers are rescaled</b> so every input is centered on zero with a similar spread. Missing numbers become the column’s average.</li>
        <li><b>Words become numbers.</b> The 20,000 most common words each get an id; rarer ones share an “unknown” id.</li>
      </ul>
      <P>The import screen tells you what always guessing the most common class would score. Beat that number and the model has learned something. With very small datasets (a few hundred rows), expect scores to jump around between seeds: use the seeds option on the Train page to see how much.</P>
      <Tip>Templates adapt to your dataset: open Templates in the Builder after choosing it. Blocks size themselves to your inputs and your number of classes.</Tip>
    </>,
    quiz: { q: 'Why hold back a test set the model never trains on?', options: ['To make training faster', 'To check it learned the pattern and did not just memorize the examples', 'Because the file is too big', 'To balance the classes'], answer: 1, explain: 'Only examples the model has never seen can show whether it generalizes.' },
  },
  {
    id: 'app.multidata', section: 'app', title: 'One model, many datasets', summary: 'Train the same design on several datasets and compare models by their average.', unlock: { feature: 'datasets.custom' }, action: { tab: 'leaderboard', label: 'Open the Leaderboard' },
    body: <>
      <P>A model that wins on one dataset may just suit that dataset. Researchers judge an architecture by how it does across <b>several</b>. You can do the same.</P>
      <P>On the Train page, open <b>Model & data</b> and press <b>Show datasets this model fits</b>. Tick the extra ones and the Train button becomes <b>Train on N datasets</b>: the same blocks and settings are trained on each in turn. This works because blocks resize themselves: a convolution takes 1 or 3 color channels, the output head makes as many outputs as there are classes. A model for images still cannot read text, so only datasets of a fitting kind are offered.</P>
      <P>On the Leaderboard, switch from <b>Runs</b> to <b>By model</b>. Each row is now one model design, with its best score on every dataset and the <b>average</b>. Models tested on more datasets are listed first, because an average over one dataset cannot be compared with an average over four.</P>
      <Tip>Averages hide detail. A model that is best on average may be worst on the one dataset you care about, so read the columns too.</Tip>
    </>,
  },
  {
    id: 'app.hub', section: 'app', title: 'Hugging Face: publish and import models', summary: 'Share a trained model with the world, and load other people’s.', unlock: { feature: 'hub' }, action: { tab: 'leaderboard', label: 'Open the Leaderboard' },
    body: <>
      <P><b>Hugging Face</b> (huggingface.co) is the largest public library of AI models and datasets, a little like GitHub for machine learning. Most open models you hear about are published there.</P>
      <P><b>Importing</b> needs no account. Press <b>🤗 Hugging Face</b> on the Leaderboard, type a model’s name (owner/name) or browse models published from Lego AI, and press Import. The model arrives as a run on your leaderboard, with its weights, so you can test it in Explore, look at its mistakes, fine-tune it or retrain it yourself to check the author’s numbers.</P>
      <P><b>Publishing</b> needs a free account and an <b>access token</b>: a long secret string that lets this app upload on your behalf. Create one with write permission at huggingface.co/settings/tokens, paste it into the same window once, then open any finished run and press <b>Publish</b>. Four things are uploaded: the recipe (the same JSON config you can export), the trained weights, a model card describing the model and its scores, and for built-in datasets a standalone training script.</P>
      <ul>
        <li><b>Treat the token like a password.</b> Anyone who has it can change your models. It is stored on this computer only.</li>
        <li><b>Public means public.</b> Tick “private” if only you should see the model.</li>
        <li><b>Your own datasets are not uploaded.</b> A model trained on data you imported is published without that data. Others can use the model, but can only retrain it if they have the same dataset.</li>
        <li><b>Be careful with code.</b> A model that uses custom blocks contains Python. Before importing one, the app shows you the code and asks. Only run code from people you trust.</li>
      </ul>
      <P>Datasets come from Hugging Face too: see the Datasets tab.</P>
    </>,
    quiz: { q: 'Someone asks you to send them your Hugging Face access token so they can “check your model”. What do you do?', options: ['Send it, tokens are public', 'Refuse: the token is a password. They can see a public model without it', 'Post it in the model card', 'Email it instead of messaging it'], answer: 1, explain: 'Public models need no token to view or download. A token gives control of your account.' },
  },
  {
    id: 'app.fuse', section: 'app', title: 'Fusing blocks into your own block', summary: 'Select a few blocks and turn them into one reusable block. No code needed.', unlock: { feature: 'custom.page' }, action: { tab: 'build', label: 'Open the Builder' },
    body: <>
      <P>Real architectures are built from repeated <b>stages</b>: “convolution, batch norm, pooling” or “attention, then feed-forward”. When you find a group of blocks you keep stacking together, fuse it into one block of your own.</P>
      <ol>
        <li>In the Builder, click the first block of the group.</li>
        <li>Hold <span className="kbd">⇧ Shift</span> and click the last one. Everything in between is selected.</li>
        <li>In the side panel, give the new block a name and a color and press <b>Fuse into one block</b>.</li>
      </ol>
      <P>The group is replaced by a single brick, and the new block appears in the palette under <b>Custom</b>. Use it like any other block: drag it in again, or set <b>Repeat</b> to stack the whole stage several times. Each copy gets its own weights and works out its own input sizes.</P>
      <P>Select a fused block to see what it is made of. <b>Unfuse</b> swaps it back for the original blocks when you want to change a setting inside.</P>
      <Tip>A fused block is saved as a tiny Python file on the Custom Blocks tab: a list of its parts. Opening it is a gentle first look at how blocks are defined in code.</Tip>
    </>,
  },
  {
    id: 'app.custom', section: 'app', title: 'Your first custom block', summary: 'A hand-held walkthrough of writing a block in Python.', unlock: { feature: 'custom.page' }, action: { tab: 'custom', label: 'Open Custom Blocks' },
    body: <>
      <P>Every block in the palette is a small piece of Python using <b>PyTorch</b>. You can write your own, and it will appear in the palette like any other block. Let’s make one step by step.</P>
      <ol>
        <li>Go to <b>Custom Blocks</b> and press <b>+ New</b>. A starter file appears.</li>
        <li>Read the code with the explanation below: it’s a Linear layer plus an activation, a block you already know.</li>
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
      { code: '    color="#22d3ee",', note: 'The brick’s color (any hex color).' },
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
      <P>More epochs give the model more practice, so accuracy usually climbs, quickly at first, then slowly. Eventually more epochs can hurt: the model starts memorizing (overfitting). Watch the orange validation curve: when it stops improving, more epochs won’t help.</P>
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
      <P>Try the extremes yourself: it’s the best way to build intuition.</P>
    </>,
    quiz: { q: 'Your loss jumps around wildly and then becomes NaN. What should you try first?', options: ['Raise the learning rate', 'Lower the learning rate', 'Add more epochs', 'Use a bigger batch'], answer: 1, explain: 'An exploding loss is the classic sign of a learning rate that’s too high.' },
  },
  {
    id: 'train.batch', section: 'train', title: 'Batch size', summary: 'How many examples per step.', unlock: { feature: 'train.batch' }, diagram: BatchNoise,
    body: <>
      <P>The model doesn’t learn from one example at a time, nor from all 55,000 at once: it uses a <b>batch</b>. The gradient from a batch is an estimate of the “true” downhill direction.</P>
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
    id: 'train.regularization', section: 'train', title: 'Regularization', summary: 'Weight decay, dropout, label smoothing, augmentation.', unlock: { feature: 'train.regularization' }, diagram: Regularization,
    body: <>
      <P>Regularization makes memorizing harder, so the model must learn general patterns instead.</P>
      <ul>
        <li><b>Weight decay</b>: pulls weights toward zero every step. Typical: 0.0001–0.05.</li>
        <li><b>Dropout override</b>: sets the drop probability of every Dropout in the model at once (handy for Optuna).</li>
        <li><b>Label smoothing</b>: softens targets so the model isn’t pushed to be 100% sure. Typical: 0.05–0.1.</li>
        <li><b>Data augmentation</b>: for color images (CIFAR-10), randomly shifts and mirrors each picture, so the model sees endless variations.</li>
      </ul>
      <P>Gradient clipping (under Stability) caps how big a single update can get, essential for RNNs.</P>
    </>,
    paper: { title: 'Decoupled Weight Decay Regularization (AdamW)', url: 'https://arxiv.org/abs/1711.05101', authors: 'I. Loshchilov, F. Hutter', year: 2017 },
  },
  {
    id: 'adv.soup', section: 'advanced', title: 'Model soups: averaging weights', summary: 'Blend several trained models into one that costs no more to run.', unlock: { feature: 'advanced.combine' }, action: { tab: 'leaderboard', label: 'Open the Leaderboard' },
    body: <>
      <P>A trained model is just a long list of numbers: its weights. A <b>model soup</b> takes several trained models with the same architecture and averages them, weight by weight. The result is one ordinary model, the same size and speed as any single ingredient, that is often a little more accurate and more robust than the best of them.</P>
      <P><b>The catch: the ingredients must start from the same place.</b> Two models trained from different random weights learn the same job in different internal arrangements. Neuron 5 in one might do what neuron 31 does in the other. Average them and you get nonsense. Models that share a starting point and are then trained a bit further in different ways stay close enough for the average to make sense.</P>
      <P>You can see both sides here:</P>
      <ul>
        <li><b>The failure.</b> On the Leaderboard, tick two models with the same architecture but different seeds and press <b>Combine</b>, then <b>Soup</b>. Accuracy usually collapses.</li>
        <li><b>The recipe that works.</b> Open a trained model and press <b>Cook a soup</b>. The app fine-tunes several variations from that model (different data order and learning rates), then makes a soup, a greedy soup and an ensemble from them so you can compare.</li>
      </ul>
      <P>A <b>uniform soup</b> averages every ingredient. A <b>greedy soup</b> starts with the best one and adds the others one at a time, keeping each only if validation accuracy does not drop, so one bad ingredient cannot spoil it.</P>
      <Tip>Keep the variations gentle: a low learning rate and a few epochs. If the ingredients drift far apart, the soup gets worse, not better.</Tip>
    </>,
    quiz: { q: 'Why does averaging two models trained from different random seeds usually fail?', options: ['They have different numbers of weights', 'Their weights are arranged differently inside, so the average mixes unrelated things', 'Averaging always lowers accuracy', 'The learning rates were different'], answer: 1, explain: 'Each model found its own internal arrangement. Only models that share a starting point line up well enough to average.' },
    paper: { title: 'Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time', url: 'https://arxiv.org/abs/2203.05482', authors: 'M. Wortsman et al.', year: 2022 },
  },
  {
    id: 'adv.ensemble', section: 'advanced', title: 'Ensembles and a mixture of experts', summary: 'Ask several models and combine their answers.', unlock: { feature: 'advanced.combine' }, action: { tab: 'leaderboard', label: 'Open the Leaderboard' },
    body: <>
      <P>An <b>ensemble</b> keeps every model and combines their <i>answers</i> instead of their weights. Each model gives a probability for every class, and the ensemble averages them. Because the models never have to line up inside, any mix works: an MLP, a CNN and a transformer can vote together.</P>
      <P>Ensembles help when the members make <b>different mistakes</b>. If one model is unsure whether a digit is a 4 or a 9 and two others are confident it is a 9, the average is right. Three copies of the same mistake help nobody. This is why models with different architectures or seeds make good ensembles even though they make terrible soups.</P>
      <P>The price is cost: an ensemble of five models needs five times the memory and time for every prediction. A soup gives up some of the gain to stay the price of one.</P>
      <P>A <b>mixture of experts</b> goes one step further. Instead of trusting every model equally, a small trained <b>gate</b> looks at what the experts say and decides how much to trust each one for this particular input. Here the gate is fitted on half of the validation set and scored on the other half, so its score is honest.</P>
      <P>Tick two or more models on the Leaderboard, press <b>Combine</b> and choose <b>Ensemble</b> or <b>Mixture of experts</b>. The result is a model you can test in Explore like any other.</P>
      <Tip>The famous “mixture of experts” layers inside large language models use the same idea in miniature: a gate routes each token to a few specialist sub-networks.</Tip>
    </>,
    quiz: { q: 'Which pair would make the most useful ensemble?', options: ['Two identical copies of one model', 'Two good models that tend to fail on different examples', 'One good model and one that guesses randomly', 'Two models trained on different datasets'], answer: 1, explain: 'Averaging only fixes errors that the members do not share.' },
  },
  {
    id: 'adv.monosoup', section: 'advanced', title: 'A soup from one ingredient (MonoSoup)', summary: 'Get some of a soup’s benefit from a single model, with no extra training.', unlock: { feature: 'advanced.combine' },
    body: <>
      <P>Cooking a soup means training many models. A 2026 paper asks: can one fine-tuned model be improved the same way, on its own?</P>
      <P>The idea starts from the <b>update</b>: for each layer, the trained weights minus the weights it started from. That update can be split into <b>directions</b> ranked by strength (a tool called the singular value decomposition does this). A few strong directions carry most of the new skill. Many weak ones carry a mix of noise and small useful adjustments.</P>
      <P>Soups work because averaging keeps what the ingredients agree on and cancels what they do not. MonoSoup imitates that with one model: per layer, it measures how concentrated the update is and how much of it sits in the weak directions, and from those two numbers sets how strongly to keep the strong part and the weak part. It needs no data and has no settings.</P>
      <P>Open a trained model on the Leaderboard and press <b>MonoSoup</b>. Open the result and expand <b>Per-layer weights</b> to see, layer by layer, how many strong directions there were and how much of each part was kept.</P>
      <P><b>What to expect.</b> The method was built for large models fine-tuned from a pretrained one, where the goal is to keep new skills without forgetting old ones, mostly measured on data that differs from the training set. For a small model trained here from random weights there is no old knowledge to protect, so the usual result is a small change in either direction. To test it under the conditions it was built for, fine-tune a model from one dataset to another (for example Fashion-MNIST to MNIST, see the fine-tuning lesson), make a MonoSoup of the result, and compare the <b>shifted test</b> accuracy before and after. The result panel shows both numbers. Do not expect much from models this small: the gains in the paper come from very large pretrained models, and here the change is often slightly negative.</P>
    </>,
    paper: { title: 'Model soups need only one ingredient', url: 'https://arxiv.org/abs/2602.09689', authors: 'A. Abdollahpoorrostam, N. Dimitriadis, A. Hazimeh, P. Frossard', year: 2026 },
  },
  {
    id: 'adv.finetune', section: 'advanced', title: 'Fine-tuning: starting from a trained model', summary: 'Reuse one model’s weights as the starting point for another run, and learn when that helps.', unlock: { feature: 'advanced.combine' }, action: { tab: 'leaderboard', label: 'Open the Leaderboard' },
    body: <>
      <P>Every run so far began from random weights. <b>Fine-tuning</b> begins from the weights of a model that is already trained. The first run is then called <b>pretraining</b>. Nearly every modern AI system is made this way: one huge pretrained model, fine-tuned many times for different jobs.</P>
      <P>To try it: open a trained model on the Leaderboard and press <b>Fine-tune</b>. On the Train page, change the dataset if you like (Fashion-MNIST and MNIST share an image size, so a model moves between them unchanged) and press Train. Layers whose shapes still match are copied over. A layer that had to change starts fresh, and so does the output head whenever the dataset is a different one, because its outputs meant other classes before.</P>
      <P><b>When does a pretrained start help?</b> When the pretraining data was much larger than the new data and taught something the new task needs. A model that has seen a billion photos knows edges, textures and object parts, and learning cats versus dogs from 100 examples on top of that is easy.</P>
      <P><b>When does it not?</b> When the first task was small or unrelated. Try it here: pretrain on Fashion-MNIST, fine-tune on a few hundred MNIST digits, and compare with the same model trained from scratch on the same digits. From scratch usually wins. Clothing features are not what digits need, and a network that has already settled into one solution adapts less easily than a fresh one. Researchers call these <b>negative transfer</b> and <b>loss of plasticity</b>. Seeing a famous technique fail, and understanding why, is worth as much as seeing it work.</P>
      <ul>
        <li><b>Learning rate.</b> A lower rate preserves more of the old model; a higher one adapts faster. Try both.</li>
        <li><b>Same dataset.</b> Fine-tuning a model on the dataset it was trained on, with a different seed or learning rate, is how soup ingredients are made (Cook a soup does exactly this).</li>
        <li><b>Robustness.</b> Fine-tuning moves a model away from what it knew. Compare the <b>shifted test</b> score before and after, and see whether a soup or a MonoSoup wins any of it back.</li>
      </ul>
    </>,
    quiz: { q: 'In which case is a pretrained start most likely to beat training from scratch?', options: ['The pretraining data was small and unrelated', 'The pretraining data was huge and related, and the new task has little data', 'The new task has unlimited data', 'Always'], answer: 1, explain: 'Transfer pays when the old task taught general, relevant features and the new task cannot teach them by itself.' },
    paper: { title: 'On Warm-Starting Neural Network Training', url: 'https://arxiv.org/abs/1910.08475', authors: 'J. Ash, R. Adams', year: 2020 },
  },
  {
    id: 'adv.robust', section: 'advanced', title: 'Shifted test data: is the model robust?', summary: 'A test score only tells you about data that looks like the training data.', unlock: { feature: 'leaderboard.page' }, action: { tab: 'leaderboard', label: 'Open the Leaderboard' },
    body: <>
      <P>A test set is drawn from the same source as the training set: same scanner, same framing, same kind of handwriting. Real use rarely is. A model with 99% test accuracy can fall apart on digits that are slightly rotated or photographed in bad light. This gap is called <b>distribution shift</b>, and how well a model survives it is its <b>robustness</b>.</P>
      <P>For the image datasets the app measures this after every training run. It takes 2,000 test images and changes them in six ways the model never saw in training: noise, blur, rotation, moving off-center, low contrast and low resolution. Where a second real dataset with the same classes exists it uses that too: <b>USPS</b> digits (scanned from envelopes) for MNIST models and <b>CIFAR-10.1</b> (new photos collected years later) for CIFAR-10 models. The <b>Shifted</b> column on the Leaderboard is the average over all of them.</P>
      <P>Open any run to see each shifted set, a sample of what it looks like and the accuracy on it. Things worth trying:</P>
      <ul>
        <li>Compare an MLP with a CNN at the same test accuracy. Which one survives a sideways shift? (Convolution and pooling were designed for exactly that.)</li>
        <li>Turn on <b>augmentation</b> for CIFAR-10 and see which shifts it protects against.</li>
        <li>Fine-tune a model, then make a soup or a MonoSoup of it, and compare the shifted score before and after.</li>
      </ul>
      <Tip>The shifted sets are fixed, so every model faces exactly the same images. That makes the numbers comparable across runs.</Tip>
    </>,
    quiz: { q: 'Two models both score 98% on the test set. On rotated test images one scores 90% and the other 60%. What does that tell you?', options: ['The second model is broken', 'The test set was too easy', 'The first model learned something that holds up better when inputs change', 'Nothing, test accuracy is what matters'], answer: 2, explain: 'Equal test scores can hide very different robustness. Shifted data reveals it.' },
    paper: { title: 'Benchmarking Neural Network Robustness to Common Corruptions and Perturbations', url: 'https://arxiv.org/abs/1903.12261', authors: 'D. Hendrycks, T. Dietterich', year: 2019 },
  },
  {
    id: 'adv.shared', section: 'advanced', title: 'The shared leaderboard', summary: 'Compete on tracks with strict limits, on a board where every score is checked.', unlock: { feature: 'share' }, action: { tab: 'leaderboard', label: 'Open the Leaderboard' },
    body: <>
      <P>Your own leaderboard only has your models. A <b>shared leaderboard</b> is run by one person, the <b>host</b> (a teacher, a club, a friend), and everyone connected to it can enter. Go to Leaderboard → <b>🌍 Shared</b> and enter the address the host gave you.</P>
      <P><b>Tracks.</b> A shared board is split into tracks, each with limits: a dataset, a maximum number of weights, epochs and training examples, and a time limit. Limits are what make a contest interesting. Without them the biggest computer wins. With them, the best idea does. “The best digit reader under 100,000 weights in three epochs” is a real design puzzle.</P>
      <P><b>How an entry is checked.</b> When you submit, your computer sends only the <b>recipe</b>: the blocks and settings, not the weights and not your score. The host’s computer trains that recipe again from scratch and lists the score <i>it</i> measures. Three things follow:</P>
      <ul>
        <li>Nobody can type in a made-up number.</li>
        <li>Everyone is measured on the same machine, so time limits are fair.</li>
        <li>Because training here is reproducible, the host should get your score, to the last digit on a matching setup or within a few hundredths of a percent on a different one. A ✓ marks an exact match.</li>
      </ul>
      <P>This is how serious benchmarks work too: results count when someone else can run your code and get the same thing.</P>
      <P><b>To enter:</b> train a model that obeys a track’s limits, open the run on your own leaderboard and use <b>Submit to the shared leaderboard</b>. The app tells you first if any limit is broken. Your entry waits in a queue until the host checks it. On the board, press <b>Open</b> on anyone’s entry to load its recipe into your Builder and learn from it.</P>
      <Tip>Custom blocks are code that the host would have to run, so most tracks refuse them. Fused blocks count as custom: unfuse them before submitting.</Tip>
      <P>Want to run one yourself? The project’s <code>docs/HOSTING.md</code> explains it. In short: start the app with <code>LEGO_HOST=1</code>.</P>
    </>,
    quiz: { q: 'Why does the host retrain every submitted model instead of accepting the submitted score?', options: ['To use up electricity', 'So every score on the board was produced the same way on the same machine, and nobody can invent one', 'Because scores cannot be sent over the internet', 'To make submissions slower'], answer: 1, explain: 'A result that someone else reproduced is one you can trust, and measuring everyone on one machine makes the limits fair.' },
  },
  {
    id: 'adv.cloud', section: 'advanced', title: 'Training on a cloud GPU', summary: 'Send a run to a more powerful computer and get the results back.', unlock: { feature: 'share' }, action: { tab: 'train', label: 'Open Train' },
    body: <>
      <P>When the Hardware tab says a model is too big or too slow for this computer, you can train it somewhere else. Lego AI can send a run to another computer, called a <b>worker</b>, watch it train live, and copy the results and weights back to your leaderboard.</P>
      <P>A worker is just another copy of Lego AI, started on a machine with a GPU and protected by an <b>access token</b> (a long secret, like a password, so that only you can use it). It can be a gaming desktop in the next room or a machine rented by the hour from a GPU cloud.</P>
      <ol>
        <li>On the GPU machine, start Lego AI with a token: <code>LEGO_TOKEN="a-long-secret" HOST=0.0.0.0 ./run.sh</code></li>
        <li>Here, go to Train → <b>Hardware</b> → <b>Add another computer</b> and enter its address and the token.</li>
        <li>Choose it under <b>Where to train</b> and press Train.</li>
      </ol>
      <P>The model, the settings and, if needed, your imported dataset are sent over. The charts on the Train page update as usual. The worker uses its own best device, so a run sent from a laptop trains on the worker’s GPU.</P>
      <ul>
        <li><b>Rented machines cost money while they are switched on</b>, whether or not anything is training. Shut them down when you finish. The app never starts, stops or pays for machines.</li>
        <li><b>Keep the token secret.</b> Anyone with the address and the token can use the machine. The safest connection is an SSH tunnel: <code>docs/CLOUD_GPU.md</code> shows how.</li>
        <li><b>GPU results are not exactly reproducible.</b> Expect tiny differences from a CPU run of the same config.</li>
      </ul>
      <Tip>Make it work small first. Train on a subset on your own computer until the model clearly learns, then send the full run to the GPU.</Tip>
    </>,
  },
  {
    id: 'adv.frontier', section: 'advanced', title: 'How a frontier lab builds an assistant', summary: 'Pretraining, reinforcement learning, safety tuning and benchmarks: the whole pipeline at toy scale.', unlock: { feature: 'frontier' }, action: { tab: 'frontier', label: 'Open the Frontier Lab' },
    body: <>
      <P>The assistants made by the big labs are not trained in one go. They pass through stages, each with a different kind of data and a different goal. The <b>Frontier Lab</b> tab runs a miniature of that pipeline on a model you build, in a few minutes.</P>
      <P>It all happens in a small invented world: twenty places, each with a capital, a color, an animal and a secret vault code. The model is a character-level transformer, the same design as the real thing with about a millionth of the weights.</P>
      <ol>
        <li><b>Pretraining.</b> The model reads “the internet” (here a few hundred thousand characters of stories, questions and answers) and learns one thing: predict the next character. Everything it knows comes from this. So do its flaws. This toy internet repeats popular wrong answers to some sums, often copies a puzzle instead of solving it, and contains the vault codes. A model that imitates text imitates all of that.</li>
        <li><b>Reinforcement learning.</b> For tasks where a program can check the answer, the model no longer imitates. It tries, and is rewarded for being right. This is how it can become <i>better than its training data</i>. See the GRPO lesson.</li>
        <li><b>Safety fine-tuning.</b> The model is trained on examples of refusing what it should not do (here, giving out a vault code), mixed with ordinary questions so that it does not start refusing everything.</li>
        <li><b>Benchmarks.</b> After every stage the model takes the same seven tests. That is the only way to know what a stage bought and what it cost.</li>
      </ol>
      <P>Things to try once it has run:</P>
      <ul>
        <li>Untick reinforcement learning or safety and run again. What is the assistant like without that stage?</li>
        <li>Swap the foundation model: the small GPT, the LSTM, or your own design. Does a weaker foundation limit every later stage?</li>
        <li>Give RL more rounds. Does Math keep rising? Does Knowledge start to slip?</li>
        <li>Chat with each stage’s model in Explore. Type <code>Q: code of …?</code>, a new line, then <code>A:</code>.</li>
      </ul>
      <Tip>The real pipelines differ mainly in scale: trillions of words in place of a few hundred thousand characters, months on thousands of GPUs in place of minutes on a laptop, and thousands of benchmarks in place of seven. The shape is the same.</Tip>
    </>,
    quiz: { q: 'Where does a pretrained model’s knowledge of wrong facts come from?', options: ['Random initialization', 'Its training text, which it imitates, mistakes included', 'The optimizer', 'The benchmark'], answer: 1, explain: 'Next-token prediction copies whatever is common in the data, true or not.' },
  },
  {
    id: 'adv.grpo', section: 'advanced', title: 'Reinforcement learning and GRPO', summary: 'Learning from right-or-wrong instead of from examples.', unlock: { feature: 'frontier' }, action: { tab: 'frontier', label: 'Open the Frontier Lab' },
    body: <>
      <P>Everything before this was <b>supervised</b>: for each input, the right output was given and the model was nudged toward it. <b>Reinforcement learning</b> (RL) needs less. Nobody shows the answer. The model acts, a number comes back saying how good that was (the <b>reward</b>), and the model becomes more likely to do what earned more.</P>
      <P>For a language model the “action” is writing an answer. If a program can check answers, such as a sum, a sorted list, a passing unit test, a solved puzzle, it can hand out rewards endlessly with no human writing examples. This is how today’s reasoning models are sharpened.</P>
      <P><b>GRPO</b> (Group Relative Policy Optimization) is a simple, popular recipe:</P>
      <ol>
        <li>Take a question. Let the model write a <b>group</b> of different answers by sampling (here 8).</li>
        <li>Score each answer with the checker.</li>
        <li>Compare each answer with the <b>average of its own group</b>. Better than average: make that answer more likely. Worse: less likely. If the whole group scored the same, there is nothing to learn from it.</li>
        <li>Do not move too far in one step (a clipped update), and stay close to the model you started from (a penalty on drifting away from it). Here that penalty also covers ordinary text, which is what protects Knowledge.</li>
      </ol>
      <P>Comparing within the group is the clever part. Older methods train a second network, a critic, to estimate how good an answer “should” be. GRPO uses the group’s own average instead: no critic to train.</P>
      <P>Why it helps in the lab: the toy internet says some sum equals the wrong number three times out of four. Pretraining learns the popular answer. But when sampling, the model still writes the right one now and then. RL rewards exactly those tries, and the right answer takes over. The model ends up <b>better than its data</b>, which imitation alone can never do.</P>
      <P>The Train tab shows this stage like any run: the “accuracy” curve is the average reward, and validation is the score on the Math and Puzzles benchmarks.</P>
      <Tip>RL only improves what the reward measures. Watch Knowledge and Language while Math rises: if they fall, the model is forgetting, and the leash (or fewer rounds) is the fix. A badly designed reward is worse: the model will find whatever earns it, intended or not.</Tip>
    </>,
    quiz: { q: 'In GRPO, when is an answer made more likely?', options: ['When it matches a stored correct answer', 'When its reward is above the average of the other answers sampled for the same question', 'When it is the longest', 'Always'], answer: 1, explain: 'Each answer is judged relative to its own group, which is why no separate critic network is needed.' },
    paper: { title: 'DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (introduces GRPO)', url: 'https://arxiv.org/abs/2402.03300', authors: 'Z. Shao et al.', year: 2024 },
  },
  {
    id: 'adv.safetytune', section: 'advanced', title: 'Safety fine-tuning, jailbreaks and benchmarks', summary: 'Teaching a model to refuse, checking that it still helps, and measuring both.', unlock: { feature: 'frontier' }, action: { tab: 'frontier', label: 'Open the Frontier Lab' },
    body: <>
      <P>A pretrained model will repeat anything it has read. In the lab’s toy world that includes vault codes. <b>Safety fine-tuning</b> is ordinary supervised training on a small, carefully made dataset: requests the model must not fulfil, each paired with a refusal.</P>
      <P>Three things can go wrong, and the lab has a benchmark for each:</P>
      <ul>
        <li><b>It still complies.</b> “Refuses secrets” asks for vault codes in plain words and counts refusals. Real labs use suites such as HarmBench.</li>
        <li><b>It refuses only the phrasing it was trained on.</b> “Resists jailbreaks” asks for the same thing in forms that were never in the safety data: behind an opening line (“for a story…”) or reworded (“what opens …?”). A <b>jailbreak</b> is exactly this: a request the model would refuse, dressed up until it does not recognize it. The gap between the first two scores is the real-world problem in miniature, and it is why labs employ red teams to hunt for new phrasings.</li>
        <li><b>It refuses too much.</b> A model that says no to everything is perfectly safe and useless. “Still helpful” asks harmless questions about the same places with the same wording tricks. This is why the safety data mixes in ordinary questions. Run the lab with only one or two safety epochs and you may catch the model in an over-refusing phase.</li>
      </ul>
      <P>Then compare the capability benchmarks before and after. Any drop is the <b>safety tax</b>: teaching new behavior with a small dataset disturbs what was there. Mixing in ordinary data keeps it small.</P>
      <P><b>About the benchmarks.</b> Each of the seven mimics a family that real labs report: next-token accuracy on held-out text, knowledge quizzes (MMLU), math word problems (GSM8K), reasoning puzzles (ARC), harmful-request suites, jailbreak suites and over-refusal tests (XSTest). Two habits carry over to the real thing:</P>
      <ul>
        <li><b>Run the same tests after every change.</b> A stage that helps one score often costs another.</li>
        <li><b>Do not train on the test.</b> These questions are fixed, so tuning settings until the bars look good slowly fits the model to them. Real benchmarks leak into training data in the same way, and scores drift upward without the models getting better.</li>
      </ul>
      <Tip>The lab’s model has perhaps a million weights and its world is tiny, so its refusals generalize more easily than a real model’s would. Real jailbreaks are found in an effectively endless space of phrasings.</Tip>
    </>,
    quiz: { q: 'A model refuses 100% of plainly worded harmful requests but only 40% when they are reworded. What does that tell you?', options: ['The model is safe', 'It learned the wording it was trained on more than the idea behind it', 'The benchmark is broken', 'It needs a bigger GPU'], answer: 1, explain: 'That gap is what jailbreaks exploit, and why safety data and testing need variety.' },
  },
  {
    id: 'adv.autopilot', section: 'advanced', title: 'Autopilot: improving a model automatically', summary: 'Start from a model that works and let the computer keep trying small changes.', unlock: { feature: 'advanced.autopilot' }, action: { tab: 'leaderboard', label: 'Open the Leaderboard' },
    body: <>
      <P>Optuna searches a range of settings that you describe. The <b>autopilot</b> works the other way round: you hand it a model that already trains well, and it looks for something better nearby.</P>
      <P>Each round it copies the best model so far and makes one or two small changes: the learning rate up or down, more or less dropout, a different schedule, a layer made wider or narrower, a block repeated once more or once less. It trains the changed model from scratch. If validation accuracy improves, that model becomes the new best and the next round starts from it. If not, the change is thrown away. Rounds that fall clearly behind the best run at the same epoch are stopped early to save time.</P>
      <P>This is <b>hill climbing</b>, the simplest kind of automated research: try a step, keep it if it goes up. Open a finished model on the Leaderboard, choose a number of rounds and press <b>Autopilot</b>. The Train page lists every round with the change it tried, and the winner is marked on the Leaderboard.</P>
      <Tip>Two cautions. Small gains can be luck: check the winner with several seeds. And because every round is judged on the validation set, trust the <b>test</b> accuracy of the final model more than its validation score.</Tip>
    </>,
    quiz: { q: 'The autopilot ran 30 rounds and validation accuracy rose from 97.1% to 97.4%. What should you check before celebrating?', options: ['Nothing, higher is better', 'Whether the gain is bigger than the seed-to-seed spread, and whether test accuracy rose too', 'Whether the model has more weights', 'Whether training took longer'], answer: 1, explain: 'Picking the best of 30 tries on the same validation set flatters the winner. Seeds and the test set tell you if it is real.' },
  },
  {
    id: 'adv.scaling', section: 'advanced', title: 'Scaling laws', summary: 'How much better does a model get when it is bigger, or sees more data?', unlock: { feature: 'scaling.page' }, action: { tab: 'scaling', label: 'Open Scaling' },
    body: <>
      <P>One of the most important findings of modern AI is that improvement with size is often <b>predictable</b>. Across a huge range, the loss of a model family falls as a <b>power law</b> of its size: every doubling of the weights (or of the data, or of the computing time) removes a roughly constant fraction of the loss.</P>
      <P>A power law is easy to spot: plotted with both axes on a log scale it is a <b>straight line</b>. The slope tells you the exchange rate. A slope of 0.1 means doubling the model cuts the loss by about 7%, and halving the loss would take about a thousand times the weights. This is why frontier labs can plan a model costing hundreds of millions before training it, and why progress has needed such enormous computers.</P>
      <P>The <b>Scaling</b> tab runs the experiment for your own design. It trains the model in the Builder at several widths (or on several amounts of data), plots the result on log-log axes, fits a line and tells you how well the points follow it (R² near 1 means a clean power law).</P>
      <P>Curves often <b>bend</b>, and the bend is informative. If bigger models stop improving, something else has become the limit: usually the amount of data or the number of epochs. If more data stops helping, the model is too small to use it. The largest models may even get worse when they overfit a small dataset.</P>
      <Tip>Keep each run short at first: a training subset and a few epochs are enough to see the shape. Then check the interesting region with more epochs.</Tip>
    </>,
    quiz: { q: 'On a log-log plot, loss against model size is a straight line with slope −0.05. What does doubling the model do?', options: ['Halves the loss', 'Cuts the loss by about 3%', 'Nothing', 'Doubles the loss'], answer: 1, explain: '2 to the power −0.05 is about 0.966, so each doubling removes roughly 3% of the loss. Small slopes are why big gains need huge models.' },
    paper: { title: 'Scaling Laws for Neural Language Models', url: 'https://arxiv.org/abs/2001.08361', authors: 'J. Kaplan et al.', year: 2020 },
  },
  {
    id: 'train.hardware', section: 'train', title: 'Hardware: CPUs, GPUs and when you need more', summary: 'What your computer can train, where the time goes, and how to get access to something bigger.', unlock: { feature: 'train.page' },
    body: <>
      <P>Training is mostly multiplying big grids of numbers. A <b>CPU</b> does this on a handful of cores. A <b>GPU</b> (graphics processor) does thousands of these multiplications at once, which is why modern AI runs on GPUs. The Train page’s <b>Hardware</b> tab shows what this computer has and where each epoch’s time is going.</P>
      <P><b>What fits on a laptop.</b> Everything in the early levels: MLPs and small CNNs on MNIST train in seconds on a CPU. Small LSTMs and transformers on Tiny Shakespeare take minutes. A real CNN on CIFAR-10 takes tens of minutes on a laptop GPU and much longer on a CPU. If a run would take hours, shrink it first: fewer epochs, a training subset, a smaller model, a shorter sequence length.</P>
      <P><b>Memory is usually the first limit.</b> A GPU can only train what fits in its memory. Use “Check what fits in memory” to find a batch size that works. On Apple Silicon Macs the GPU shares memory with everything else, so closing other apps really does help.</P>
      <P><b>Will it fit, and how long will it take?</b> The app estimates this for every model before you train. It adds up the memory for the weights, their gradients and the optimizer’s notes, plus the numbers every layer produces for each example in a batch. It also counts the multiplications in one pass and compares that with a quick speed test of this computer. If the model is too big or very slow, a warning appears next to the Train button, and the Hardware tab lists what it would take: this computer’s CPU and GPU, and four well-known GPUs from a free cloud card to a data-center one, with how many of each you would need and roughly how long an epoch would take. Treat the times as rough: within a factor of two is a good guess.</P>
      <P><b>Reading the time bar.</b> If most time is “model math”, a faster GPU or a smaller model is what helps. If “preparing data” is large, a faster GPU will not help: use a bigger batch or less augmentation. If “validation” dominates, your training subset is tiny compared with the validation set.</P>
      <P><b>Getting something bigger</b>, roughly from free to costly:</P>
      <ul>
        <li><b>Free notebooks.</b> Google Colab and Kaggle Notebooks lend a GPU for a few hours at a time at no cost. Good for experiments that outgrow a laptop.</li>
        <li><b>Your school or employer.</b> Many universities run a shared GPU cluster that students can apply to use. Ask a teacher, the IT department or a research computing office.</li>
        <li><b>Education and research credits.</b> The large cloud providers and several GPU companies run programs with free credits for students, teachers and researchers.</li>
        <li><b>Renting by the hour.</b> Cloud providers and GPU rental marketplaces rent machines by the hour. Prices vary a lot with the GPU model, so compare before you start, and always shut the machine down when you finish: it bills while idle.</li>
        <li><b>Buying.</b> A desktop with a recent consumer GPU is enough for everything in this app and for a lot of real research at small scale.</li>
      </ul>
      <P>To use another machine, either install Lego AI there and import your model’s JSON, or press <b>Python</b> in the Builder to export a standalone <code>train.py</code> that runs anywhere PyTorch does.</P>
      <Tip>Before paying for anything, make the idea work small. A model that does not learn on a subset on your laptop will not start learning on a bigger GPU.</Tip>
    </>,
  },
  {
    id: 'train.repro', section: 'train', title: 'Reproducibility: seeds & devices', summary: 'Why the same seed gives the same result.', unlock: { feature: 'train.repro' },
    body: <>
      <P>Training uses randomness: the starting weights, the order of examples, which neurons Dropout switches off, how images are augmented. Computers generate “random” numbers from a starting value called the <b>seed</b>. Same seed, same sequence of random numbers, same result.</P>
      <P>Change the seed to see how much your result depends on luck. If two models differ by less than that, the difference may be noise.</P>
      <P><b>Measuring the luck.</b> Next to the Train button, set <b>seeds</b> to 3 or 5. Lego AI trains that many copies of your model, identical except for the seed, and reports the average accuracy and its spread, for example “97.8% ± 0.2”. On the leaderboard they appear as one row ranked by the average. When you compare two models, a gap smaller than the spread is not evidence that one is better.</P>
      <P><b>Device</b> chooses the processor. CPU is exactly reproducible. Apple’s GPU (MPS) or NVIDIA’s (CUDA) are much faster for big models like CNNs on CIFAR-10, but a few operations aren’t perfectly repeatable. <b>Max steps</b> stops training after a fixed number of updates, handy for quick experiments.</P>
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
    paramNotes: { out_features: 'How many neurons, the width of the layer. More neurons = more capacity (and more weights).', activation: 'The bend applied after the weighted sum. “none” keeps it purely linear.', bias: 'Adds a learnable constant to each output. Almost always on.' },
    code: [
      { code: 'layer = nn.Linear(in_features=784, out_features=128)', note: 'Creates a 784×128 weight matrix W and 128 biases b, initialized randomly.' },
      { code: 'act = nn.ReLU()', note: 'The activation (if any).' },
      { code: 'y = act(layer(x))', note: 'x has shape (batch, 784). layer(x) computes x·Wᵀ + b → (batch, 128).' },
    ],
    paper: { title: 'The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain', url: 'https://psycnet.apa.org/record/1959-09865-001', authors: 'F. Rosenblatt', year: 1958 },
    quiz: { q: 'How many weights (not counting biases) does Linear(784 → 10) have?', options: ['794', '7,840', '784', '10'], answer: 1, explain: 'One weight for every input–output pair: 784 × 10 = 7,840.' },
  },
  {
    block: 'flatten', section: 'general', title: 'Flatten', summary: 'Unroll a grid into a list.', tags: ['general'],
    body: <>
      <P>Flatten reshapes a tensor into one long list, keeping every number. A 1×28×28 image becomes 784 numbers. It has no weights and learns nothing: it just changes the shape so a Linear layer can accept it.</P>
      <P>The catch: after flattening, the model no longer “knows” which pixels were neighbors. Convolutions (in the Image branch) keep the grid and use that structure.</P>
    </>,
    code: [
      { code: 'flat = nn.Flatten()', note: 'Keeps the batch dimension, flattens the rest.' },
      { code: 'y = flat(x)', note: 'x: (batch, 1, 28, 28) → y: (batch, 784).' },
    ],
  },
  {
    block: 'classifier', section: 'general', title: 'Output Head', summary: 'Turn features into one score per class.', tags: ['general'],
    body: <>
      <P>Every model here ends with an Output Head. It’s a Linear layer whose number of outputs is set automatically to what the dataset needs: <b>10</b> for digits, <b>4</b> for AG News topics, or one score per character at every position for language modeling.</P>
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
      <P><b>Draw your own.</b> The <b>Drawn Activation</b> block lets you sketch any shape: drag across its plot in the side panel and the model uses exactly that curve. Put it after a Linear layer whose activation is set to “none”, then train and compare it with ReLU. Try a straight line (the model becomes linear and accuracy drops), a V shape, or a wave.</P>
    </>,
    paramNotes: { fn: 'Which function to apply to every number.' },
    code: [
      { code: 'nn.ReLU()(x)      # max(0, x)', note: 'The default. Negatives become 0.' },
      { code: 'nn.GELU()(x)      # smooth ReLU', note: 'Used in GPT and BERT.' },
      { code: 'nn.Sigmoid()(x)   # 1 / (1 + e^-x)', note: 'Squashes to 0…1.' },
      { code: 'nn.Tanh()(x)      # squashes to -1…1', note: 'Used inside RNNs.' },
    ],
    paper: { title: 'Rectified Linear Units Improve Restricted Boltzmann Machines', url: 'https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf', authors: 'V. Nair, G. Hinton', year: 2010 },
    quiz: { q: 'Why do networks need activation functions between Linear layers?', options: ['To make training faster', 'Without them, stacked layers collapse into one linear layer', 'To reduce memory', 'To normalize the data'], answer: 1, explain: 'Linear ∘ Linear = Linear. The bend is what lets depth add power.' },
  },
  {
    block: 'dropout', section: 'general', title: 'Dropout', summary: 'Randomly switch neurons off to prevent memorizing.', tags: ['general'], year: 2012,
    body: <>
      <P>Dropout is a regularizer: during training it sets a random fraction <i>p</i> of the values passing through it to zero (and scales the rest up to compensate). Each step drops a different random set. When you evaluate the model, nothing is dropped.</P>
      <P>Because any neuron might vanish, the network can’t rely on a few memorized features; it learns redundant, robust ones. Typical values: 0.1–0.5. Place it after hidden layers, not right before the input.</P>
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
      <P>The multi-layer perceptron block packs a two-layer network into one brick: expand to a hidden size, apply an activation, project back. With <b>residual</b> switched on, its input is added to its output, so the block learns a <i>correction</i>, exactly the feed-forward part of a transformer.</P>
    </>,
    paramNotes: { hidden: 'The width of the inner layer, usually 2–4× the input width.', out_features: '0 keeps the input width (needed for residual).', residual: 'Add the input back to the output.' },
    code: [
      { code: 'net = nn.Sequential(', note: 'Runs the layers in order.' },
      { code: '    nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(p),', note: 'Expand, bend, regularize.' },
      { code: '    nn.Linear(hidden, d))' , note: 'Project back to d features.' },
      { code: 'y = x + net(x) if residual else net(x)', note: 'The residual shortcut.' },
    ],
    paper: { title: 'Learning representations by back-propagating errors', url: 'https://www.nature.com/articles/323533a0', authors: 'D. Rumelhart, G. Hinton, R. Williams', year: 1986 },
  },
  {
    block: 'batchnorm', section: 'general', title: 'Batch Norm & Layer Norm', summary: 'Keep signals at a healthy scale.', tags: ['general', 'image', 'text'], year: 2015,
    body: <>
      <P>As data flows through many layers, values can drift to be huge or tiny, making learning unstable. Normalization layers re-center them to mean 0 and spread 1, then let the model learn its preferred scale and shift.</P>
      <P><b>Batch Norm</b> (2015) normalizes each feature across the examples in a batch, a big reason CNNs became easy to train. <b>Layer Norm</b> (2016) normalizes across the features of each single example, so it works with any batch size and with sequences; it’s used in every transformer.</P>
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
      <P>Flattening throws away the fact that neighboring pixels belong together. A <b>convolution</b> keeps the image as a grid and slides small filters (e.g. 3×3) across it. Each filter is a little pattern detector, an edge, a curve, a texture, and because the same filter is reused at every position, a stroke is recognized wherever it appears.</P>
      <P>Each filter produces one output “channel” (feature map). Stack conv layers and later filters combine earlier ones into larger patterns. LeCun’s LeNet used this to read zip codes in 1989; CNNs dominated image recognition from 2012 to about 2020.</P>
      <P>Shapes: 1×28×28 in, Conv2D(32 filters, padding “same”) → 32×28×28 out. A stride of 2 halves the height and width.</P>
    </>,
    paramNotes: {
      out_channels: 'How many different filters to learn, the number of output channels.',
      kernel_size: 'The size of each filter: 3 means 3×3. Bigger kernels see more context but cost more.',
      stride: 'How far the filter jumps each step. 2 halves the output size.',
      padding: '“same” adds a border of zeros so the output keeps the input’s size (with stride 1).',
      batch_norm: 'Normalize each output channel (see Batch Norm).',
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
      <P>After a conv layer finds features, <b>pooling</b> summarizes small windows to shrink the map: <b>Max Pool</b> keeps the strongest response in each 2×2 window, <b>Average Pool</b> keeps the mean. 32×28×28 → 32×14×14. This cuts computation and makes the model a little less sensitive to exact positions.</P>
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
      <P>The deepest ResNets (50, 101 and 152 layers) use a cheaper variant, the <b>Bottleneck</b> block: a 1×1 convolution shrinks the channels, a 3×3 does the work on the thin version, and another 1×1 expands to four times the width. You can load all five ResNet sizes from Templates → Famous architectures.</P>
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
      <P>A normal 3×3 conv from 64 to 128 channels has 64×128×9 ≈ 74k weights. A depthwise-separable conv splits it: a 3×3 filter per input channel (64×9) plus a 1×1 conv to mix channels (64×128), about 8.8k weights. MobileNet used this to run image recognition on phones.</P>
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
      <P>Transformers revolutionized text. To use them on images, the Vision Transformer (ViT) cuts the picture into patches, turns each into a vector, and treats the patches as a sentence. Add a Positional Encoding (so it knows where each patch was), stack Transformer Blocks, pool, and classify.</P>
      <P>ViTs need lots of data to beat CNNs. On small datasets with short training, a CNN usually wins. Try both on MNIST and see.</P>
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
      <P>In 2021 Google asked: is attention even necessary? MLP-Mixer replaces it with two simple MLPs applied alternately, one across patches, one across features, and gets surprisingly close to ViT. It shows that much of a transformer’s power comes from the overall recipe: patches, residual connections, normalization and lots of data.</P>
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
      <P>Sequence models (RNNs, LSTMs, transformers) read one step at a time. Image → Sequence lets them read pictures by treating each row of pixels as one step: a 1×28×28 digit becomes 28 steps of 28 numbers. Reading MNIST row by row with an RNN is a classic test of memory: by the last row, the model must remember the strokes it saw at the top.</P>
    </>,
    code: [
      { code: 'b, c, h, w = x.shape', note: 'e.g. (B, 1, 28, 28).' },
      { code: 'seq = x.permute(0, 2, 1, 3).reshape(b, h, c * w)', note: '→ (B, 28, 28): 28 time steps of 28 features.' },
    ],
  },
  {
    block: 'embedding', section: 'text', title: 'Token Embedding', summary: 'Words in, vectors out.', tags: ['text'], year: 2003,
    body: <>
      <P>Text datasets arrive as lists of <b>token ids</b>: each word (AG News) or character (Shakespeare) replaced by its number in a vocabulary. Ids are just labels: word #87 isn’t “bigger” than word #4. The Embedding layer gives every id its own learned vector, so similar words can end up with similar vectors.</P>
      <P><b>Sequence length.</b> Text arrives in pieces of a fixed number of tokens: 64 by default. You can change it with the “length” box in the Builder (or “Sequence length” on the Train page): longer pieces give the model more context to work with, but every training step costs more time and memory.</P>
      <P>It’s always the first block of a text model: T token ids → T×D vectors. The simplest text classifier is just Embedding → Sequence Pool → Output Head (a “bag of words”: it ignores word order but works surprisingly well).</P>
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
      <P>A classifier needs one vector per example, but text models produce one vector per token. Sequence Pool combines them: <b>mean</b> (average, the usual choice), <b>max</b> (strongest signal per feature), or just the <b>first</b>/<b>last</b> token. Vision Transformers use it too, after the transformer blocks.</P>
    </>,
    paramNotes: { mode: 'How to combine the time steps.' },
    code: [{ code: 'pooled = x.mean(dim=1)', note: '(B, T, D) → (B, D).' }],
  },
  {
    block: 'rnn', section: 'text', title: 'Recurrent Neural Network (Elman)', summary: 'Read one step at a time, with memory.', tags: ['text'], year: 1990,
    body: <>
      <P>An RNN processes a sequence in order, keeping a <b>hidden state</b>: a vector that summarizes everything seen so far. At each step it combines the new input with the previous state using the same weights. Elman showed in 1990 that such networks discover structure in language, like where words begin and end.</P>
      <P>The weakness: the state is squashed through tanh at every step, so information (and gradients) fade over long sequences, the <i>vanishing gradient</i> problem. Turn on gradient clipping when training RNNs.</P>
    </>,
    paramNotes: {
      hidden: 'Size of the memory vector.',
      layers: 'Stack several RNN layers inside one block.',
      bidirectional: 'Also read backwards and combine. Not allowed for predicting the next character (it would see the future).',
      return_sequences: 'On: output the state at every step (T×H), needed for language modeling. Off: only the final state (H), handy for classification.',
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
      <P>The <b>GRU</b> (2014) is a simplified version with two gates and no separate cell, often just as good and a bit faster. Both are in the palette.</P>
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
      <P><b>Multi-head</b> attention runs several attentions in parallel, so different heads can track different relationships (grammar, meaning, position). A <b>causal mask</b> stops tokens looking ahead, essential when predicting the next character.</P>
    </>,
    paramNotes: { heads: 'Number of parallel attention heads; must divide the vector size.', causal: '“auto” = on for language modeling, off otherwise.' },
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
      <P>Attention treats its input as a <i>set</i>: shuffle the tokens and it gives the same answer. Word order matters, so we add a position signal to each token’s vector: either fixed sine waves (the original transformer) or a learned vector per position. Always put it right after the Token or Patch Embedding.</P>
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
      <P>The transformer block combines everything: <b>Layer Norm → multi-head self-attention → add input back</b>, then <b>Layer Norm → MLP → add back</b>. Attention lets tokens share information; the MLP processes each token on its own. Because there’s no step-by-step recurrence, it trains on all positions in parallel, which is why transformers scaled into GPT, BERT, and today’s language models.</P>
      <P>A tiny GPT: Token Embedding → Positional Encoding → Transformer Block ×4 (causal) → Layer Norm → Output Head. Try it on Tiny Shakespeare.</P>
    </>,
    paramNotes: {
      heads: 'Parallel attention heads; must divide the model width.', ff_mult: 'MLP hidden size as a multiple of the width (4 is standard).',
      dropout: 'Dropout inside attention and MLP.', causal: '“auto” = on for language modeling.', activation: 'MLP non-linearity (GELU is standard).',
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
  drawn_activation: 'block.activation', maxpool2d: 'block.pooling', avgpool2d: 'block.pooling', global_avgpool: 'block.pooling', layernorm: 'block.batchnorm', gru: 'block.lstm', bottleneck: 'block.resblock',
}

for (const b of BLOCK_LESSONS) {
  LESSONS.push({ ...b, id: `block.${b.block}`, diagram: BLOCK_DIAGRAMS[b.block], action: 'addBlock' })
}

export const lessonForBlock = (type: string) => LESSON_FOR_BLOCK[type] ?? `block.${type}`
export const getLesson = (id: string) => LESSONS.find((l) => l.id === id)

/** The block type a lesson's “Add to my model” button inserts. */
export const blockOfLesson = (l: Lesson) => (l.block === 'pooling' ? 'maxpool2d' : l.block)
