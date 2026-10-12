/**
 * The learning path: levels (and what each unlocks), quests ("new things to try"),
 * achievements, the skill tree and avatars.
 *
 * Everything here is data; `engine.ts` evaluates it against the learner's stats.
 */

export type Feature =
  | 'lab' // the simplified first page (one hidden layer, width presets, auto-train)
  | 'lab.layers' // choose how many hidden layers in the Lab (= repeating a block)
  | 'builder' // full drag-and-drop builder with block palette
  | 'builder.repeat'
  | 'builder.drag'
  | 'lesson.shapes'
  | 'train.page'
  | 'train.epochs'
  | 'train.lr'
  | 'leaderboard.page'
  | 'skilltree'
  | 'datasets'
  | 'train.optimizer'
  | 'train.batch'
  | 'config.io'
  | 'train.resume'
  | 'builder.templates'
  | 'train.schedule'
  | 'train.regularization'
  | 'train.stability'
  | 'train.optuna'
  | 'custom.page'
  | 'train.repro'
  | 'advanced.combine' // soups, ensembles, fine-tuned variations
  | 'advanced.autopilot'
  | 'scaling.page'
  | 'datasets.custom' // the Datasets tab, training on several datasets, the model view of the leaderboard
  | 'hub' // Hugging Face: import and publish
  | 'share' // the shared leaderboard and training on other computers
  | 'frontier' // the Frontier Lab pipeline

export interface Stats {
  runs: number
  widths: string[]
  lessons: string[]
  quizzes: string[]
  datasets: string[]
  best: Record<string, number> // best validation accuracy per dataset
  blocksTrained: string[]
  maxLayers: number
  maxEpochs: number
  flags: Record<string, boolean>
}

export interface Progress {
  version: 1
  mode: 'guided' | 'unlocked' | null
  xp: number
  avatar: string
  quests: string[]
  achievements: Record<string, number>
  skills: string[]
  stats: Stats
  levelSeen: number
  tourDone: boolean
  challenge: 'active' | 'done' | null
}

export const emptyStats = (): Stats => ({
  runs: 0, widths: [], lessons: [], quizzes: [], datasets: [], best: {}, blocksTrained: [], maxLayers: 0, maxEpochs: 0, flags: {},
})

export const newProgress = (): Progress => ({
  version: 1, mode: null, xp: 0, avatar: '🐣', quests: [], achievements: {}, skills: [], stats: emptyStats(),
  levelSeen: 1, tourDone: false, challenge: null,
})

// ---------------------------------------------------------------------------
// Levels
// ---------------------------------------------------------------------------
export interface Level {
  level: number
  xp: number
  title: string
  unlocks: Feature[]
  /** Plain-language description of what's new, shown on level-up. */
  news: string[]
  /** Skill-tree nodes granted for free at this level. */
  grants?: string[]
}

export const LEVELS: Level[] = [
  { level: 1, xp: 0, title: 'Spark', unlocks: ['lab'], news: ['The Lab: pick how big your model is and watch it learn.'] },
  { level: 2, xp: 200, title: 'Neuron', unlocks: ['lab.layers'], news: ['Stack more than one hidden layer: your first taste of "deep" learning.'] },
  {
    level: 3, xp: 450, title: 'Perceptron', unlocks: ['builder', 'builder.drag', 'builder.repeat'], grants: ['dropout'],
    news: ['The Builder replaces the Lab: a block panel on the left and drag-and-drop.', 'Repeat any block N times.', 'New block: Dropout.'],
  },
  {
    level: 4, xp: 750, title: 'Shape Shifter', unlocks: ['lesson.shapes'],
    news: ['Lesson: tensor shapes, and why blocks have to fit together.', 'Challenge: repair a broken model.'],
  },
  {
    level: 5, xp: 1100, title: 'Trainer', unlocks: ['train.page', 'train.epochs', 'train.lr', 'leaderboard.page'], grants: ['act_fns'],
    news: ['Training gets its own page.', 'Control epochs and the learning rate.', 'The full Leaderboard page.', 'Choose activation functions.'],
  },
  {
    level: 6, xp: 1500, title: 'Explorer', unlocks: ['skilltree', 'datasets'],
    news: ['The Skill Tree: spend skill points to unlock image and text blocks.', 'New datasets.'],
  },
  { level: 7, xp: 2000, title: 'Tinkerer', unlocks: ['train.optimizer', 'train.batch'], news: ['Pick the optimizer.', 'Batch size, plus a memory checker that tells you what fits.'] },
  {
    level: 8, xp: 2550, title: 'Archivist', unlocks: ['config.io', 'train.resume', 'builder.templates'],
    news: ['Save and load models as JSON.', 'Resume training from a checkpoint.', 'Ready-made templates for every dataset.'],
  },
  {
    level: 9, xp: 3150, title: 'Scheduler', unlocks: ['train.schedule', 'train.regularization', 'train.stability'],
    news: ['Learning-rate schedules and warm-up.', 'Regularization: weight decay, label smoothing, augmentation.', 'Gradient clipping and loss choice.'],
  },
  { level: 10, xp: 3800, title: 'Tuner', unlocks: ['train.optuna'], news: ['Optuna: let the computer search for the best hyperparameters.'] },
  { level: 11, xp: 4500, title: 'Maker', unlocks: ['custom.page'], news: ['Write your own blocks in Python.'] },
  { level: 12, xp: 5250, title: 'Researcher', unlocks: ['train.repro'], news: ['Seeds, devices and step caps: full control over reproducibility.'] },
  { level: 13, xp: 6100, title: 'Chef', unlocks: ['advanced.combine'], news: ['Combine trained models: soups (average the weights), ensembles and a mixture of experts.', 'Tick models on the Leaderboard, or open one and cook a soup from it.'] },
  { level: 14, xp: 7000, title: 'Pilot', unlocks: ['advanced.autopilot'], news: ['Autopilot: start from a trained model and let the computer keep improving it.'] },
  { level: 15, xp: 8000, title: 'Sage', unlocks: ['scaling.page'], news: ['The Scaling tab: train one design at several sizes and see if it follows a scaling law.'] },
  { level: 16, xp: 9100, title: 'Curator', unlocks: ['datasets.custom'], news: ['The Datasets tab: import your own tables, text, images or a Hugging Face dataset.', 'Train one model on several datasets at once.', 'Leaderboard by model: averages across datasets.'] },
  { level: 17, xp: 10300, title: 'Publisher', unlocks: ['hub'], news: ['Hugging Face: publish your models to the world, and import other people’s.'] },
  { level: 18, xp: 11600, title: 'Legend', unlocks: ['share'], news: ['The shared leaderboard: submit your best models to tracks with strict limits and see how they rank.', 'Train on another computer, such as a rented cloud GPU.'] },
  { level: 19, xp: 13000, title: 'Lab Director', unlocks: ['frontier'], news: ['The Frontier Lab: take a model through the stages a real AI lab uses. Pretrain it, improve it with reinforcement learning, teach it to refuse, and benchmark it after every step.'] },
]

export const MAX_LEVEL = LEVELS[LEVELS.length - 1].level

export function levelFor(xp: number): Level {
  let cur = LEVELS[0]
  for (const l of LEVELS) if (xp >= l.xp) cur = l
  return cur
}

export const ALL_FEATURES: Feature[] = LEVELS.flatMap((l) => l.unlocks)

/** Skill points: 2 when the tree opens at level 6, and 2 more every level after. */
export const skillPointsEarned = (level: number) => (level >= 6 ? 2 * (level - 5) : 0)

// ---------------------------------------------------------------------------
// Skill tree
// ---------------------------------------------------------------------------
export type Branch = 'general' | 'image' | 'text'

export interface SkillNode {
  id: string
  title: string
  branch: Branch
  blocks: string[]
  datasets?: string[]
  requires: string[]
  year: number
  lesson: string
  blurb: string
  col: number // layout column within the branch
  row: number
}

export const SKILLS: SkillNode[] = [
  // General
  { id: 'dropout', title: 'Dropout', branch: 'general', blocks: ['dropout'], requires: [], year: 2012, lesson: 'block.dropout', blurb: 'Randomly switch neurons off while training so the model can’t memorize.', col: 0, row: 0 },
  { id: 'act_fns', title: 'Activation functions', branch: 'general', blocks: ['activation', 'drawn_activation'], requires: [], year: 1986, lesson: 'block.activation', blurb: 'ReLU, GELU, tanh, sigmoid… choose the non-linearity, or draw your own.', col: 1, row: 0 },
  { id: 'mlp', title: 'MLP Block', branch: 'general', blocks: ['mlp'], requires: ['act_fns'], year: 1986, lesson: 'block.mlp', blurb: 'Two layers in one block, with an optional shortcut.', col: 1, row: 1 },
  { id: 'norms', title: 'Normalization', branch: 'general', blocks: ['batchnorm', 'layernorm'], requires: ['dropout'], year: 2015, lesson: 'block.batchnorm', blurb: 'Batch Norm & Layer Norm keep signals well-scaled.', col: 0, row: 1 },
  // Image
  { id: 'conv', title: 'Convolution', branch: 'image', blocks: ['conv2d'], datasets: ['cifar10'], requires: [], year: 1989, lesson: 'block.conv2d', blurb: 'Small filters that slide over the image. Unlocks CIFAR-10.', col: 0, row: 0 },
  { id: 'pooling', title: 'Pooling', branch: 'image', blocks: ['maxpool2d', 'avgpool2d', 'global_avgpool'], requires: ['conv'], year: 1989, lesson: 'block.pooling', blurb: 'Shrink feature maps, keep what matters.', col: 0, row: 1 },
  { id: 'resnet', title: 'Residual blocks', branch: 'image', blocks: ['resblock', 'bottleneck'], requires: ['pooling'], year: 2015, lesson: 'block.resblock', blurb: 'Skip connections make very deep CNNs trainable.', col: 0, row: 2 },
  { id: 'depthwise', title: 'Depthwise conv', branch: 'image', blocks: ['depthwise_conv'], requires: ['resnet'], year: 2017, lesson: 'block.depthwise_conv', blurb: 'MobileNet’s cheap convolutions.', col: 0, row: 3 },
  { id: 'vit', title: 'Vision Transformer', branch: 'image', blocks: ['patch_embed'], requires: ['resnet', 'transformer'], year: 2020, lesson: 'block.patch_embed', blurb: 'Cut images into patches and use a transformer.', col: 1, row: 3 },
  { id: 'mixer', title: 'MLP-Mixer', branch: 'image', blocks: ['mixer'], requires: ['vit'], year: 2021, lesson: 'block.mixer', blurb: 'Attention-free mixing with plain MLPs.', col: 1, row: 4 },
  // Text
  { id: 'embedding', title: 'Embeddings', branch: 'text', blocks: ['embedding', 'seq_pool'], datasets: ['ag_news', 'shakespeare'], requires: [], year: 2003, lesson: 'block.embedding', blurb: 'Turn words into vectors. Unlocks the text datasets.', col: 0, row: 0 },
  { id: 'rnn', title: 'Recurrent nets', branch: 'text', blocks: ['rnn'], requires: ['embedding'], year: 1990, lesson: 'block.rnn', blurb: 'Read a sequence one step at a time, with memory.', col: 0, row: 1 },
  { id: 'lstm', title: 'LSTM & GRU', branch: 'text', blocks: ['lstm', 'gru'], requires: ['rnn'], year: 1997, lesson: 'block.lstm', blurb: 'Gated memory that lasts much longer.', col: 0, row: 2 },
  { id: 'img_seq', title: 'Images as sequences', branch: 'text', blocks: ['image_to_sequence'], requires: ['rnn'], year: 1990, lesson: 'block.image_to_sequence', blurb: 'Let sequence models read images row by row.', col: 1, row: 2 },
  { id: 'attention', title: 'Attention', branch: 'text', blocks: ['self_attention'], requires: ['lstm'], year: 2014, lesson: 'block.self_attention', blurb: 'Every word looks at every other word.', col: 0, row: 3 },
  { id: 'transformer', title: 'Transformer', branch: 'text', blocks: ['transformer', 'positional_encoding'], requires: ['attention'], year: 2017, lesson: 'block.transformer', blurb: 'The architecture behind GPT and BERT.', col: 0, row: 4 },
]

export const BASE_BLOCKS = ['linear', 'flatten', 'classifier']
export const BASE_DATASETS = ['mnist']
export const DATASETS_AT_EXPLORER = ['mnist', 'fashion_mnist', 'spirals']

/** Skill nodes granted automatically by reaching a level. */
export const GRANTED_SKILLS = (level: number) => LEVELS.filter((l) => l.level <= level).flatMap((l) => l.grants ?? [])

// ---------------------------------------------------------------------------
// Avatars
// ---------------------------------------------------------------------------
export const AVATARS: { icon: string; level: number; name: string }[] = [
  { icon: '🐣', level: 1, name: 'Hatchling' },
  { icon: '🧱', level: 2, name: 'Brick' },
  { icon: '🔢', level: 3, name: 'Digits' },
  { icon: '🧩', level: 4, name: 'Puzzle' },
  { icon: '🧠', level: 5, name: 'Brain' },
  { icon: '🧭', level: 6, name: 'Compass' },
  { icon: '⚙️', level: 7, name: 'Gear' },
  { icon: '📦', level: 8, name: 'Archive' },
  { icon: '📈', level: 9, name: 'Chart' },
  { icon: '🎯', level: 10, name: 'Target' },
  { icon: '🛠️', level: 11, name: 'Tools' },
  { icon: '🔬', level: 12, name: 'Microscope' },
  { icon: '🤖', level: 13, name: 'Robot' },
  { icon: '🦉', level: 14, name: 'Owl' },
  { icon: '🧙', level: 15, name: 'Wizard' },
  { icon: '🚀', level: 16, name: 'Rocket' },
  { icon: '🐉', level: 17, name: 'Dragon' },
  { icon: '👑', level: 18, name: 'Crown' },
  { icon: '🏭', level: 19, name: 'Lab' },
]

// ---------------------------------------------------------------------------
// Quests: the "things to try" for each level. Completing them earns the XP for the next level.
// ---------------------------------------------------------------------------
export interface Go {
  tab?: string
  lesson?: string
  spot?: string // data-tour id to highlight
}

export interface Quest {
  id: string
  level: number
  title: string
  how: string
  xp: number
  done: (s: Stats, p: Progress) => boolean
  go?: Go
}

const f = (name: string) => (s: Stats) => !!s.flags[name]
const read = (id: string) => (s: Stats) => s.lessons.includes(id)
const best = (ds: string, x: number) => (s: Stats) => (s.best[ds] ?? 0) >= x

export const QUESTS: Quest[] = [
  // Level 1 - the Lab
  { id: 'q.mnist', level: 1, title: 'Meet MNIST', how: 'Read the short intro to the dataset of handwritten digits.', xp: 20, done: read('start.mnist'), go: { lesson: 'start.mnist' } },
  { id: 'q.data', level: 1, title: 'Look before you train', how: 'Open the Explore tab and browse some real examples from the dataset.', xp: 20, done: f('sawData'), go: { tab: 'explore' } },
  { id: 'q.first', level: 1, title: 'Train your first model', how: 'In the Lab, pick a size for the hidden layer. Training starts by itself.', xp: 50, done: (s) => s.runs >= 1, go: { tab: 'lab', spot: 'lab-widths' } },
  { id: 'q.widths', level: 1, title: 'Too small? Too big?', how: 'Try at least 3 different sizes and compare their accuracy.', xp: 40, done: (s) => s.widths.length >= 3, go: { tab: 'lab', spot: 'lab-widths' } },
  { id: 'q.90', level: 1, title: 'Read 9 out of 10 digits', how: 'Reach 90% validation accuracy on MNIST.', xp: 30, done: best('mnist', 0.9), go: { tab: 'lab', spot: 'lab-widths' } },
  // Level 2 - depth
  { id: 'q.mistakes', level: 2, title: 'Where does it go wrong?', how: 'In Explore → Test a model, look at your model’s most confident mistakes.', xp: 30, done: f('mistakes'), go: { tab: 'explore', spot: 'explore-tabs' } },
  { id: 'q.own', level: 2, title: 'Your own handwriting', how: 'In Explore → Test a model, draw a digit yourself and see if your model can read it.', xp: 40, done: f('ownInput'), go: { tab: 'explore', spot: 'explore-tabs' } },
  { id: 'q.stacking', level: 2, title: 'Learn about layers', how: 'Read “Layers & depth” in the guidebook.', xp: 20, done: read('start.layers'), go: { lesson: 'start.layers' } },
  { id: 'q.two', level: 2, title: 'Stack ’em', how: 'Train a model with 2 or more hidden layers.', xp: 40, done: (s) => s.maxLayers >= 2, go: { tab: 'lab', spot: 'lab-layers' } },
  { id: 'q.four', level: 2, title: 'Go deep', how: 'Train a model with 4 hidden layers.', xp: 40, done: (s) => s.maxLayers >= 4, go: { tab: 'lab', spot: 'lab-layers' } },
  { id: 'q.95', level: 2, title: 'Sharp eyes', how: 'Reach 95% validation accuracy on MNIST. (Hint: bigger layers help.)', xp: 50, done: best('mnist', 0.95), go: { tab: 'lab', spot: 'lab-widths' } },
  // Level 3 - the builder
  { id: 'q.builder', level: 3, title: 'Tour the Builder', how: 'Read the Builder guide.', xp: 20, done: read('app.builder'), go: { lesson: 'app.builder' } },
  { id: 'q.drag', level: 3, title: 'Drag in a block', how: 'Drag a block from the panel on the left onto your model.', xp: 40, done: f('dragBlock'), go: { tab: 'build', spot: 'palette' } },
  { id: 'q.move', level: 3, title: 'Rearrange', how: 'Drag a block in the stack to a new position.', xp: 40, done: f('moveBlock'), go: { tab: 'build', spot: 'stack' } },
  { id: 'q.repeat', level: 3, title: 'Repeat yourself', how: 'Click a block and use Repeat in the side panel to stack it 3 times.', xp: 30, done: f('repeat3'), go: { tab: 'build', spot: 'stack' } },
  { id: 'q.dropout', level: 3, title: 'Meet Dropout', how: 'Read the Dropout lesson.', xp: 20, done: read('block.dropout'), go: { lesson: 'block.dropout' } },
  { id: 'q.trainbuilt', level: 3, title: 'Your own design', how: 'Train a model you changed yourself in the Builder.', xp: 50, done: f('trainedBuilt'), go: { tab: 'build', spot: 'quick-train' } },
  // Level 4 - shapes
  { id: 'q.shapes', level: 4, title: 'Why shapes matter', how: 'Read “Tensors & shapes”.', xp: 30, done: read('start.shapes'), go: { lesson: 'start.shapes' } },
  { id: 'q.shapesquiz', level: 4, title: 'Shape quiz', how: 'Answer the quiz at the end of the shapes lesson.', xp: 30, done: (s) => s.quizzes.includes('start.shapes'), go: { lesson: 'start.shapes' } },
  { id: 'q.fix', level: 4, title: 'Repair job', how: 'Start the “broken model” challenge from the shapes lesson and fix it.', xp: 120, done: f('fixedShapes'), go: { lesson: 'start.shapes' } },
  { id: 'q.flatten', level: 4, title: 'Flat out', how: 'Read two lessons and press “Got it” on each: Flatten, then Output Head (the next lesson after Flatten).', xp: 30, done: (s) => s.lessons.includes('block.flatten') && s.lessons.includes('block.classifier'), go: { lesson: 'block.flatten' } },
  // Level 5 - training knobs
  { id: 'q.epochslesson', level: 5, title: 'Epochs explained', how: 'Read the Epochs lesson.', xp: 20, done: read('train.epochs'), go: { lesson: 'train.epochs' } },
  { id: 'q.lrlesson', level: 5, title: 'Learning rate explained', how: 'Read the Learning rate lesson.', xp: 20, done: read('train.lr'), go: { lesson: 'train.lr' } },
  { id: 'q.epochs5', level: 5, title: 'Practice makes perfect', how: 'On the Train page, train for 5 or more epochs.', xp: 40, done: (s) => s.maxEpochs >= 5, go: { tab: 'train', spot: 'hp-epochs' } },
  { id: 'q.lrhigh', level: 5, title: 'Too fast!', how: 'Train with a learning rate of 0.1 or more and see what happens.', xp: 40, done: f('lrHigh'), go: { tab: 'train', spot: 'hp-lr' } },
  { id: 'q.lrlow', level: 5, title: 'Too slow…', how: 'Train with a learning rate of 0.00001 or less.', xp: 40, done: f('lrLow'), go: { tab: 'train', spot: 'hp-lr' } },
  { id: 'q.lb', level: 5, title: 'Check the standings', how: 'Open the Leaderboard page.', xp: 20, done: f('visit:leaderboard'), go: { tab: 'leaderboard' } },
  { id: 'q.97', level: 5, title: 'Nearly there', how: 'Reach 97% validation accuracy on MNIST.', xp: 60, done: best('mnist', 0.97), go: { tab: 'train' } },
  { id: 'q.draw', level: 5, title: 'Draw your own', how: 'Add a Drawn Activation block after a Linear layer (set that layer’s activation to “none”), sketch a curve and train.', xp: 60, done: (s) => s.blocksTrained.includes('drawn_activation'), go: { tab: 'build', spot: 'palette' } },
  { id: 'q.actlesson', level: 5, title: 'Bend the line', how: 'Read the Activation functions lesson.', xp: 20, done: read('block.activation'), go: { lesson: 'block.activation' } },
  // Level 6 - explore
  { id: 'q.skill', level: 6, title: 'Choose a path', how: 'Open the Skill Tree in the guidebook and unlock a skill.', xp: 50, done: (_s, p) => p.skills.length >= 1, go: { tab: 'guide', spot: 'skilltree' } },
  { id: 'q.newblock', level: 6, title: 'New toys', how: 'Train a model that uses a block from the skill tree.', xp: 80, done: f('usedSkillBlock'), go: { tab: 'build', spot: 'palette' } },
  { id: 'q.dataset', level: 6, title: 'New data', how: 'Train on a dataset other than MNIST.', xp: 60, done: (s) => s.datasets.some((d) => d !== 'mnist'), go: { tab: 'build', spot: 'dataset' } },
  { id: 'q.skill2', level: 6, title: 'Branch out', how: 'Unlock a second skill.', xp: 50, done: (_s, p) => p.skills.length >= 2, go: { tab: 'guide', spot: 'skilltree' } },
  // Level 7 - optimizer & batch
  { id: 'q.optlesson', level: 7, title: 'Meet the optimizers', how: 'Read the Optimizers lesson.', xp: 20, done: read('train.optimizers'), go: { lesson: 'train.optimizers' } },
  { id: 'q.sgd', level: 7, title: 'Old school', how: 'Train with the SGD optimizer.', xp: 40, done: f('opt:sgd'), go: { tab: 'train', spot: 'hp-optimizer' } },
  { id: 'q.probe', level: 7, title: 'Will it fit?', how: 'Use “Check what fits in memory” on the Train page.', xp: 40, done: f('probe'), go: { tab: 'train', spot: 'hp-batch_size' } },
  { id: 'q.batch', level: 7, title: 'Bigger bites', how: 'Train with a batch size other than 128.', xp: 40, done: f('hp:batch_size'), go: { tab: 'train', spot: 'hp-batch_size' } },
  { id: 'q.batchlesson', level: 7, title: 'Batch size explained', how: 'Read the Batch size lesson.', xp: 20, done: read('train.batch'), go: { lesson: 'train.batch' } },
  // Level 8 - files
  { id: 'q.export', level: 8, title: 'Save your work', how: 'Export a model as JSON from the Builder (or download a run’s config).', xp: 40, done: f('export'), go: { tab: 'build', spot: 'export' } },
  { id: 'q.import', level: 8, title: 'Load a model', how: 'Import a JSON config into the Builder.', xp: 40, done: f('import'), go: { lesson: 'app.json' } },
  { id: 'q.resume', level: 8, title: 'Pick up where you left off', how: 'Resume training a run from its checkpoint.', xp: 60, done: f('resumed'), go: { lesson: 'app.resume' } },
  { id: 'q.repro', level: 8, title: 'Same again', how: 'Retrain a run from scratch on the Leaderboard and get exactly the same result.', xp: 80, done: f('reproduced'), go: { lesson: 'app.reproduce' } },
  { id: 'q.template', level: 8, title: 'Stand on shoulders', how: 'Load a template in the Builder.', xp: 20, done: f('template'), go: { tab: 'build', spot: 'templates' } },
  // Level 9 - schedules & regularization
  { id: 'q.schedlesson', level: 9, title: 'Schedules explained', how: 'Read the LR schedulers lesson.', xp: 20, done: read('train.schedulers'), go: { lesson: 'train.schedulers' } },
  { id: 'q.sched', level: 9, title: 'Change of pace', how: 'Train with a scheduler other than cosine.', xp: 40, done: f('hp:scheduler'), go: { tab: 'train', spot: 'hp-scheduler' } },
  { id: 'q.reglesson', level: 9, title: 'Fighting overfitting', how: 'Read the Regularization lesson.', xp: 20, done: read('train.regularization'), go: { lesson: 'train.regularization' } },
  { id: 'q.reg', level: 9, title: 'Hold back', how: 'Train with a different weight decay, dropout override or label smoothing.', xp: 40, done: (s) => !!(s.flags['hp:weight_decay'] || s.flags['hp:dropout'] || s.flags['hp:label_smoothing']), go: { tab: 'train' } },
  // Level 10 - Optuna
  { id: 'q.optunalesson', level: 10, title: 'Automatic tuning', how: 'Read the Optuna guide.', xp: 20, done: read('app.optuna'), go: { lesson: 'app.optuna' } },
  { id: 'q.optuna', level: 10, title: 'Let the computer search', how: 'Run an Optuna study to completion.', xp: 100, done: f('optuna'), go: { tab: 'train', spot: 'optuna' } },
  { id: 'q.applybest', level: 10, title: 'Take the advice', how: 'Click “Use best params” after a study.', xp: 40, done: f('applyBest'), go: { tab: 'train' } },
  // Level 11 - custom blocks
  { id: 'q.customlesson', level: 11, title: 'Read the custom block guide', how: 'Read “Your first custom block”.', xp: 20, done: read('app.custom'), go: { lesson: 'app.custom' } },
  { id: 'q.customnew', level: 11, title: 'Start a block', how: 'On the Custom Blocks page press “+ New”.', xp: 30, done: f('customNew'), go: { tab: 'custom', spot: 'custom-new' } },
  { id: 'q.customsave', level: 11, title: 'Make it real', how: 'Save your block with “Save & load”.', xp: 60, done: f('customSaved'), go: { tab: 'custom', spot: 'custom-save' } },
  { id: 'q.customuse', level: 11, title: 'Use it', how: 'Train a model that uses your custom block.', xp: 100, done: f('customUsed'), go: { tab: 'build', spot: 'palette' } },
  // Level 12 - reproducibility
  { id: 'q.seedlesson', level: 12, title: 'Seeds & devices', how: 'Read the Reproducibility lesson.', xp: 20, done: read('train.repro'), go: { lesson: 'train.repro' } },
  { id: 'q.seed', level: 12, title: 'Roll the dice', how: 'Train with a different random seed.', xp: 40, done: f('hp:seed'), go: { tab: 'train', spot: 'hp-seed' } },
  { id: 'q.seeds', level: 12, title: 'How sure are you?', how: 'Next to the Train button, set “seeds” to 3 and train. You get an average and a spread instead of one lucky number.', xp: 80, done: f('seeds'), go: { tab: 'train' } },
]

// ---------------------------------------------------------------------------
// Achievements
// ---------------------------------------------------------------------------
export interface Achievement {
  id: string
  icon: string
  title: string
  desc: string
  xp: number
  done: (s: Stats, p: Progress) => boolean
}

const trained = (t: string) => (s: Stats) => s.blocksTrained.includes(t)

export const ACHIEVEMENTS: Achievement[] = [
  { id: 'a.first', icon: '🎉', title: 'Hello, MNIST', desc: 'Train your first model.', xp: 50, done: (s) => s.runs >= 1 },
  { id: 'a.goldilocks', icon: '🐻', title: 'Goldilocks', desc: 'Try all five hidden-layer sizes in the Lab.', xp: 75, done: (s) => s.widths.length >= 5 },
  { id: 'a.95', icon: '👀', title: 'Sharp Eyes', desc: 'Get over 95% validation accuracy on MNIST.', xp: 75, done: best('mnist', 0.95) },
  { id: 'a.98', icon: '🏅', title: '98% Club', desc: 'Get over 98% validation accuracy on MNIST.', xp: 150, done: best('mnist', 0.98) },
  { id: 'a.99', icon: '🏆', title: 'Superhuman(ish)', desc: 'Get over 99% validation accuracy on MNIST.', xp: 250, done: best('mnist', 0.99) },
  { id: 'a.deep', icon: '🕳️', title: 'Going Deep', desc: 'Train a model with 4 or more hidden layers.', xp: 50, done: (s) => s.maxLayers >= 4 },
  { id: 'a.architect', icon: '📐', title: 'Architect', desc: 'Make your first custom model: train something you assembled yourself.', xp: 100, done: f('trainedBuilt') },
  { id: 'a.shapes', icon: '🧩', title: 'Shape Shifter', desc: 'Repair the model with mismatched tensor shapes.', xp: 100, done: f('fixedShapes') },
  { id: 'a.overshoot', icon: '🎢', title: 'Overshoot!', desc: 'Train with a learning rate of 0.1 or higher.', xp: 50, done: f('lrHigh') },
  { id: 'a.patience', icon: '⏳', title: 'Patience', desc: 'Train for 10 or more epochs.', xp: 75, done: (s) => s.maxEpochs >= 10 },
  { id: 'a.resume', icon: '⏯️', title: 'Comeback', desc: 'Resume the training of a model.', xp: 100, done: f('resumed') },
  { id: 'a.repro', icon: '🔁', title: 'Déjà Vu', desc: 'Retrain a run from scratch and reproduce it exactly.', xp: 150, done: f('reproduced') },
  { id: 'a.archivist', icon: '💾', title: 'Archivist', desc: 'Export or import a model as JSON.', xp: 50, done: (s) => !!(s.flags.export || s.flags.import) },
  { id: 'a.tinkerer', icon: '🔧', title: 'Tinkerer', desc: 'Save your first custom Python block.', xp: 150, done: f('customSaved') },
  { id: 'a.inventor', icon: '💡', title: 'Inventor', desc: 'Train a model that uses your own custom block.', xp: 150, done: f('customUsed') },
  { id: 'a.optuna', icon: '🎛️', title: 'Autopilot', desc: 'Complete an Optuna hyperparameter study.', xp: 150, done: f('optuna') },
  { id: 'a.batch', icon: '🧮', title: 'Batch Master', desc: 'Use the memory checker to find a batch size.', xp: 30, done: f('probe') },
  { id: 'a.explorer', icon: '🗺️', title: 'Explorer', desc: 'Train on three different datasets.', xp: 100, done: (s) => s.datasets.length >= 3 },
  { id: 'a.cifar', icon: '🌈', title: 'Seeing in Color', desc: 'Get over 70% validation accuracy on CIFAR-10.', xp: 150, done: best('cifar10', 0.7) },
  { id: 'a.news', icon: '📰', title: 'Headline Reader', desc: 'Get over 88% validation accuracy on AG News.', xp: 100, done: best('ag_news', 0.88) },
  { id: 'a.bard', icon: '🎭', title: 'The Bard', desc: 'Get over 50% next-character accuracy on Tiny Shakespeare.', xp: 150, done: best('shakespeare', 0.5) },
  { id: 'a.conv', icon: '🔍', title: 'Convolutional', desc: 'Train a model with a Conv2D block.', xp: 50, done: trained('conv2d') },
  { id: 'a.resnet', icon: '⤴️', title: 'Skip Ahead', desc: 'Train a model with a residual block.', xp: 50, done: trained('resblock') },
  { id: 'a.lstm', icon: '🧵', title: 'Long Memory', desc: 'Train a model with an LSTM or GRU.', xp: 50, done: (s) => s.blocksTrained.includes('lstm') || s.blocksTrained.includes('gru') },
  { id: 'a.attention', icon: '✨', title: 'Attention Is All You Need', desc: 'Train a model with a Transformer block.', xp: 75, done: trained('transformer') },
  { id: 'a.librarian', icon: '📚', title: 'Librarian', desc: 'Read 10 guidebook lessons.', xp: 75, done: (s) => s.lessons.length >= 10 },
  { id: 'a.scholar', icon: '🎓', title: 'Scholar', desc: 'Read 30 guidebook lessons.', xp: 200, done: (s) => s.lessons.length >= 30 },
  { id: 'a.quiz', icon: '❓', title: 'Quiz Whiz', desc: 'Answer 5 quizzes correctly.', xp: 75, done: (s) => s.quizzes.length >= 5 },
  { id: 'a.collector', icon: '🌳', title: 'Collector', desc: 'Unlock 8 skills in the skill tree.', xp: 100, done: (_s, p) => p.skills.length >= 8 },
  { id: 'a.detective', icon: '🕵️', title: 'Detective', desc: 'Inspect a model’s mistakes in Explore.', xp: 50, done: f('mistakes') },
  { id: 'a.handwriting', icon: '✍️', title: 'In Your Own Hand', desc: 'Test a model on something you drew, uploaded or typed yourself.', xp: 75, done: f('ownInput') },
  { id: 'a.chat', icon: '💬', title: 'Small Talk', desc: 'Chat with a text model you trained.', xp: 100, done: f('chat') },
  { id: 'a.seeds', icon: '🎲', title: 'Statistician', desc: 'Train a model with several seeds to measure how much luck is involved.', xp: 100, done: f('seeds') },
  { id: 'a.artist', icon: '🎨', title: 'Function Artist', desc: 'Train a model with an activation function you drew yourself.', xp: 75, done: (s) => s.blocksTrained.includes('drawn_activation') },
  { id: 'a.fuse', icon: '🧱', title: 'Master Builder', desc: 'Fuse several blocks into one new block of your own.', xp: 75, done: f('fuse') },
  { id: 'a.python', icon: '🐍', title: 'Leaving the Nest', desc: 'Export a model as standalone Python scripts.', xp: 75, done: f('exportPython') },
  { id: 'a.famous', icon: '🏛️', title: 'Standing on Shoulders', desc: 'Load a famous architecture from the Templates.', xp: 50, done: f('famous') },
  { id: 'a.soup', icon: '🍲', title: 'Soup Chef', desc: 'Average the weights of several models into one.', xp: 100, done: (s) => !!(s.flags.soup || s.flags.cook) },
  { id: 'a.ensemble', icon: '👥', title: 'Wisdom of the Crowd', desc: 'Combine models into an ensemble or a mixture of experts.', xp: 75, done: f('ensemble') },
  { id: 'a.finetune', icon: '🎯', title: 'Second Career', desc: 'Start a training run from another model’s trained weights.', xp: 75, done: f('finetune') },
  { id: 'a.monosoup', icon: '🥄', title: 'One Ingredient', desc: 'Make a MonoSoup from a single model.', xp: 75, done: f('monosoup') },
  { id: 'a.autopilot', icon: '🚀', title: 'Hands Off', desc: 'Let the autopilot improve a model for you.', xp: 100, done: f('autopilot') },
  { id: 'a.scaling', icon: '📈', title: 'Law Finder', desc: 'Finish a scaling study.', xp: 150, done: f('scaling') },
  { id: 'a.dataset', icon: '🗃️', title: 'Bring Your Own', desc: 'Import a dataset of your own.', xp: 100, done: f('importDataset') },
  { id: 'a.suite', icon: '🎒', title: 'Generalist', desc: 'Train one model on several datasets in one go.', xp: 100, done: f('suite') },
  { id: 'a.publish', icon: '🤗', title: 'Out in the World', desc: 'Publish a model to Hugging Face.', xp: 150, done: f('publish') },
  { id: 'a.hubimport', icon: '📥', title: 'Borrowed Brains', desc: 'Import a model from Hugging Face.', xp: 75, done: f('hubImport') },
  { id: 'a.submit', icon: '🌍', title: 'On the Board', desc: 'Submit a model to the shared leaderboard.', xp: 150, done: f('submit') },
  { id: 'a.worker', icon: '☁️', title: 'Head in the Cloud', desc: 'Train a model on another computer.', xp: 150, done: f('worker') },
  { id: 'a.feedback', icon: '💬', title: 'Playtester', desc: 'Send feedback about the app.', xp: 50, done: f('feedback') },
  { id: 'a.lab', icon: '🏭', title: 'Lab Director', desc: 'Take a model through the whole Frontier Lab pipeline.', xp: 250, done: f('lab') },
  { id: 'a.regular', icon: '📋', title: 'Regular', desc: 'Complete 10 training runs.', xp: 75, done: (s) => s.runs >= 10 },
]

/** Which feature reveals each training hyperparameter. Hidden ones use beginner defaults. */
export const HP_FEATURE: Record<string, Feature> = {
  epochs: 'train.epochs', lr: 'train.lr', optimizer: 'train.optimizer', momentum: 'train.optimizer', batch_size: 'train.batch',
  weight_decay: 'train.regularization', dropout: 'train.regularization', label_smoothing: 'train.regularization', augment: 'train.regularization',
  scheduler: 'train.schedule', warmup_epochs: 'train.schedule', step_size: 'train.schedule', gamma: 'train.schedule',
  grad_clip: 'train.stability', loss: 'train.stability', seed: 'train.repro', device: 'train.repro', max_steps: 'train.repro',
}

/** Beginner values for hidden hyperparameters (otherwise the server defaults apply). */
export const BEGINNER_TRAINING: Record<string, unknown> = { epochs: 1, device: 'cpu', seed: 42, max_steps: null }
