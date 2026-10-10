export type ParamType = 'int' | 'float' | 'bool' | 'choice' | 'activation' | 'str' | 'curve'

export interface ParamSpec {
  name: string
  type: ParamType
  default: unknown
  label: string
  help: string
  min?: number | null
  max?: number | null
  step?: number | null
  choices?: unknown[] | null
  /** 'curve' params: the fixed x positions and the allowed y range. */
  xs?: number[]
  y_limit?: number
}

export interface BlockSpec {
  type: string
  name: string
  category: string
  color: string
  description: string
  params: ParamSpec[]
  expects: string
  paper: string | null
  source: 'builtin' | 'custom'
  source_file: string | null
  tags: string[]
  year: number | null
  /** Fused blocks: the blocks this one was made from. */
  parts?: Omit<BlockInst, 'id'>[] | null
}

export interface BlockInst {
  id: string
  type: string
  params: Record<string, unknown>
  repeat: number
}

export interface DatasetInfo {
  id: string
  name: string
  description: string
  task: 'classification' | 'lm'
  input_kind: 'image' | 'tokens' | 'vector'
  input_shape: number[]
  num_classes: number
  vocab_size: number | null
  seq_len: number | null
  class_names: string[]
  source: string
  train_size: number | null
  test_size: number | null
  augment: boolean
  cached: boolean
  /** Text datasets: allowed [min, max] sequence length. */
  seq_len_range: [number, number] | null
  /** 1 easy, 2 medium, 3 hard */
  difficulty: number
  difficulty_note: string
  targets: { baseline: number; good: number; great: number } | null
}

export interface HyperParam {
  name: string
  label: string
  type: 'int' | 'float' | 'bool' | 'choice'
  default: unknown
  min?: number
  max?: number
  choices?: unknown[]
  nullable?: boolean
  group: string
  help: string
  show_if?: Record<string, unknown[]>
  tune?: SearchSpec
}

export interface SearchSpec {
  type: 'float' | 'int' | 'categorical'
  low?: number
  high?: number
  log?: boolean
  choices?: unknown[]
  enabled?: boolean
}

export interface Meta {
  environment: Record<string, unknown> & { available_devices: string[]; torch: string; python: string }
  hyperparams: HyperParam[]
  defaults: Training
  categories: string[]
  search_space: Record<string, SearchSpec>
  custom_errors: Record<string, string>
}

export type Training = Record<string, unknown>

export interface Template {
  id: string
  name: string
  description: string
  blocks: BlockInst[]
  training: Training
  recommended: boolean
  /** Recreations of published architectures. */
  famous?: { year: number; authors: string; paper: string }
  params?: number
}

export interface BlockReport {
  id: string
  type: string
  in_shape: number[] | null
  out_shape: number[] | null
  params: number
  error: string | null
}

export interface InspectResult {
  ok: boolean
  blocks: BlockReport[]
  input_shape: number[]
  output_shape: number[] | null
  expected_output: number[]
  total_params: number
  error: string | null
  /** Per training example: bytes of activations, and multiply-adds for one forward pass. */
  act_bytes_per_sample?: number
  flops_per_sample?: number
}

export interface RunConfig {
  format?: string
  version?: number
  name: string
  dataset: { id: string; train_subset: number | null; seq_len?: number | null }
  architecture: { blocks: BlockInst[] }
  training: Training
  custom_blocks?: Record<string, { source: string; sha256: string }>
  /** Weight-initialization scheme (absent = 1, the original). */
  init_stream?: number
  environment?: Record<string, unknown>
}

export interface Summary {
  params?: number
  epochs_done?: number
  steps?: number
  best_val_acc?: number
  best_val_loss?: number
  best_epoch?: number
  final_val_acc?: number
  final_val_loss?: number
  final_train_loss?: number
  test_acc?: number
  test_loss?: number
  duration?: number
  reproduced?: boolean
  reproduce_diff?: Record<string, [number, number]>
  steps_per_epoch?: number
  train_examples?: number
  /** Set while a run is paused mid-training: the (fractional) epoch it stopped at. */
  paused_at?: number
}

export interface StepPoint { step: number; epoch: number; loss: number; acc: number; lr: number }
export interface EpochPoint {
  epoch: number; step: number; train_loss: number; train_acc: number; val_loss: number; val_acc: number; lr: number; seconds: number
  data_seconds?: number; compute_seconds?: number; eval_seconds?: number
}
export interface Metrics { steps: StepPoint[]; epochs: EpochPoint[] }

export interface RunRow {
  id: string
  name: string
  status: string
  created_at: number
  updated_at: number
  dataset: string
  summary: Summary
  error: string | null
  study: { id: string; trial: number; params: Record<string, unknown>; best: boolean } | null
  reproduce_of: string | null
  group: SeedGroup | null
  training: Training
  n_blocks: number
  block_types: string[]
  has_checkpoint: boolean
}

export interface Stat { mean: number | null; std: number | null; min: number | null; max: number | null; n: number }

/** Several runs of one config that differ only in their random seed. */
export interface SeedGroup {
  id: string
  index: number
  n: number
  seeds: number[]
  name: string
  completed?: number
  stats?: Record<string, Stat>
}

export interface Hardware {
  cpu: string
  cores: number
  threads: number
  ram_gb: number
  ram_free_gb: number
  swap_used_gb: number
  gpus: { device: string; name: string; memory_gb: number; free_gb: number; shared: boolean }[]
  best_device: string
  notes: string[]
}

/** How a model fits on one device: memory at the chosen batch size, and rough time per epoch. */
export interface DeviceFit { fits: boolean; need_gb: number; max_batch: number; epoch_seconds: number | null }

/** Memory and time needed to train a model here and on well-known GPUs. */
export interface Estimate {
  verdict: 'ok' | 'long' | 'slow' | 'too_big'
  headline: string
  need_gb: number
  fixed_gb: number
  per_sample_mb: number
  batch_size: number
  epochs: number
  here: (DeviceFit & { device: string; label: string; capacity_gb: number; selected: boolean })[]
  gpus: (DeviceFit & { name: string; memory_gb: number; note: string; count: number | null })[]
  faster_here: string | null
}

export interface ProfileInfo { id: string; name: string; active: boolean; mode: string | null; xp: number; avatar: string; demo: boolean }

export interface RunDetail {
  id: string
  name: string
  status: string
  created_at: number
  config: RunConfig
  summary: Summary
  error: string | null
  metrics: Metrics
  has_checkpoint: boolean
  reproduce_of?: string | null
  study?: RunRow['study']
  group?: SeedGroup | null
}

export interface Job {
  id: string
  kind: string
  status: string
  error: string | null
  meta: Record<string, unknown>
}

export interface JobEvent {
  seq: number
  t: number
  type: string
  [k: string]: unknown
}

export interface ProbeRow {
  batch_size: number
  estimated_mb: number
  measured_mb?: number
  fits: boolean | null
  tested: boolean
  reason?: string
  step_ms?: number
  samples_per_s?: number
  epoch_s?: number
}

export interface ProbeResult {
  device: string
  budget_mb: number
  budget_label: string
  params: number
  fixed_mb: number
  activation_kb_per_sample: number
  results: ProbeRow[]
  largest_fitting: number | null
  fastest: number | null
  requested: number | null
  requested_fits: boolean | null
  train_examples?: number
  /** Why some sizes could not be tested (time limit, test process failure). */
  note?: string | null
}

export interface Prediction {
  image?: number[]
  shape?: number[]
  text?: string
  vector?: number[]
  index?: number
  label: number | string
  pred: number
  top: { cls: number; p: number }[]
}

export interface Predictions {
  kind: string
  class_names?: string[]
  examples?: Prediction[]
  prompt?: string
  generated?: string
}

/** Random examples from a dataset (no model involved). */
export interface Samples {
  dataset: string
  kind: 'image' | 'tokens' | 'vector'
  class_names: string[]
  examples: Omit<Prediction, 'pred' | 'top'>[]
  total: number
  class_counts: number[] | null
  split: string
}

/** A model scored over a split: where it is right, and where and how it is wrong. */
export interface Evaluation {
  kind: string
  class_names: string[]
  split: string
  n: number
  total: number
  accuracy: number
  wrong_total: number
  confusion: number[][]
  per_class: { cls: number; n: number; acc: number | null }[]
  examples: Prediction[]
  selected: number
  unsure: Prediction[]
}

export interface InputPrediction {
  kind: string
  class_names: string[]
  input: { image?: number[]; shape?: number[]; text?: string; unknown?: string[]; truncated?: boolean; vector?: number[] }
  pred: number
  probs: number[]
}

export interface Generation { text: string; seq_len: number; context: string; dropped: string[] }
