export type ParamType = 'int' | 'float' | 'bool' | 'choice' | 'activation' | 'str'

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
}

export interface RunConfig {
  format?: string
  version?: number
  name: string
  dataset: { id: string; train_subset: number | null }
  architecture: { blocks: BlockInst[] }
  training: Training
  custom_blocks?: Record<string, { source: string; sha256: string }>
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
  training: Training
  n_blocks: number
  block_types: string[]
  has_checkpoint: boolean
}

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

export interface HistoryStage {
  id: string
  year: number
  title: string
  era: string
  blocks: BlockInst[]
  training: Training
  blurb: string
  added: string
  paper: { title: string; url: string; authors: string }
  run: { id: string; status: string; summary: Summary } | null
  metrics: Metrics | null
}

export interface Prediction {
  image?: number[]
  shape?: number[]
  text?: string
  label: number
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
