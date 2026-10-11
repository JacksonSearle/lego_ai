import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { api, streamJob } from './api'
import { BEGINNER_TRAINING, HP_FEATURE } from './learn/curriculum'
import type { TrainInfo } from './learn/engine'
import { useLearn } from './learn/LearnContext'
import type {
  BlockInst, BlockSpec, Combined, DatasetInfo, EpochPoint, InspectResult, JobEvent, Meta, RunConfig, Stat, StepPoint, Summary, Training,
} from './types'

export interface Draft {
  name: string
  dataset: string
  train_subset: number | null
  /** Text datasets: tokens per example (null = the dataset's default). */
  seq_len?: number | null
  blocks: BlockInst[]
  training: Training
  custom_blocks?: RunConfig['custom_blocks']
  /** Fine-tuning: start from this trained run's weights instead of random ones. */
  init_from?: RunConfig['init_from']
  /** Set when an imported run used the original weight-initialization scheme, so retraining it matches. */
  init_stream?: number
  /** Where the architecture came from, and whether the user has changed it since (for quests). */
  origin?: 'lab' | 'template' | 'challenge' | 'import' | 'scratch'
  edited?: boolean
}

/** Extra facts about a training job that the learning engine needs when it finishes. */
export interface LearnMeta {
  built?: boolean
  width?: string
  resumed?: boolean
}

export interface TrialRow {
  trial: number
  run_id: string
  params: Record<string, unknown>
  value: number | null
  state: string
}

/** One finished size of a scaling study. */
export interface ScalePoint {
  index: number; factor: number; run_id: string; params?: number; n_train?: number
  best_val_acc?: number; best_val_loss?: number; test_acc?: number; test_loss?: number; duration?: number; error?: string
}

export interface SeedRow { index: number; seed: number; run_id: string; best_val_acc: number | null; test_acc: number | null }

export interface Session {
  jobId: string
  /** autopilot behaves like tune (rounds = trials); cook like seeds (variations = seeds). */
  kind: 'train' | 'tune' | 'seeds' | 'autopilot' | 'cook' | 'scale'
  status: string
  runId: string | null
  name: string
  message: string
  params: number | null
  epochs: number | null
  totalSteps: number | null
  /** How many training examples the run uses (after any subset), and the step it (re)started at. */
  trainExamples: number | null
  baseStep: number
  device: string | null
  steps: StepPoint[]
  epochPts: EpochPoint[]
  summary: Summary | null
  error: string | null
  /** Soups and ensembles made by this job. */
  combined: Combined[]
  /** Scaling studies: which study, and the sizes finished so far. */
  scale: { id: string; axis: 'width' | 'data'; n: number } | null
  points: ScalePoint[]
  trials: TrialRow[]
  currentTrial: number | null
  best: { trial: number; value: number; params: Record<string, unknown>; run_id: string } | null
  metric?: string
  /** 'seeds' jobs: one row per finished seed, and the group's mean/spread at the end. */
  seedRuns: SeedRow[]
  seedTotal: number | null
  seedStats: Record<string, Stat> | null
  startedAt: number
  learn?: LearnMeta
  reported?: boolean
}

interface Store {
  meta: Meta | null
  blocks: Record<string, BlockSpec>
  datasets: DatasetInfo[]
  dataset: DatasetInfo | undefined
  draft: Draft
  setDraft: (fn: (d: Draft) => Draft, opts?: { undoable?: boolean }) => void
  undo: () => void
  redo: () => void
  canUndo: boolean
  canRedo: boolean
  inspect: InspectResult | null
  /** True while the backend can't be reached (shape checks and training are unavailable). */
  offline: boolean
  refreshBlocks: () => Promise<void>
  session: Session | null
  attachJob: (jobId: string, kind: Session['kind'], name: string, runId?: string | null, learn?: LearnMeta) => void
  clearSession: () => void
  toConfig: () => RunConfig
  effectiveTraining: (t: Training) => Training
  loadConfig: (cfg: RunConfig) => void
  tab: string
  setTab: (t: string) => void
  toast: (msg: string, kind?: 'info' | 'error' | 'ok') => void
  toasts: { id: number; msg: string; kind: string }[]
}

const Ctx = createContext<Store>(null as unknown as Store)
export const useStore = () => useContext(Ctx)

const DRAFT_KEY = 'lego-ai/draft'
let idCounter = Date.now() % 100000

export function newId() {
  idCounter += 1
  return `b${idCounter.toString(36)}`
}

function loadDraft(): Draft | null {
  try {
    const raw = localStorage.getItem(DRAFT_KEY)
    return raw ? (JSON.parse(raw) as Draft) : null
  } catch {
    return null
  }
}

function emptySession(jobId: string, kind: Session['kind'], name: string, runId: string | null): Session {
  return {
    jobId, kind, name, runId, status: 'running', message: 'Starting…', params: null, epochs: null, totalSteps: null, trainExamples: null, baseStep: 0, device: null,
    steps: [], epochPts: [], summary: null, error: null, trials: [], currentTrial: null, best: null, seedRuns: [], seedTotal: null, seedStats: null, combined: [], scale: null, points: [], startedAt: Date.now(),
  }
}

/** Fold one job event into the session state. */
function reduce(s: Session, e: JobEvent): Session {
  switch (e.type) {
    case 'status':
      return { ...s, message: e.message as string }
    case 'start': {
      const hist = e.history as { steps: StepPoint[]; epochs: EpochPoint[] } | null
      return {
        ...s, runId: (e.run_id as string) ?? s.runId, params: e.params as number, epochs: e.epochs as number,
        totalSteps: e.total_steps as number, device: e.device as string, message: 'Training…',
        trainExamples: (e.train_examples as number) ?? null,
        baseStep: hist?.steps.length ? hist.steps[hist.steps.length - 1].step : 0,
        steps: hist?.steps ?? [], epochPts: hist?.epochs ?? [], summary: null,
      }
    }
    case 'step':
      return { ...s, steps: [...s.steps, e as unknown as StepPoint] }
    case 'epoch':
      return { ...s, epochPts: [...s.epochPts, e as unknown as EpochPoint] }
    case 'done':
      return s.kind !== 'train' ? s : { ...s, summary: e.summary as Summary, status: e.status as string, message: e.status === 'stopped' ? 'Paused' : `Run ${e.status}` }
    case 'error':
      return { ...s, error: e.message as string }
    case 'study_start':
      return { ...s, metric: (e.study as { metric: string }).metric, message: 'Study started' }
    case 'trial_start':
      return {
        ...s, currentTrial: e.trial as number, runId: e.run_id as string, steps: [], epochPts: [],
        message: `Trial ${e.trial}`,
      }
    case 'trial_end': {
      const row = e as unknown as TrialRow & { best: Session['best'] }
      return { ...s, trials: [...s.trials.filter((t) => t.trial !== row.trial), row], best: row.best }
    }
    case 'seed_start':
      return {
        ...s, currentTrial: (e.index as number) + 1, seedTotal: e.n as number, runId: e.run_id as string, steps: [], epochPts: [],
        message: `Seed ${e.seed} (${(e.index as number) + 1} of ${e.n})`,
      }
    case 'seed_end':
      return { ...s, seedRuns: [...s.seedRuns, e as unknown as SeedRow] }
    case 'combined':
      return { ...s, combined: [...s.combined, e as unknown as Combined] }
    case 'scale_start':
      return { ...s, scale: { id: e.id as string, axis: e.axis as 'width' | 'data', n: (e.points as unknown[]).length }, seedTotal: (e.points as unknown[]).length }
    case 'point_start':
      return { ...s, currentTrial: (e.index as number) + 1, runId: e.run_id as string, steps: [], epochPts: [], message: `${e.label} (${(e.index as number) + 1} of ${e.n})` }
    case 'point_end':
      return { ...s, points: [...s.points, e as unknown as ScalePoint] }
    case 'seeds_end':
      return { ...s, seedStats: e.stats as Record<string, Stat> }
    case 'job_end':
      return { ...s, status: e.status as string, error: (e.error as string) ?? s.error, message: s.kind === 'train' && e.status === 'stopped' ? 'Paused' : `Job ${e.status}` }
    default:
      return s
  }
}

export function StoreProvider({ children }: { children: ReactNode }) {
  const learn = useLearn()
  const [meta, setMeta] = useState<Meta | null>(null)
  const [blockList, setBlockList] = useState<BlockSpec[]>([])
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  // Undo history kept in one state object so every updater stays pure.
  const [hist, setHist] = useState<{ past: Draft[]; present: Draft; future: Draft[] }>(() => ({
    past: [],
    present: loadDraft() ?? { name: 'My model', dataset: 'mnist', train_subset: null, blocks: [], training: {} },
    future: [],
  }))
  const draft = hist.present
  const [inspect, setInspect] = useState<InspectResult | null>(null)
  const [offline, setOffline] = useState(false)
  const [retry, setRetry] = useState(0)
  const [session, setSession] = useState<Session | null>(null)
  const [tab, setTab] = useState(() => location.hash.slice(1) || 'build')
  const [toasts, setToasts] = useState<{ id: number; msg: string; kind: string }[]>([])
  const unsub = useRef<(() => void) | null>(null)

  const toast = useCallback((msg: string, kind: 'info' | 'error' | 'ok' = 'info') => {
    const id = Math.random()
    setToasts((t) => [...t, { id, msg, kind }])
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), kind === 'error' ? 7000 : 3500)
  }, [])

  const refreshBlocks = useCallback(async () => setBlockList(await api.blocks()), [])

  useEffect(() => {
    api.meta().then((m) => {
      setMeta(m)
      // Fill in any training defaults missing from a stored draft.
      setHist((h) => ({ ...h, present: { ...h.present, training: { ...m.defaults, ...h.present.training } } }))
    })
    refreshBlocks()
    api.datasets().then(setDatasets)
  }, [refreshBlocks])

  useEffect(() => {
    location.hash = tab
  }, [tab])

  useEffect(() => {
    localStorage.setItem(DRAFT_KEY, JSON.stringify(draft))
  }, [draft])

  // Debounced shape inference whenever the architecture or dataset changes.
  const archKey = JSON.stringify([draft.dataset, draft.seq_len ?? null, draft.blocks, blockList.length])
  useEffect(() => {
    const h = setTimeout(() => {
      api
        .inspect(draft.dataset, { blocks: draft.blocks }, draft.seq_len)
        .then((r) => { setInspect(r); setOffline(false) })
        .catch((e) => {
          setInspect(null)
          // fetch() rejects with a TypeError when the server isn't there at all.
          if (e instanceof TypeError) setOffline(true)
        })
    }, 200)
    return () => clearTimeout(h)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [archKey, retry])

  // While offline, check every few seconds and re-run the shape check once the server is back.
  useEffect(() => {
    if (!offline) return
    const h = setInterval(() => {
      fetch('/api/meta').then((r) => { if (r.ok) setRetry((n) => n + 1) }).catch(() => {})
    }, 3000)
    return () => clearInterval(h)
  }, [offline])

  const setDraft = useCallback((fn: (d: Draft) => Draft, opts?: { undoable?: boolean }) => {
    setHist((h) => {
      const next = fn(h.present)
      if (next === h.present) return h
      if (opts?.undoable === false) return { ...h, present: next }
      return { past: [...h.past.slice(-80), h.present], present: next, future: [] }
    })
  }, [])

  const undo = useCallback(() => {
    setHist((h) => (h.past.length ? { past: h.past.slice(0, -1), present: h.past[h.past.length - 1], future: [h.present, ...h.future] } : h))
  }, [])

  const redo = useCallback(() => {
    setHist((h) => (h.future.length ? { past: [...h.past, h.present], present: h.future[0], future: h.future.slice(1) } : h))
  }, [])

  const attachJob = useCallback((jobId: string, kind: Session['kind'], name: string, runId: string | null = null, learn?: LearnMeta) => {
    unsub.current?.()
    setSession({ ...emptySession(jobId, kind, name, runId), learn })
    unsub.current = streamJob(
      jobId,
      (e) => setSession((s) => (s && s.jobId === jobId ? reduce(s, e) : s)),
      // The stream closed. If the final event never arrived, take the job's status from the server
      // so the page can't be left showing "running" for something that has finished.
      async () => {
        const all = await api.allJobs().catch(() => [])
        const j = all.find((x) => x.id === jobId)
        setSession((s) => (s && s.jobId === jobId && s.status === 'running'
          ? { ...s, status: j && j.status !== 'running' ? j.status : 'completed', error: j?.error ?? s.error, message: 'Finished' }
          : s))
      },
    )
  }, [])

  const clearSession = useCallback(() => {
    unsub.current?.()
    setSession(null)
  }, [])

  // Re-attach to a training/tuning job that is still running (e.g. after a page reload).
  useEffect(() => {
    api.jobs().then((jobs) => {
      const j = jobs.find((x) => ['train', 'tune', 'seeds', 'autopilot', 'cook', 'scale'].includes(x.kind))
      if (j) attachJob(j.id, j.kind as Session['kind'], String(j.meta.name ?? 'Run'), (j.meta.run_id as string) ?? null)
    })
  }, [attachJob])

  // Hyperparameters the learner hasn't unlocked yet use beginner defaults, whatever the draft says.
  const learnAccess = learn.access
  const effectiveTraining = useCallback(
    (t: Training): Training => {
      if (learnAccess.unlocked || !meta) return t
      const out: Training = { ...t }
      for (const [name, feature] of Object.entries(HP_FEATURE)) {
        if (!learnAccess.has(feature)) out[name] = name in BEGINNER_TRAINING ? BEGINNER_TRAINING[name] : meta.defaults[name]
      }
      return out
    },
    [learnAccess, meta],
  )

  const toConfig = useCallback(
    (): RunConfig => ({
      name: draft.name,
      dataset: { id: draft.dataset, train_subset: draft.train_subset, seq_len: draft.seq_len ?? null },
      architecture: { blocks: draft.blocks },
      training: effectiveTraining(draft.training),
      custom_blocks: draft.custom_blocks,
      ...(draft.origin === 'import' && draft.init_stream ? { init_stream: draft.init_stream } : {}),
      ...(draft.init_from ? { init_from: draft.init_from } : {}),
    }),
    [draft, effectiveTraining],
  )

  // Report finished jobs to the learning engine (XP, quests, achievements).
  const { track, flag } = learn
  useEffect(() => {
    if (!session || session.reported || session.status === 'running') return
    setSession((s) => (s ? { ...s, reported: true } : s))
    if (session.kind === 'tune') {
      if (session.status === 'completed' && session.trials.length) flag('optuna')
      return
    }
    if (session.kind === 'seeds') {
      if (session.status === 'completed') flag('seeds')
      return
    }
    if (session.kind !== 'train') {
      if (session.status === 'completed') flag(session.kind === 'scale' ? 'scaling' : session.kind)
      return
    }
    if (session.status !== 'completed' || !session.runId) return
    api.run(session.runId).then((run) => {
      const t = run.config.training
      const defaults = meta?.defaults ?? {}
      const info: TrainInfo = {
        dataset: run.config.dataset.id,
        valAcc: run.summary.best_val_acc ?? 0,
        blockTypes: Array.from(new Set(run.config.architecture.blocks.map((b) => b.type))),
        epochs: run.summary.epochs_done ?? 0,
        layers: run.config.architecture.blocks.filter((b) => b.type === 'linear' || b.type === 'mlp').reduce((a, b) => a + (b.repeat || 1), 0),
        width: session.learn?.width,
        changed: Object.keys(t).filter((k) => t[k] !== undefined && defaults[k] !== undefined && JSON.stringify(t[k]) !== JSON.stringify(defaults[k])),
        optimizer: String(t.optimizer),
        lr: Number(t.lr),
        built: !!session.learn?.built,
        reproduced: run.summary.reproduced === true,
        resumed: !!session.learn?.resumed,
      }
      track({ type: 'train_done', info })
    })
  }, [session, meta, track, flag])

  const loadConfig = useCallback(
    (cfg: RunConfig) => {
      setDraft(() => ({
        name: cfg.name,
        dataset: cfg.dataset.id,
        train_subset: cfg.dataset.train_subset ?? null,
        seq_len: cfg.dataset.seq_len ?? null,
        blocks: cfg.architecture.blocks.map((b) => ({ ...b, id: b.id || newId(), repeat: b.repeat || 1 })),
        training: { ...(meta?.defaults ?? {}), ...cfg.training },
        custom_blocks: cfg.custom_blocks,
        init_stream: cfg.init_stream ?? (cfg.environment ? 1 : undefined),
        init_from: cfg.init_from,
        origin: 'import',
        edited: false,
      }))
    },
    [meta, setDraft],
  )

  const blocks = useMemo(() => Object.fromEntries(blockList.map((b) => [b.type, b])), [blockList])
  const dataset = datasets.find((d) => d.id === draft.dataset)

  const value: Store = {
    meta, blocks, datasets, dataset, draft, setDraft, undo, redo, canUndo: hist.past.length > 0, canRedo: hist.future.length > 0,
    inspect, offline, refreshBlocks, session, attachJob, clearSession, toConfig, effectiveTraining, loadConfig, tab, setTab, toast, toasts,
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
