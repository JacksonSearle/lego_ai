import type {
  DatasetPreview, HubModel, HubStatus, Predictions, Samples, Evaluation, InputPrediction, Generation, BlockInst, BlockSpec, DatasetInfo, Estimate, Hardware, InspectResult, ProfileInfo, Job, JobEvent, Meta, ProbeResult, RunConfig, RunDetail, RunRow, ScalePlanRow, ShiftInfo, ShiftScores, Template,
} from './types'

async function req<T>(method: string, url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  const text = await res.text()
  const data = text ? JSON.parse(text) : null
  if (!res.ok) {
    const msg = data?.detail ?? data?.error ?? res.statusText
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
  }
  return data as T
}

export const api = {
  meta: () => req<Meta>('GET', '/api/meta'),
  blocks: () => req<BlockSpec[]>('GET', '/api/blocks'),
  datasets: () => req<DatasetInfo[]>('GET', '/api/datasets'),
  templates: (dataset: string) => req<Template[]>('GET', `/api/templates/${dataset}`),
  inspect: (dataset: string, architecture: RunConfig['architecture'], seq_len?: number | null) =>
    req<InspectResult>('POST', '/api/inspect', { dataset, architecture, seq_len }),
  hardware: () => req<Hardware>('GET', '/api/hardware'),
  estimate: (body: { dataset: string; seq_len?: number | null; train_subset?: number | null; params: number; act_bytes: number; flops: number; batch_size: number; optimizer: string; epochs: number; device: string }) =>
    req<Estimate>('POST', '/api/estimate', body),
  fuse: (name: string, color: string, blocks: BlockInst[]) =>
    req<{ ok: boolean; block: BlockSpec }>('POST', '/api/custom-blocks/fuse', { name, color, blocks }),
  trainSeeds: (config: RunConfig, n: number) => req<{ job: Job }>('POST', '/api/train-seeds', { config, n }),
  profiles: () => req<{ active: string; profiles: ProfileInfo[] }>('GET', '/api/profiles'),
  createProfile: (name: string, opts?: { demo?: boolean; progress?: unknown }) =>
    req<{ id: string; active: string; profiles: ProfileInfo[] }>('POST', '/api/profiles', { name, ...opts }),
  activateProfile: (id: string) => req('POST', `/api/profiles/${id}/activate`),
  renameProfile: (id: string, name: string) => req('PATCH', `/api/profiles/${id}`, { name }),
  deleteProfile: (id: string) => req('DELETE', `/api/profiles/${id}`),

  customBlocks: () => req<{ files: { filename: string; source: string; blocks: string[] }[]; example: string }>('GET', '/api/custom-blocks'),
  validateCustom: (filename: string, source: string) =>
    req<{ ok: boolean; error?: string; blocks?: BlockSpec[] }>('POST', '/api/custom-blocks/validate', { filename, source }),
  saveCustom: (filename: string, source: string) =>
    req<{ ok: boolean; error?: string; blocks?: BlockSpec[] }>('PUT', `/api/custom-blocks/${encodeURIComponent(filename)}`, { source }),
  deleteCustom: (filename: string) => req('DELETE', `/api/custom-blocks/${encodeURIComponent(filename)}`),

  train: (config: RunConfig, reproduce_of?: string) => req<{ job: Job; run: RunDetail }>('POST', '/api/train', { config, reproduce_of }),
  resume: (runId: string, epochs: number) => req<{ job: Job; run: RunDetail }>('POST', `/api/runs/${runId}/resume`, { epochs }),
  retrain: (runId: string) =>
    req<{ job: Job; run: RunDetail; notes: string[]; env_diffs: string[] }>('POST', `/api/runs/${runId}/retrain`),
  probe: (config: RunConfig) => req<ProbeResult>('POST', '/api/batch-size', { config }),
  probeQuick: (config: RunConfig) => req<ProbeResult>('POST', '/api/batch-size', { config, only_requested: true }),
  tune: (config: RunConfig, optuna: unknown) => req<{ job: Job }>('POST', '/api/tune', { config, optuna }),
  datasetShifts: (dataset: string, example = 0) => req<ShiftInfo[]>('GET', `/api/datasets/${dataset}/shifts?example=${example}`),
  measureShifts: (runId: string) => req<ShiftScores>('POST', `/api/runs/${runId}/shifts`),
  uploadDataset: (body: { example?: boolean; filename?: string; content?: string }) => req<DatasetPreview>('POST', '/api/datasets/upload', body),
  hubDataset: (repo: string, config?: string) => req<DatasetPreview>('POST', '/api/datasets/hub', { repo, config }),
  importDataset: (body: Record<string, unknown>) => req<DatasetInfo>('POST', '/api/datasets/import', body),
  deleteDataset: (id: string) => req<{ ok: boolean; runs_left: number }>('DELETE', `/api/datasets/${id}`),
  compatible: (architecture: RunConfig['architecture'], custom_blocks?: RunConfig['custom_blocks']) =>
    req<{ id: string; ok: boolean; params: number | null }[]>('POST', '/api/datasets/compatible', { architecture, custom_blocks }),
  trainSuite: (config: RunConfig, datasets: string[]) => req<{ job: Job }>('POST', '/api/train-suite', { config, datasets }),
  hubStatus: () => req<HubStatus>('GET', '/api/hub/status'),
  hubLogin: (token: string) => req<HubStatus>('POST', '/api/hub/login', { token }),
  hubLogout: () => req<HubStatus>('POST', '/api/hub/logout'),
  hubModels: (q: string) => req<HubModel[]>('GET', `/api/hub/models?q=${encodeURIComponent(q)}`),
  hubImport: (repo: string, allow_code = false) =>
    req<{ run_id?: string; name?: string; needs_confirm?: boolean; custom_blocks?: Record<string, string> }>('POST', '/api/hub/import', { repo, allow_code }),
  publish: (runId: string, name: string, isPrivate: boolean) => req<{ repo: string; url: string; files: string[] }>('POST', `/api/runs/${runId}/publish`, { name, private: isPrivate }),
  combine: (kind: string, runs: string[]) => req<{ job: Job }>('POST', '/api/combine', { kind, runs }),
  cook: (runId: string, o: { n: number; epochs: number; lr_scale: number }) => req<{ job: Job }>('POST', `/api/runs/${runId}/cook`, o),
  autopilot: (runId: string, o: { rounds: number; architecture: boolean }) => req<{ job: Job }>('POST', `/api/runs/${runId}/autopilot`, o),
  scalingPlan: (config: RunConfig, axis: string, factors: number[]) => req<ScalePlanRow[]>('POST', '/api/scaling/plan', { config, axis, factors }),
  scaling: (config: RunConfig, axis: string, factors: number[]) => req<{ job: Job }>('POST', '/api/scaling', { config, axis, factors }),
  importConfig: (cfg: unknown) =>
    req<{ config: RunConfig; notes: string[]; env_diffs: string[]; ok: boolean; error: string | null }>('POST', '/api/config/import', cfg),

  runs: () => req<RunRow[]>('GET', '/api/runs'),
  predict: (id: string, n = 12, seed = 0, split = 'test') => req<Predictions>('POST', `/api/runs/${id}/predict`, { n, seed, split }),
  samples: (dataset: string, o: { n?: number; split?: string; seed?: number; label?: number | null }) =>
    req<Samples>('GET', `/api/datasets/${dataset}/samples?n=${o.n ?? 24}&split=${o.split ?? 'train'}&seed=${o.seed ?? 0}${o.label != null ? `&label=${o.label}` : ''}`),
  evaluate: (id: string, o: { split?: string; limit?: number; true?: number | null; pred?: number | null }) =>
    req<Evaluation>('POST', `/api/runs/${id}/evaluate`, o),
  predictInput: (id: string, input: { image?: number[]; text?: string; vector?: number[] }) =>
    req<InputPrediction>('POST', `/api/runs/${id}/predict-input`, input),
  generate: (id: string, o: { prompt: string; max_new?: number; temperature?: number; seed?: number }) =>
    req<Generation>('POST', `/api/runs/${id}/generate`, o),
  run: (id: string) => req<RunDetail>('GET', `/api/runs/${id}`),
  renameRun: (id: string, name: string) => req('PATCH', `/api/runs/${id}`, { name }),
  deleteRun: (id: string) => req('DELETE', `/api/runs/${id}`),

  jobs: () => req<Job[]>('GET', '/api/jobs'),
  allJobs: () => req<Job[]>('GET', '/api/jobs?active_only=false'),
  stopJob: (id: string) => req('POST', `/api/jobs/${id}/stop`),

}

/** Stream a job's events. Returns an unsubscribe function. */
export function streamJob(jobId: string, onEvent: (e: JobEvent) => void, onEnd?: () => void, since = 0): () => void {
  let closed = false
  let last = since - 1
  let es: EventSource | null = null
  const open = () => {
    es = new EventSource(`/api/jobs/${jobId}/events?since=${last + 1}`)
    es.onmessage = (m) => {
      const e = JSON.parse(m.data) as JobEvent
      if (e.seq <= last) return
      last = e.seq
      onEvent(e)
    }
    es.addEventListener('end', () => {
      es?.close()
      if (!closed) onEnd?.()
      closed = true
    })
    es.onerror = () => {
      // Reconnect from the last seen event (EventSource's own retry would restart at `since`).
      es?.close()
      if (!closed) setTimeout(() => !closed && open(), 1000)
    }
  }
  open()
  return () => {
    closed = true
    es?.close()
  }
}

export function downloadJSON(data: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}

/** Download a model as standalone Python scripts (a zip with train.py and generate.py or predict.py). */
export async function downloadPython(config: RunConfig) {
  const res = await fetch('/api/export', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(config) })
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? res.statusText)
  const name = /filename="([^"]+)"/.exec(res.headers.get('Content-Disposition') ?? '')?.[1] ?? 'model_python.zip'
  const a = document.createElement('a')
  a.href = URL.createObjectURL(await res.blob())
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}
