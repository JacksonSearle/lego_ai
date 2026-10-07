import type {
  BlockSpec, DatasetInfo, HistoryStage, InspectResult, Job, JobEvent, Meta, ProbeResult, RunConfig, RunDetail, RunRow, Template,
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
  inspect: (dataset: string, architecture: RunConfig['architecture']) =>
    req<InspectResult>('POST', '/api/inspect', { dataset, architecture }),

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
  importConfig: (cfg: unknown) =>
    req<{ config: RunConfig; notes: string[]; env_diffs: string[]; ok: boolean; error: string | null }>('POST', '/api/config/import', cfg),

  runs: () => req<RunRow[]>('GET', '/api/runs'),
  run: (id: string) => req<RunDetail>('GET', `/api/runs/${id}`),
  renameRun: (id: string, name: string) => req('PATCH', `/api/runs/${id}`, { name }),
  deleteRun: (id: string) => req('DELETE', `/api/runs/${id}`),

  jobs: () => req<Job[]>('GET', '/api/jobs'),
  stopJob: (id: string) => req('POST', `/api/jobs/${id}/stop`),

  history: () => req<{ stages: HistoryStage[]; budget: Record<string, unknown>; job: Job | null }>('GET', '/api/history'),
  historyRun: (only?: string[]) => req<{ job: Job }>('POST', '/api/history/run', { only }),
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
