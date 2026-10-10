import { useMemo } from 'react'
import { api } from '../api'
import { LineChart } from '../components/LineChart'
import { useStore } from '../store'
import { fmtNum, fmtPct } from '../util'

/** One-click "train & test" panel shown in the Builder until the Train page unlocks. */
export function QuickTrain({ valid }: { valid: boolean }) {
  const { draft, session, attachJob, toConfig, toast } = useStore()
  const running = session?.status === 'running'
  const lastEp = session?.epochPts[session.epochPts.length - 1]
  const lastStep = session?.steps[session.steps.length - 1]
  const series = useMemo(() => [{ label: 'training accuracy', color: '#60a5fa', points: (session?.steps ?? []).map((s) => ({ x: s.epoch, y: s.acc })) }], [session?.steps])

  const start = async () => {
    try {
      const cfg = toConfig()
      const { job, run } = await api.train(cfg)
      attachJob(job.id, 'train', cfg.name, run.id, { built: !!draft.edited })
    } catch (e) {
      toast((e as Error).message, 'error')
    }
  }

  return (
    <div className="inspector-head quick-train" data-tour="quick-train">
      <h4>Train & test</h4>
      <div className="row" style={{ marginTop: 8 }}>
        <button className="primary" disabled={!valid || running} onClick={start}>{running ? <><span className="spinner" /> Training…</> : '▶ Train this model'}</button>
        {running && session && <button className="danger sm" onClick={() => api.stopJob(session.jobId)}>Stop</button>}
      </div>
      {!valid && <div className="tiny bad" style={{ marginTop: 6 }}>Fix the red blocks first: the model must be valid to train.</div>}
      {session && (
        <div style={{ marginTop: 10 }}>
          <div className="row">
            <span className="mono" style={{ fontSize: 22, fontWeight: 800, color: 'var(--green)' }}>{fmtPct(lastEp?.val_acc ?? lastStep?.acc, 1)}</span>
            <span className="tiny muted grow">{lastEp ? 'validation accuracy' : 'training accuracy so far'}</span>
            {session.params != null && <span className="tiny muted">{fmtNum(session.params)} weights</span>}
          </div>
          {session.steps.length > 1 && <LineChart series={series} percent height={110} />}
          {session.error && <div className="error-box">{session.error}</div>}
        </div>
      )}
      <div className="tiny faint" style={{ marginTop: 6 }}>Trains for 1 epoch on all 55,000 digits. The Train page with more controls unlocks at level 5.</div>
    </div>
  )
}
