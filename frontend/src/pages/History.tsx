import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, streamJob } from '../api'
import { ArchView } from '../components/ArchView'
import { LiveCharts } from '../components/LiveCharts'
import { GuideButton } from '../learn/GuideButton'
import { useLearn } from '../learn/LearnContext'
import { useStore } from '../store'
import type { EpochPoint, HistoryStage, JobEvent, StepPoint } from '../types'
import { fmtDur, fmtNum, fmtPct } from '../util'

const ERA_COLORS: Record<string, string> = {
  Statistics: '#94a3b8',
  'Early nets': '#60a5fa',
  Connectionism: '#22c55e',
  'Deep learning': '#f97316',
  Attention: '#a78bfa',
}

interface Live { steps: StepPoint[]; epochs: EpochPoint[] }

export function History() {
  const { blocks: specs, datasets, toast, loadConfig, setTab } = useStore()
  const { flag } = useLearn()
  const [stages, setStages] = useState<HistoryStage[]>([])
  const [budget, setBudget] = useState<Record<string, unknown>>({})
  const [jobId, setJobId] = useState<string | null>(null)
  const [running, setRunning] = useState<number | null>(null)
  const [selected, setSelected] = useState(0)
  const [follow, setFollow] = useState(true)
  const [live, setLive] = useState<Record<number, Live>>({})
  const unsub = useRef<(() => void) | null>(null)
  const autoStarted = useRef(false)
  const followRef = useRef(follow)
  followRef.current = follow

  const refresh = useCallback(async () => {
    const h = await api.history()
    setStages(h.stages)
    setBudget(h.budget)
    return h
  }, [])

  const attach = useCallback(
    (id: string) => {
      unsub.current?.()
      setJobId(id)
      unsub.current = streamJob(
        id,
        (e: JobEvent) => {
          const i = e.stage as number | undefined
          if (e.type === 'stage_start' && i != null) {
            setRunning(i)
            setLive((l) => ({ ...l, [i]: { steps: [], epochs: [] } }))
            if (followRef.current) setSelected(i)
          } else if (e.type === 'step' && i != null) {
            setLive((l) => ({ ...l, [i]: { ...(l[i] ?? { epochs: [] }), steps: [...(l[i]?.steps ?? []), e as unknown as StepPoint] } }))
          } else if (e.type === 'epoch' && i != null) {
            setLive((l) => ({ ...l, [i]: { ...(l[i] ?? { steps: [] }), epochs: [...(l[i]?.epochs ?? []), e as unknown as EpochPoint] } }))
          } else if (e.type === 'stage_end') {
            refresh()
          } else if (e.type === 'error') {
            toast(`History run failed: ${e.message}`, 'error')
          }
        },
        () => {
          setRunning(null)
          setJobId(null)
          refresh()
        },
      )
    },
    [refresh, toast],
  )

  const start = useCallback(
    async (only?: string[]) => {
      try {
        const { job } = await api.historyRun(only)
        setFollow(true)
        attach(job.id)
      } catch (e) {
        toast((e as Error).message, 'error')
      }
    },
    [attach, toast],
  )

  // On open: re-attach to a running tour, or start it automatically if nothing has been trained yet.
  useEffect(() => {
    refresh().then((h) => {
      if (h.job) attach(h.job.id)
      else if (!autoStarted.current && h.stages.every((s) => !s.run || s.run.status !== 'completed')) {
        autoStarted.current = true
        start()
      }
    })
    return () => unsub.current?.()
  }, [refresh, attach, start])

  const stage = stages[selected]
  const mnist = datasets.find((d) => d.id === 'mnist')
  const missing = stages.filter((s) => s.run?.status !== 'completed').map((s) => s.id)
  const allDone = stages.length > 0 && missing.length === 0
  useEffect(() => {
    if (allDone) flag('historyDone')
  }, [allDone, flag])

  const charts = useMemo(() => {
    if (!stage) return { steps: [], epochs: [] }
    if (live[selected] && (running === selected || !stage.metrics?.steps.length)) return live[selected]
    return { steps: stage.metrics?.steps ?? [], epochs: stage.metrics?.epochs ?? [] }
  }, [stage, live, selected, running])

  const board = useMemo(
    () =>
      stages
        .map((s, i) => ({ s, i, acc: s.run?.status === 'completed' ? s.run.summary.test_acc ?? null : null }))
        .filter((x) => x.acc != null)
        .sort((a, b) => (b.acc ?? 0) - (a.acc ?? 0)),
    [stages],
  )

  if (!stages.length) return <div className="page"><span className="spinner" /></div>

  const left = (i: number) => `${(i / (stages.length - 1)) * 100}%`
  const eras = stages.reduce<{ era: string; first: number; last: number }[]>((acc, s, i) => {
    const prev = acc[acc.length - 1]
    if (prev?.era === s.era) return [...acc.slice(0, -1), { ...prev, last: i }]
    return [...acc, { era: s.era, first: i, last: i }]
  }, [])
  const step = 100 / (stages.length - 1)

  return (
    <div className="page">
      <div className="row wrap" style={{ marginBottom: 14, gap: 12 }}>
        <div className="grow">
          <h2>From least squares to transformers <GuideButton lesson="app.history" /></h2>
          <div className="small muted" style={{ marginTop: 3 }}>
            Every model trains live on the same {fmtNum(budget.train_subset as number)} MNIST digits for {String(budget.epochs)} epochs (batch {String(budget.batch_size)}, {String(budget.device)}). Same data, same budget, only the architecture changes.
          </div>
        </div>
        {!follow && running != null && <button onClick={() => { setFollow(true); setSelected(running) }}>Follow live ●</button>}
        {jobId ? (
          <button className="danger" onClick={() => api.stopJob(jobId)}>■ Stop tour</button>
        ) : (
          <>
            {missing.length > 0 && missing.length < stages.length && <button onClick={() => start(missing)}>▶ Train remaining ({missing.length})</button>}
            <button className="primary" onClick={() => start()}>{missing.length === stages.length ? '▶ Start the tour' : '↻ Replay all'}</button>
          </>
        )}
      </div>

      {/* Timeline */}
      <div className="timeline" data-tour="timeline">
        <div className="timeline-track">
          {eras.map((e) => (
            <div key={e.era} className="timeline-era" title={e.era}
              style={{ left: `calc(${e.first * step}% - 42px)`, width: `calc(${(e.last - e.first) * step}% + 84px)`, color: ERA_COLORS[e.era] }}>
              {e.era}
            </div>
          ))}
          <div className="timeline-axis" />
          {stages.map((s, i) => {
            const done = s.run?.status === 'completed'
            const cls = ['tl-stop', done && 'done', running === i && 'running', selected === i && 'selected'].filter(Boolean).join(' ')
            return (
              <button key={s.id} className={cls} style={{ left: left(i) }} onClick={() => { setSelected(i); setFollow(running === i) }}>
                <span className="tl-year">{s.year}</span>
                <span className="tl-dot" style={done ? { background: ERA_COLORS[s.era] } : undefined} />
                <span className="tl-name">{s.title}</span>
                {done && <span className="tl-acc">{fmtPct(s.run!.summary.test_acc, 1)}</span>}
                {running === i && <span className="tl-acc" style={{ color: 'var(--accent)' }}>training…</span>}
              </button>
            )
          })}
        </div>
      </div>

      <div className="history-grid">
        {/* Architecture */}
        <div className="panel">
          <div className="panel-head">
            <h3 className="grow">Architecture</h3>
            <button className="sm ghost" title="Copy this model into the builder"
              onClick={() => { loadConfig({ name: `${stage.year} ${stage.title}`, dataset: { id: 'mnist', train_subset: null }, architecture: { blocks: stage.blocks }, training: stage.training }); setTab('build'); toast('Copied into the builder', 'ok') }}>
              Open in builder
            </button>
          </div>
          <div className="panel-body" style={{ display: 'flex', justifyContent: 'center' }}>
            <ArchView blocks={stage.blocks} specs={specs} dataset={mnist} width={250} />
          </div>
        </div>

        {/* Live training */}
        <div className="col" style={{ gap: 14 }}>
          <div className="panel">
            <div className="panel-head">
              <span className="pill" style={{ color: ERA_COLORS[stage.era], borderColor: ERA_COLORS[stage.era] + '55' }}>{stage.year}</span>
              <h3 className="grow">{stage.title}</h3>
              {running === selected && <span className="pill accent"><span className="spinner" style={{ width: 10, height: 10 }} /> training</span>}
              {stage.run?.status === 'completed' && running !== selected && (
                <span className="small muted">
                  test {fmtPct(stage.run.summary.test_acc)} · {fmtNum(stage.run.summary.params)} params · {fmtDur(stage.run.summary.duration)}
                </span>
              )}
              {!jobId && <button className="sm" onClick={() => start([stage.id])}>Retrain</button>}
            </div>
          </div>
          {charts.steps.length > 0 ? (
            <LiveCharts steps={charts.steps} epochs={charts.epochs} showLr={false} height={210} />
          ) : (
            <div className="empty">{jobId ? 'Waiting for its turn…' : 'Not trained yet: press Start.'}</div>
          )}
        </div>

        {/* Demo leaderboard */}
        <div className="panel">
          <div className="panel-head"><h3 className="grow">Demo leaderboard</h3><span className="small muted">test accuracy</span></div>
          <div className="panel-body" style={{ padding: 10 }}>
            {board.length === 0 && <div className="small muted" style={{ padding: 8 }}>Results appear here as each model finishes.</div>}
            {board.map(({ s, i, acc }, rank) => (
              <div key={s.id} className={`demo-lb-row ${selected === i ? 'selected' : ''}`} onClick={() => { setSelected(i); setFollow(false) }}>
                <span className="mono small muted">{rank + 1}. {s.year}</span>
                <div className="demo-lb-bar">
                  <div style={{ width: `${Math.max(0, ((acc! - 0.5) / 0.5) * 100)}%`, background: ERA_COLORS[s.era] }} />
                  <span>{s.title}</span>
                </div>
                <span className="mono small right">{fmtPct(acc, 1)}</span>
              </div>
            ))}
            {board.length > 0 && <div className="tiny faint" style={{ marginTop: 8, paddingLeft: 4 }}>Bars start at 50%. Kept separate from your own leaderboard.</div>}
          </div>
        </div>
      </div>

      {/* Blurb */}
      <div className="panel blurb" style={{ padding: 18 }}>
        <div>
          <h4 style={{ color: ERA_COLORS[stage.era] }}>{stage.year} · {stage.era}</h4>
          <h2 style={{ marginTop: 4 }}>{stage.title}</h2>
          <p>{stage.blurb}</p>
          <span className="added">＋ What it added: {stage.added}</span>
        </div>
        <a className="card" href={stage.paper.url} target="_blank" rel="noreferrer" style={{ maxWidth: 300, display: 'block', color: 'inherit' }}>
          <div className="tiny muted" style={{ fontWeight: 700, letterSpacing: '0.06em' }}>READ THE PAPER ↗</div>
          <div style={{ fontWeight: 650, marginTop: 4 }}>{stage.paper.title}</div>
          <div className="small muted" style={{ marginTop: 2 }}>{stage.paper.authors}, {stage.year}</div>
        </a>
      </div>
    </div>
  )
}
