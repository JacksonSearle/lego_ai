import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import { MiniStack } from '../components/ArchView'
import { LiveCharts } from '../components/LiveCharts'
import { NumInput } from '../components/NumInput'
import { ParamField } from '../components/ParamField'
import { HP_FEATURE } from '../learn/curriculum'
import { skillForDataset } from '../learn/engine'
import { GuideButton } from '../learn/GuideButton'
import { useLearn } from '../learn/LearnContext'
import { useStore } from '../store'
import type { HyperParam, ProbeResult, RunRow, SearchSpec } from '../types'
import { fmtDur, fmtLoss, fmtNum, fmtPct } from '../util'

const SPACE_KEY = 'lego-ai/search-space-v2'

export function Train() {
  const { meta, draft, setDraft, dataset, datasets, inspect, blocks: specs, session, attachJob, toConfig, setTab, toast, loadConfig } = useStore()
  const [probe, setProbe] = useState<ProbeResult | null>(null)
  const [probing, setProbing] = useState(false)
  const [starting, setStarting] = useState(false)
  const [override, setOverride] = useState<string | null>(null)
  const [useOptuna, setUseOptuna] = useState(false)
  const [optuna, setOptuna] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(SPACE_KEY) ?? 'null') as OptunaOpts | null
    } catch {
      return null
    }
  })
  const [runs, setRuns] = useState<RunRow[]>([])
  const { access, flag } = useLearn()

  useEffect(() => {
    if (meta && !optuna) setOptuna({ n_trials: 10, epochs: 2, metric: 'val_acc', pruning: true, space: meta.search_space })
  }, [meta, optuna])
  useEffect(() => {
    if (optuna) localStorage.setItem(SPACE_KEY, JSON.stringify(optuna))
  }, [optuna])
  useEffect(() => {
    api.runs().then(setRuns)
  }, [session?.status])
  // A memory check is only valid for the model/device/optimizer it was run with.
  const probeKey = JSON.stringify([draft.dataset, draft.blocks, draft.training.device, draft.training.optimizer, draft.training.dropout])
  useEffect(() => setProbe(null), [probeKey])

  const t = draft.training
  const setT = (k: string, v: unknown) => setDraft((d) => ({ ...d, training: { ...d.training, [k]: v } }), { undoable: false })
  const running = session?.status === 'running'
  const valid = !!inspect?.ok && draft.blocks.length > 0

  const groups = useMemo(() => {
    const g: Record<string, HyperParam[]> = {}
    for (const h of meta?.hyperparams ?? []) (g[h.group] ??= []).push(h)
    return Object.entries(g)
  }, [meta])

  const visible = (h: HyperParam) =>
    !h.show_if || Object.entries(h.show_if).every(([k, vals]) => vals.includes(t[k] ?? meta?.defaults[k]))
  const shown = (h: HyperParam) => visible(h) && (!HP_FEATURE[h.name] || access.has(HP_FEATURE[h.name]))

  const runProbe = async () => {
    setProbing(true)
    setProbe(null)
    flag('probe')
    try {
      setProbe(await api.probe(toConfig()))
    } catch (e) {
      toast(`Batch size check failed: ${(e as Error).message}`, 'error')
    } finally {
      setProbing(false)
    }
  }

  const start = async () => {
    setStarting(true)
    try {
      const cfg = toConfig()
      // Make sure the chosen batch size fits before committing to a run.
      const quick = await api.probeQuick(cfg)
      if (quick.requested_fits === false && override !== probeKey + t.batch_size) {
        toast(`Batch size ${t.batch_size} probably won't fit in memory on ${quick.device}. Testing alternatives: pick one, or press “Train anyway”.`, 'error')
        setOverride(probeKey + t.batch_size)
        setStarting(false)
        await runProbe()
        return
      }
      if (useOptuna && optuna && access.has('train.optuna')) {
        const { job } = await api.tune(cfg, optuna)
        attachJob(job.id, 'tune', cfg.name)
        toast(`Optuna study started: ${optuna.n_trials} trials`, 'ok')
      } else {
        const { job, run } = await api.train(cfg)
        attachJob(job.id, 'train', cfg.name, run.id, { built: !!draft.edited })
      }
    } catch (e) {
      toast((e as Error).message, 'error')
    } finally {
      setStarting(false)
    }
  }

  const stop = () => session && api.stopJob(session.jobId).then(() => toast(session.kind === 'train' ? 'Pausing after the current step. Press Resume to continue from the same point.' : 'Stopping after the current step…'))

  return (
    <div className="page">
      <div className="train-grid">
        {/* ------------- Settings column ------------- */}
        <div className="col" style={{ gap: 14 }}>
          <div className="panel">
            <div className="panel-head">
              <h3 className="grow">Model & data</h3>
              <button className="sm ghost" onClick={() => setTab('build')}>Edit architecture</button>
            </div>
            <div className="panel-body col" style={{ gap: 12 }}>
              <div className="row" style={{ alignItems: 'flex-start', gap: 14 }}>
                <MiniStack blocks={draft.blocks} specs={specs} width={70} />
                <div className="grow col" style={{ gap: 8, paddingLeft: 18 }}>
                  <input value={draft.name} onChange={(e) => setDraft((d) => ({ ...d, name: e.target.value }), { undoable: false })} />
                  <div className="small muted">
                    {draft.blocks.length} blocks · {fmtNum(inspect?.total_params)} params ·{' '}
                    {valid ? <span className="ok">valid</span> : <span className="bad">{draft.blocks.length ? 'invalid: fix in builder' : 'empty'}</span>}
                  </div>
                </div>
              </div>
              <div className="hp-grid">
                <label className="field">
                  <span className="lbl">Dataset</span>
                  <select value={draft.dataset} disabled={!access.has('datasets')} onChange={(e) => { setDraft((d) => ({ ...d, dataset: e.target.value })); flag('datasetChanged') }}>
                    {datasets.map((d) => {
                      const ok = !access.datasets || access.datasets.has(d.id)
                      return <option key={d.id} value={d.id} disabled={!ok}>{ok ? d.name : `🔒 ${d.name} (${skillForDataset(d.id)?.title ?? 'skill'})`}</option>
                    })}
                  </select>
                </label>
                {access.has('datasets') && <label className="field">
                  <span className="lbl">Training subset</span>
                  <NumInput int min={1} placeholder={`all ${fmtNum(dataset?.train_size)}`} value={draft.train_subset}
                    onChange={(n) => setDraft((d) => ({ ...d, train_subset: n }), { undoable: false })} />
                </label>}
                <span className="help full">{dataset?.description}</span>
              </div>
              {(access.has('config.io') || access.has('train.resume')) && <PreviousRuns runs={runs} onLoad={async (id) => { const r = await api.run(id); loadConfig({ ...r.config, name: r.config.name }); toast(`Loaded config of “${r.name}”: press Train to retrain from scratch`, 'ok') }} />}
            </div>
          </div>

          <div className="panel" data-tour="hyperparams">
            <div className="panel-head"><h3 className="grow">Hyperparameters</h3><GuideButton lesson="app.train" label="guide" /></div>
            <div className="panel-body">
              {groups.filter(([, hs]) => hs.some(shown)).map(([group, hs]) => (
                <div key={group} className="hp-group">
                  <h4>{group}</h4>
                  <div className="hp-grid">
                    {hs.filter(shown).map((h) => {
                      const spec = h.name === 'device' ? { ...h, choices: (h.choices ?? []).filter((c) => meta?.environment.available_devices.includes(String(c))) } : h
                      const tuned = useOptuna && optuna?.space[h.name]?.enabled
                      return (
                        <div key={h.name} className={h.name === 'batch_size' ? 'full' : ''} data-tour={`hp-${h.name}`}>
                          <ParamField spec={spec} value={t[h.name]} onChange={(v) => setT(h.name, v)} disabled={!!tuned} />
                          {tuned && <span className="pill accent" style={{ marginTop: 4 }}>tuned by Optuna</span>}
                          {h.name === 'batch_size' && (
                            <div className="row" style={{ marginTop: 6 }}>
                              <button className="sm" onClick={runProbe} disabled={probing || !valid}>
                                {probing ? <><span className="spinner" /> Testing batch sizes…</> : 'Check what fits in memory'}
                              </button>
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                  {group === 'Basics' && probe && <ProbeTable probe={probe} current={t.batch_size as number} onPick={(b) => setT('batch_size', b)} />}
                </div>
              ))}
            </div>
          </div>

          {optuna && meta && access.has('train.optuna') && (
            <OptunaPanel enabled={useOptuna} setEnabled={setUseOptuna} opts={optuna} setOpts={setOptuna} hps={meta.hyperparams} reset={() => setOptuna({ n_trials: 10, epochs: 2, metric: 'val_acc', pruning: true, space: meta.search_space })} />
          )}

          <div className="row">
            {running ? (
              <button className="big danger" onClick={stop}>{session?.kind === 'train' ? '⏸ Pause' : '■ Stop'}</button>
            ) : (
              <button className="big primary" onClick={start} disabled={!valid || starting}>
                {starting ? <><span className="spinner" /> Checking memory…</> : override === probeKey + t.batch_size ? '▶ Train anyway' : useOptuna ? `▶ Run Optuna (${optuna?.n_trials} trials)` : '▶ Train'}
              </button>
            )}
            <span className="small muted">
              {fmtNum(draft.train_subset ? Math.min(draft.train_subset, dataset?.train_size ?? Infinity) : dataset?.train_size)} examples · seed {String(t.seed)} · {String(t.device)}
            </span>
          </div>
        </div>

        {/* ------------- Live column ------------- */}
        <LivePanel task={dataset?.task} />
      </div>
    </div>
  )
}

interface OptunaOpts {
  n_trials: number
  epochs: number
  metric: 'val_acc' | 'val_loss'
  pruning: boolean
  space: Record<string, SearchSpec>
}

function OptunaPanel({ enabled, setEnabled, opts, setOpts, hps, reset }: {
  enabled: boolean; setEnabled: (b: boolean) => void; opts: OptunaOpts; setOpts: (o: OptunaOpts) => void; hps: HyperParam[]; reset: () => void
}) {
  const set = (k: string, s: SearchSpec) => setOpts({ ...opts, space: { ...opts.space, [k]: s } })
  return (
    <div className="panel" data-tour="optuna">
      <div className="panel-head">
        <label className="switch">
          <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
          <span className="track" />
        </label>
        <h3 className="grow">Hyperparameter search (Optuna)</h3>
        {enabled && <button className="sm ghost" onClick={reset}>Reset ranges</button>}
      </div>
      {!enabled ? (
        <div className="panel-body small muted">Give ranges instead of exact values and let Optuna's TPE sampler search them. Bad trials are pruned early; every trial is saved as a reproducible run.</div>
      ) : (
        <div className="panel-body col" style={{ gap: 12 }}>
          <div className="hp-grid">
            <label className="field"><span className="lbl">Trials</span>
              <NumInput int min={1} max={500} value={opts.n_trials} emptyValue={10} onChange={(n) => setOpts({ ...opts, n_trials: n ?? 10 })} /></label>
            <label className="field"><span className="lbl">Epochs per trial</span>
              <NumInput int min={1} max={100} value={opts.epochs} emptyValue={2} onChange={(n) => setOpts({ ...opts, epochs: n ?? 2 })} /></label>
            <label className="field"><span className="lbl">Optimize</span>
              <select value={opts.metric} onChange={(e) => setOpts({ ...opts, metric: e.target.value as OptunaOpts['metric'] })}>
                <option value="val_acc">max validation accuracy</option>
                <option value="val_loss">min validation loss</option>
              </select></label>
            <label className="field"><span className="lbl">Pruning</span>
              <label className="switch"><input type="checkbox" checked={opts.pruning} onChange={(e) => setOpts({ ...opts, pruning: e.target.checked })} /><span className="track" /><span className="small muted">stop bad trials early</span></label></label>
          </div>
          <h4>Search space</h4>
          <div>
            {Object.entries(opts.space).map(([k, s]) => {
              const h = hps.find((x) => x.name === k)
              return (
                <div key={k} className="range-row">
                  <input type="checkbox" checked={!!s.enabled} onChange={(e) => set(k, { ...s, enabled: e.target.checked })} />
                  <span className="small" style={{ fontWeight: 600, opacity: s.enabled ? 1 : 0.5 }}>{h?.label ?? k}</span>
                  {s.type === 'categorical' ? (
                    <ChoicesInput disabled={!s.enabled} choices={s.choices ?? []} onChange={(choices) => set(k, { ...s, choices })} />
                  ) : (
                    <div className="range-inputs">
                      <NumInput disabled={!s.enabled} value={s.low} emptyValue={s.low ?? 0} onChange={(n) => set(k, { ...s, low: n ?? 0 })} />
                      <span className="faint">to</span>
                      <NumInput disabled={!s.enabled} value={s.high} emptyValue={s.high ?? 1} onChange={(n) => set(k, { ...s, high: n ?? 1 })} />
                      <label className="row tiny muted" title="Sample on a log scale"><input type="checkbox" disabled={!s.enabled} checked={!!s.log} onChange={(e) => set(k, { ...s, log: e.target.checked })} />log</label>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

/** Comma-separated choices; parsed when the field loses focus so typing commas isn't fought. */
function ChoicesInput({ choices, onChange, disabled }: { choices: unknown[]; onChange: (c: unknown[]) => void; disabled?: boolean }) {
  const [text, setText] = useState(choices.join(', '))
  useEffect(() => setText(choices.join(', ')), [choices])
  const commit = () => onChange(text.split(',').map((x) => x.trim()).filter(Boolean).map((x) => (isNaN(Number(x)) ? x : Number(x))))
  return <input disabled={disabled} value={text} title="Comma-separated choices" onChange={(e) => setText(e.target.value)} onBlur={commit} onKeyDown={(e) => e.key === 'Enter' && commit()} />
}

function ProbeTable({ probe, current, onPick }: { probe: ProbeResult; current: number; onPick: (b: number) => void }) {
  const [min, setMin] = useState(false)
  if (min) {
    return (
      <div className="card row small" style={{ marginTop: 12, padding: '8px 12px' }}>
        <span className="grow">
          <b>Memory check</b>: largest batch that fits is <b>{probe.largest_fitting ?? 'none'}</b>
          {probe.requested_fits === false && <span className="bad"> · {probe.requested} does not fit</span>}
        </span>
        <button className="ghost sm" title="Show the full table" onClick={() => setMin(false)}>＋</button>
      </div>
    )
  }
  return (
    <div className="card" style={{ marginTop: 12, padding: 0 }}>
      <div style={{ padding: '10px 12px' }} className="small">
        <button className="ghost sm" style={{ float: 'right', marginLeft: 8 }} title="Minimize" onClick={() => setMin(true)}>－</button>
        <b>Memory check on {probe.device}</b>{probe.train_examples ? <> (epoch times are for {fmtNum(probe.train_examples)} examples)</> : null}: budget {fmtNum(Math.round(probe.budget_mb))} MB ({probe.budget_label}). Weights + grads + optimizer state ≈ {probe.fixed_mb.toFixed(1)} MB;
        activations ≈ {probe.activation_kb_per_sample.toFixed(0)} KB per example.
        {probe.requested_fits === false && <div className="bad" style={{ marginTop: 4 }}>Your batch size of {probe.requested} does not fit.</div>}
        {probe.note && <div className="warn" style={{ marginTop: 4 }}>Some sizes are estimates only: {probe.note}.</div>}
        {probe.largest_fitting && <div style={{ marginTop: 4 }}>Largest that fits: <b>{probe.largest_fitting}</b>{probe.fastest && <> · highest throughput: <b>{probe.fastest}</b></>}. Click a row to use it.</div>}
      </div>
      <table className="tbl">
        <thead><tr><th>Batch</th><th>Fits</th><th className="right">Memory</th><th className="right">Step</th><th className="right">Epoch</th></tr></thead>
        <tbody>
          {probe.results.map((r) => (
            <tr key={r.batch_size} className={`probe-row ${r.fits ? 'fit' : 'nofit'} ${r.batch_size === current ? 'pick' : ''} ${r.fits ? 'clickable' : ''}`} onClick={() => r.fits && onPick(r.batch_size)}>
              <td className="mono">{r.batch_size}</td>
              <td>{r.fits ? <span className="ok">✓ {r.tested ? 'tested' : 'est.'}</span> : <span className="bad" title={r.reason}>✗ {r.reason}</span>}</td>
              <td className="num">{(r.measured_mb ?? r.estimated_mb).toFixed(0)} MB</td>
              <td className="num">{r.step_ms ? `${r.step_ms.toFixed(0)} ms` : '-'}</td>
              <td className="num">{r.epoch_s ? fmtDur(r.epoch_s) : '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function PreviousRuns({ runs, onLoad }: { runs: RunRow[]; onLoad: (id: string) => void }) {
  const { attachJob, toast } = useStore()
  const { access } = useLearn()
  const canLoad = access.has('config.io')
  const canResume = access.has('train.resume')
  const [pick, setPick] = useState('')
  const [epochs, setEpochs] = useState(0)
  const run = runs.find((r) => r.id === pick)
  useEffect(() => {
    if (!run) return
    // A paused run resumes to its own target; a finished one gets some extra epochs by default.
    const target = run.training.epochs as number
    const done = run.summary.epochs_done ?? 0
    setEpochs(done < target ? target : done + Math.max(1, Math.ceil(target / 2)))
  }, [run])
  if (!runs.length) return null
  return (
    <div className="card col" style={{ gap: 8 }}>
      <span className="lbl small" style={{ fontWeight: 600 }}>Start from a previous run</span>
      <select value={pick} onChange={(e) => setPick(e.target.value)}>
        <option value="">Choose a run…</option>
        {runs.map((r) => (
          <option key={r.id} value={r.id}>
            {r.name} · {r.dataset} · {fmtPct(r.summary.best_val_acc)} · {r.status}
          </option>
        ))}
      </select>
      {run && (
        <div className="row wrap">
          {canLoad && <button className="sm" onClick={() => onLoad(run.id)} title="Copy its architecture and hyperparameters into the editor">Load config (retrain from scratch)</button>}
          {canResume && run.has_checkpoint && run.status !== 'running' && (
            <>
              {canLoad && <span className="faint small">or</span>}
              <button className="sm" onClick={async () => {
                try {
                  const { job } = await api.resume(run.id, epochs)
                  attachJob(job.id, 'train', run.name, run.id, { resumed: true })
                  toast(`Resuming “${run.name}” from epoch ${run.summary.epochs_done}`, 'ok')
                } catch (e) { toast((e as Error).message, 'error') }
              }}>Resume checkpoint to</button>
              <NumInput int style={{ width: 64 }} min={(run.summary.epochs_done ?? 0) + 1} value={epochs} emptyValue={epochs} onChange={(n) => setEpochs(n ?? epochs)} />
              <span className="small muted">epochs</span>
            </>
          )}
        </div>
      )}
    </div>
  )
}

function LivePanel({ task }: { task?: string }) {
  const { session, clearSession, setTab, attachJob, toast } = useStore()
  const [now, setNow] = useState(Date.now())
  const [resuming, setResuming] = useState(false)
  useEffect(() => {
    if (session?.status !== 'running') return
    setNow(Date.now())
    const h = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(h)
  }, [session?.status, session?.jobId])

  if (!session) {
    return (
      <div className="empty" style={{ padding: 60 }}>
        <h3 style={{ color: 'var(--text)' }}>No training in progress</h3>
        <p>Set your hyperparameters and press <b>Train</b>. Loss and accuracy will stream in here live.</p>
      </div>
    )
  }
  const lastStep = session.steps[session.steps.length - 1]
  const lastEp = session.epochPts[session.epochPts.length - 1]
  const progress = session.totalSteps && lastStep ? Math.min(1, lastStep.step / session.totalSteps) : 0
  const running = session.status === 'running'
  // While running: time since this session (re)started. Afterwards: the run's total training time.
  const elapsed = running ? Math.max(0, (now - session.startedAt) / 1000) : (session.summary?.duration ?? 0)
  const rate = lastStep && running && elapsed > 2 ? (lastStep.step - session.baseStep) / elapsed : 0
  const paused = session.kind === 'train' && session.status === 'stopped' && !!session.runId
  const resume = async () => {
    if (!session.runId) return
    setResuming(true)
    try {
      // epochs: 0 means "the run's own target"
      const { job } = await api.resume(session.runId, 0)
      attachJob(job.id, 'train', session.name, session.runId, { ...session.learn, resumed: true })
    } catch (e) {
      toast((e as Error).message, 'error')
    } finally {
      setResuming(false)
    }
  }
  const eta = rate > 0 && session.totalSteps && lastStep ? (session.totalSteps - lastStep.step) / rate : null
  const bestVal = session.epochPts.length ? Math.max(...session.epochPts.map((e) => e.val_acc)) : null
  const statusCls = session.status === 'running' ? 'accent' : session.status === 'completed' ? 'ok' : session.status === 'failed' ? 'bad' : 'warn'

  return (
    <div className="col" style={{ gap: 14 }}>
      <div className="panel">
        <div className="panel-head">
          <h3 className="grow">
            {session.name}
            {session.kind === 'tune' && session.currentTrial && <span className="muted"> · trial {session.currentTrial}</span>}
          </h3>
          <span className={`pill ${statusCls}`}>{session.status === 'running' && <span className="spinner" style={{ width: 10, height: 10 }} />} {paused ? 'paused' : session.status}</span>
          {session.status !== 'running' && <button className="sm ghost" onClick={clearSession}>Dismiss</button>}
        </div>
        <div className="panel-body col" style={{ gap: 10 }}>
          <div className="row small muted">
            <span>{session.message}</span>
            <span className="spacer" />
            {session.device && <span>{session.device}</span>}
            {session.params != null && <span>· {fmtNum(session.params)} params</span>}
            {session.trainExamples != null && <span>· {fmtNum(session.trainExamples)} examples</span>}
            {lastStep && <span>· epoch {lastStep.epoch.toFixed(2)} / {session.epochs}</span>}
            <span>· {fmtDur(elapsed)}</span>
            {session.status === 'running' && eta != null && <span>· ~{fmtDur(eta)} left</span>}
          </div>
          <div className="progress"><div style={{ width: `${progress * 100}%` }} /></div>
          {session.error && <div className="error-box">{session.error}</div>}
          {paused && (
            <div className="note-box row">
              <span className="grow">
                {lastStep
                  ? <>Paused at epoch {lastStep.epoch.toFixed(2)} of {session.epochs}. Resuming continues from exactly this point, as if it had never stopped.</>
                  : <>Stopped before the first training step.</>}
              </span>
              {lastStep && <button className="primary" disabled={resuming} onClick={resume}>{resuming ? <><span className="spinner" /> Resuming…</> : '▶ Resume'}</button>}
            </div>
          )}
        </div>
      </div>

      <div className="kpis">
        <div className="kpi"><div className="k">Train loss</div><div className="v">{fmtLoss(lastStep?.loss)}</div></div>
        <div className="kpi"><div className="k">Val loss</div><div className="v">{fmtLoss(lastEp?.val_loss)}</div></div>
        <div className="kpi"><div className="k">Val acc</div><div className="v">{fmtPct(lastEp?.val_acc)}</div></div>
        <div className="kpi"><div className="k">Best val acc</div><div className="v ok">{fmtPct(bestVal)}</div></div>
        {session.summary?.test_acc != null && <div className="kpi"><div className="k">Test acc</div><div className="v" style={{ color: 'var(--accent)' }}>{fmtPct(session.summary.test_acc)}</div></div>}
      </div>

      <LiveCharts steps={session.steps} epochs={session.epochPts} task={task} />

      {session.kind === 'tune' && <TrialsTable />}

      {session.status !== 'running' && session.kind === 'train' && (
        <div className="row">
          <button onClick={() => setTab('leaderboard')}>View on leaderboard →</button>
        </div>
      )}
    </div>
  )
}

function TrialsTable() {
  const { session, setDraft, toast } = useStore()
  const { flag } = useLearn()
  if (!session) return null
  const fmt = (v: unknown) => (typeof v === 'number' ? (Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(2) : +v.toPrecision(3)) : String(v))
  const keys = Array.from(new Set(session.trials.flatMap((t) => Object.keys(t.params))))
  const pct = session.metric !== 'val_loss'
  return (
    <div className="panel">
      <div className="panel-head">
        <h3 className="grow">Optuna trials</h3>
        {session.best && (
          <button className="sm primary" onClick={() => { setDraft((d) => ({ ...d, training: { ...d.training, ...session.best!.params } })); flag('applyBest'); toast('Applied the best hyperparameters to the training form', 'ok') }}>
            Use best params
          </button>
        )}
      </div>
      <table className="tbl">
        <thead>
          <tr><th>#</th>{keys.map((k) => <th key={k}>{k}</th>)}<th className="right">{pct ? 'val acc' : 'val loss'}</th><th>state</th></tr>
        </thead>
        <tbody>
          {[...session.trials].sort((a, b) => a.trial - b.trial).map((tr) => (
            <tr key={tr.trial} className={session.best?.trial === tr.trial ? 'selected' : ''}>
              <td className="mono">{tr.trial}{session.best?.trial === tr.trial && ' ★'}</td>
              {keys.map((k) => <td key={k} className="mono small">{fmt(tr.params[k])}</td>)}
              <td className="num">{tr.value == null ? '-' : pct ? fmtPct(tr.value) : fmtLoss(tr.value)}</td>
              <td><span className={`pill ${tr.state === 'complete' ? 'ok' : tr.state === 'pruned' ? 'warn' : 'bad'}`}>{tr.state}</span></td>
            </tr>
          ))}
          {session.status === 'running' && session.currentTrial && !session.trials.some((x) => x.trial === session.currentTrial) && (
            <tr><td className="mono">{session.currentTrial}</td><td colSpan={keys.length + 2} className="small muted"><span className="spinner" style={{ width: 10, height: 10 }} /> running…</td></tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
