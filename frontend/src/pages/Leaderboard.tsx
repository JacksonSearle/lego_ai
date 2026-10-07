import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import { ArchView, Chips } from '../components/ArchView'
import { LiveCharts } from '../components/LiveCharts'
import { useStore } from '../store'
import type { RunDetail, RunRow } from '../types'
import { fmtAgo, fmtDur, fmtLoss, fmtNum, fmtPct, MEDALS } from '../util'

type SortKey = 'best_val_acc' | 'test_acc' | 'best_val_loss' | 'params' | 'duration' | 'created_at'
const SORTS: { key: SortKey; label: string; desc: boolean }[] = [
  { key: 'best_val_acc', label: 'Best val acc', desc: true },
  { key: 'test_acc', label: 'Test acc', desc: true },
  { key: 'best_val_loss', label: 'Val loss', desc: false },
  { key: 'params', label: 'Params', desc: false },
  { key: 'duration', label: 'Train time', desc: false },
  { key: 'created_at', label: 'Newest', desc: true },
]

export function Leaderboard() {
  const { blocks: specs, datasets } = useStore()
  const [rows, setRows] = useState<RunRow[]>([])
  const [ds, setDs] = useState<string>('all')
  const [sort, setSort] = useState<SortKey>('best_val_acc')
  const [showTrials, setShowTrials] = useState(false)
  const [showFailed, setShowFailed] = useState(false)
  const [q, setQ] = useState('')
  const [selected, setSelected] = useState<string | null>(null)

  const load = useCallback(() => api.runs().then(setRows), [])
  useEffect(() => {
    load()
  }, [load])
  const anyRunning = rows.some((r) => r.status === 'running')
  useEffect(() => {
    if (!anyRunning) return
    const h = setInterval(load, 3000)
    return () => clearInterval(h)
  }, [anyRunning, load])

  const dsWithRuns = useMemo(() => datasets.filter((d) => rows.some((r) => r.dataset === d.id)), [datasets, rows])

  const filtered = useMemo(() => {
    const s = SORTS.find((x) => x.key === sort)!
    const val = (r: RunRow) => (sort === 'created_at' ? r.created_at : (r.summary[sort as keyof typeof r.summary] as number | undefined))
    return rows
      .filter((r) => ds === 'all' || r.dataset === ds)
      .filter((r) => showTrials || !r.study || r.study.best)
      .filter((r) => showFailed || !['failed', 'pruned'].includes(r.status))
      .filter((r) => !q || r.name.toLowerCase().includes(q.toLowerCase()))
      .sort((a, b) => {
        const va = val(a), vb = val(b)
        if (va == null && vb == null) return b.created_at - a.created_at
        if (va == null) return 1
        if (vb == null) return -1
        return s.desc ? vb - va : va - vb
      })
  }, [rows, ds, sort, showTrials, showFailed, q])

  const maxAcc = Math.max(0.0001, ...filtered.map((r) => r.summary.best_val_acc ?? 0))
  const ranked = sort === 'best_val_acc' || sort === 'test_acc' || sort === 'best_val_loss'

  return (
    <div className="page">
      <div className="row wrap" style={{ marginBottom: 14, gap: 12 }}>
        <h2>Leaderboard</h2>
        <div className="seg">
          <button className={ds === 'all' ? 'on' : ''} onClick={() => setDs('all')}>All</button>
          {dsWithRuns.map((d) => <button key={d.id} className={ds === d.id ? 'on' : ''} onClick={() => setDs(d.id)}>{d.name}</button>)}
        </div>
        <span className="spacer" />
        <input placeholder="Search…" value={q} onChange={(e) => setQ(e.target.value)} style={{ width: 160 }} />
        <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)}>
          {SORTS.map((s) => <option key={s.key} value={s.key}>Sort: {s.label}</option>)}
        </select>
        <label className="row small muted"><input type="checkbox" checked={showTrials} onChange={(e) => setShowTrials(e.target.checked)} /> all Optuna trials</label>
        <label className="row small muted"><input type="checkbox" checked={showFailed} onChange={(e) => setShowFailed(e.target.checked)} /> failed/pruned</label>
      </div>

      <div className={`lb-layout ${selected ? 'with-detail' : ''}`}>
        <div className="panel" style={{ overflow: 'auto' }}>
          {filtered.length === 0 ? (
            <div className="empty" style={{ margin: 20 }}>No runs yet. Build a model and train it — it will show up here.</div>
          ) : (
            <table className="tbl">
              <thead>
                <tr>
                  <th style={{ width: 36 }}>#</th>
                  <th>Model</th>
                  {ds === 'all' && <th>Dataset</th>}
                  <th>Architecture</th>
                  <th className="right sortable" onClick={() => setSort('params')}>Params</th>
                  <th className="sortable" onClick={() => setSort('best_val_acc')}>Best val acc</th>
                  <th className="right sortable" onClick={() => setSort('test_acc')}>Test acc</th>
                  <th className="right sortable" onClick={() => setSort('best_val_loss')}>Val loss</th>
                  <th className="right">Epochs</th>
                  <th className="right sortable" onClick={() => setSort('duration')}>Time</th>
                  <th className="right sortable" onClick={() => setSort('created_at')}>When</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r, i) => (
                  <tr key={r.id} className={`clickable ${selected === r.id ? 'selected' : ''}`} onClick={() => setSelected(selected === r.id ? null : r.id)}>
                    <td className={`rank r${i + 1}`}>{ranked && i < 3 && r.summary.best_val_acc != null ? MEDALS[i] : i + 1}</td>
                    <td>
                      <div style={{ fontWeight: 600 }}>{r.name}</div>
                      <div className="row" style={{ gap: 4, marginTop: 2 }}>
                        {r.status !== 'completed' && <span className={`pill ${r.status === 'running' ? 'accent' : r.status === 'failed' ? 'bad' : 'warn'}`}>{r.status}</span>}
                        {r.study?.best && <span className="pill accent" title="Best trial of an Optuna study">★ optuna best</span>}
                        {r.study && !r.study.best && <span className="pill">trial {r.study.trial}</span>}
                        {r.summary.reproduced === true && <span className="pill ok" title="Retrained from the saved config and matched the original bit-for-bit">✓ reproduced</span>}
                        {r.summary.reproduced === false && <span className="pill bad" title="Retrained from the saved config but results differ">≠ not reproduced</span>}
                      </div>
                    </td>
                    {ds === 'all' && <td className="small muted">{datasets.find((d) => d.id === r.dataset)?.name ?? r.dataset}</td>}
                    <td><Chips types={r.block_types} specs={specs} /></td>
                    <td className="num">{fmtNum(r.summary.params)}</td>
                    <td>
                      <div className="row">
                        <span className="mono" style={{ width: 62, fontWeight: 650 }}>{fmtPct(r.summary.best_val_acc)}</span>
                        <div className="bar grow"><div style={{ width: `${((r.summary.best_val_acc ?? 0) / maxAcc) * 100}%` }} /></div>
                      </div>
                    </td>
                    <td className="num">{fmtPct(r.summary.test_acc)}</td>
                    <td className="num">{fmtLoss(r.summary.best_val_loss)}</td>
                    <td className="num">{r.summary.epochs_done ?? 0}/{String(r.training.epochs)}</td>
                    <td className="num">{fmtDur(r.summary.duration)}</td>
                    <td className="num muted">{fmtAgo(r.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        {selected && <RunDetailPanel id={selected} onClose={() => setSelected(null)} onChanged={load} />}
      </div>
    </div>
  )
}

function RunDetailPanel({ id, onClose, onChanged }: { id: string; onClose: () => void; onChanged: () => void }) {
  const { blocks: specs, datasets, loadConfig, setTab, attachJob, toast } = useStore()
  const [run, setRun] = useState<RunDetail | null>(null)
  const [epochs, setEpochs] = useState(0)
  const [showJson, setShowJson] = useState(false)
  const [name, setName] = useState('')

  useEffect(() => {
    setRun(null)
    api.run(id).then((r) => {
      setRun(r)
      setName(r.name)
      setEpochs((r.summary.epochs_done ?? 0) + Math.max(1, Math.ceil((r.config.training.epochs as number) / 2)))
    })
  }, [id])

  if (!run) return <div className="panel panel-body"><span className="spinner" /></div>
  const ds = datasets.find((d) => d.id === run.config.dataset.id)
  const s = run.summary
  const env = run.config.environment as Record<string, unknown> | undefined

  const act = async (fn: () => Promise<unknown>, msg?: string) => {
    try {
      await fn()
      if (msg) toast(msg, 'ok')
      onChanged()
    } catch (e) {
      toast((e as Error).message, 'error')
    }
  }

  return (
    <div className="panel" style={{ position: 'sticky', top: 0, maxHeight: 'calc(100vh - 110px)', overflow: 'auto' }}>
      <div className="panel-head">
        <input value={name} onChange={(e) => setName(e.target.value)} onBlur={() => name !== run.name && act(() => api.renameRun(run.id, name), 'Renamed')} style={{ fontWeight: 650, flex: 1 }} />
        <button className="ghost" onClick={onClose}>✕</button>
      </div>
      <div className="panel-body col" style={{ gap: 14 }}>
        <div className="row wrap small muted">
          <span className={`pill ${run.status === 'completed' ? 'ok' : run.status === 'running' ? 'accent' : 'warn'}`}>{run.status}</span>
          <span>{ds?.name}</span>·<span className="mono">{run.id}</span>
        </div>
        {run.error && <div className="error-box">{run.error}</div>}
        <div className="kpis">
          <div className="kpi"><div className="k">Best val acc</div><div className="v ok">{fmtPct(s.best_val_acc)}</div></div>
          <div className="kpi"><div className="k">Test acc</div><div className="v">{fmtPct(s.test_acc)}</div></div>
          <div className="kpi"><div className="k">Test loss</div><div className="v">{fmtLoss(s.test_loss)}</div></div>
          <div className="kpi"><div className="k">Params</div><div className="v">{fmtNum(s.params)}</div></div>
        </div>
        {s.reproduce_diff && (
          <div className={s.reproduced ? 'note-box' : 'error-box'}>
            {s.reproduced ? '✓ Retrained from scratch and matched the original run exactly.' : '≠ Retrained, but metrics differ from the original:'}
            {!s.reproduced && Object.entries(s.reproduce_diff).map(([k, [a, b]]) => <div key={k} className="mono">{k}: {a} → {b}</div>)}
          </div>
        )}

        <div className="row wrap">
          <button className="primary" onClick={() => { loadConfig(run.config); setTab('build'); toast(`Loaded “${run.name}” into the builder`, 'ok') }}>Open in builder</button>
          <button onClick={() => act(async () => { const r = await api.retrain(run.id); attachJob(r.job.id, 'train', run.name, r.run.id); r.notes.forEach((n) => toast(n)); if (r.env_diffs.length) toast(`Environment differs: ${r.env_diffs.join('; ')}`, 'error'); setTab('train') }, 'Retraining from scratch with the identical config')}
            title="Train the exact same config again and check the result matches">
            Retrain from scratch
          </button>
          <a className="btn" href={`/api/runs/${run.id}/config`} download>Download config</a>
          {run.has_checkpoint && <a className="btn" href={`/api/runs/${run.id}/checkpoint?which=best`} download>Weights (.pt)</a>}
        </div>
        {run.has_checkpoint && run.status !== 'running' && (
          <div className="row">
            <button onClick={() => act(async () => { const r = await api.resume(run.id, epochs); attachJob(r.job.id, 'train', run.name, run.id); setTab('train') }, 'Resuming from checkpoint')}>
              Resume from checkpoint
            </button>
            <span className="small muted">continue to</span>
            <input type="number" style={{ width: 70 }} value={epochs} min={(s.epochs_done ?? 0) + 1} onChange={(e) => setEpochs(parseInt(e.target.value, 10))} />
            <span className="small muted">epochs (has {s.epochs_done ?? 0})</span>
          </div>
        )}

        {run.metrics.steps.length > 0 && <LiveCharts steps={run.metrics.steps} epochs={run.metrics.epochs} showLr={false} height={170} task={ds?.task} stacked />}

        <div className="row" style={{ alignItems: 'flex-start', gap: 18 }}>
          <div>
            <h4 style={{ marginBottom: 10 }}>Architecture</h4>
            <ArchView blocks={run.config.architecture.blocks} specs={specs} dataset={ds} width={230} />
          </div>
          <div className="grow">
            <h4 style={{ marginBottom: 6 }}>Hyperparameters</h4>
            <table className="tbl small">
              <tbody>
                {Object.entries(run.config.training).filter(([, v]) => v !== null).map(([k, v]) => (
                  <tr key={k}><td className="muted">{k}</td><td className="mono">{String(v)}</td></tr>
                ))}
                {run.config.dataset.train_subset && <tr><td className="muted">train_subset</td><td className="mono">{run.config.dataset.train_subset}</td></tr>}
              </tbody>
            </table>
          </div>
        </div>

        {env && (
          <div>
            <h4 style={{ marginBottom: 6 }}>Environment</h4>
            <div className="small mono muted" style={{ lineHeight: 1.7 }}>
              torch {String(env.torch)} · python {String(env.python)} · numpy {String(env.numpy)}<br />
              {String(env.platform)} · {String(env.num_threads)} threads
            </div>
          </div>
        )}

        <div>
          <button className="sm ghost" onClick={() => setShowJson(!showJson)}>{showJson ? '▾' : '▸'} Full run config (JSON)</button>
          {showJson && <pre className="json" style={{ marginTop: 6 }}>{JSON.stringify(run.config, null, 2)}</pre>}
        </div>

        <div className="row">
          <span className="spacer" />
          <button className="danger sm" onClick={() => confirm(`Delete run “${run.name}” and its checkpoints?`) && act(async () => { await api.deleteRun(run.id); onClose() }, 'Deleted')}>
            Delete run
          </button>
        </div>
      </div>
    </div>
  )
}
