import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, streamJob } from '../api'
import { ArchView, Chips } from '../components/ArchView'
import { LiveCharts } from '../components/LiveCharts'
import { NumInput } from '../components/NumInput'
import { TensorImage } from '../components/TensorImage'
import { GuideButton } from '../learn/GuideButton'
import { useLearn } from '../learn/LearnContext'
import { useStore } from '../store'
import type { Combined, DatasetInfo, HubModel, HubStatus, RunDetail, RunRow, ShiftInfo, ShiftScores } from '../types'
import { fmtAgo, fmtDur, fmtLoss, fmtNum, fmtPct, MEDALS } from '../util'

type SortKey = 'best_val_acc' | 'test_acc' | 'shift_acc' | 'best_val_loss' | 'params' | 'duration' | 'created_at'
const SORTS: { key: SortKey; label: string; desc: boolean }[] = [
  { key: 'best_val_acc', label: 'Best val acc', desc: true },
  { key: 'test_acc', label: 'Test acc', desc: true },
  { key: 'shift_acc', label: 'Shifted test acc', desc: true },
  { key: 'best_val_loss', label: 'Val loss', desc: false },
  { key: 'params', label: 'Params', desc: false },
  { key: 'duration', label: 'Train time', desc: false },
  { key: 'created_at', label: 'Newest', desc: true },
]

const COMBINE_ICON: Record<string, string> = { soup: '🍲', ensemble: '👥', monosoup: '🥄' }
const combineLabel = (c: NonNullable<RunRow['combine']>) =>
  c.kind === 'monosoup' ? 'MonoSoup' : c.kind === 'soup' ? `${c.method === 'greedy' ? 'greedy soup' : 'soup'} of ${c.n}` : `${c.method === 'gate' ? 'mixture of experts' : 'ensemble'} of ${c.n}`

export function Leaderboard() {
  const { blocks: specs, datasets } = useStore()
  const { access } = useLearn()
  const canCombine = access.has('advanced.combine')
  const [picked, setPicked] = useState<string[]>([])
  const [combining, setCombining] = useState(false)
  const [view, setView] = useState<'runs' | 'models'>(() => (localStorage.getItem('lego-ai/lb-view') === 'models' ? 'models' : 'runs'))
  useEffect(() => localStorage.setItem('lego-ai/lb-view', view), [view])
  const [hubOpen, setHubOpen] = useState(false)
  const byModel = view === 'models' && access.has('datasets.custom')
  const [rows, setRows] = useState<RunRow[]>([])
  // Remember the dataset tab and sort order between visits.
  const [ds, setDs] = useState<string>(() => localStorage.getItem('lego-ai/lb-dataset') ?? 'all')
  const [sort, setSort] = useState<SortKey>(() => (localStorage.getItem('lego-ai/lb-sort') as SortKey) ?? 'best_val_acc')
  useEffect(() => {
    localStorage.setItem('lego-ai/lb-dataset', ds)
    localStorage.setItem('lego-ai/lb-sort', sort)
  }, [ds, sort])
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

  /** Stats of a seed group when it is collapsed into a single row (null otherwise). */
  const grouped = useCallback((r: RunRow) => (!showTrials && r.group?.stats && (r.group.completed ?? 0) > 1 ? r.group.stats : null), [showTrials])

  const dsWithRuns = useMemo(() => datasets.filter((d) => rows.some((r) => r.dataset === d.id)), [datasets, rows])

  const filtered = useMemo(() => {
    const s = SORTS.find((x) => x.key === sort)!
    // A seed group shown as one row is ranked by its average, not by its luckiest member.
    const val = (r: RunRow) => (sort === 'created_at' ? r.created_at : sort === 'shift_acc' ? r.summary.shift_acc : (grouped(r)?.[sort]?.mean ?? (r.summary[sort as keyof typeof r.summary] as number | undefined)))
    return rows
      .filter((r) => ds === 'all' || r.dataset === ds)
      .filter((r) => showTrials || !r.study || r.study.best)
      .filter((r) => showTrials || !r.group || r.group.index === 0)
      .filter((r) => showTrials || !r.variation)
      .filter((r) => showFailed || !['failed', 'pruned'].includes(r.status))
      .filter((r) => !q || r.name.toLowerCase().includes(q.toLowerCase()))
      .sort((a, b) => {
        const va = val(a), vb = val(b)
        if (va == null && vb == null) return b.created_at - a.created_at
        if (va == null) return 1
        if (vb == null) return -1
        return s.desc ? vb - va : va - vb
      })
  }, [rows, ds, sort, showTrials, showFailed, q, grouped])

  // "97.8% ± 0.2" for a collapsed seed group, a plain percentage otherwise.
  const pm = (r: RunRow, k: 'best_val_acc' | 'test_acc') => {
    const g = grouped(r)?.[k]
    return g?.mean != null ? `${(g.mean * 100).toFixed(2)}% ± ${((g.std ?? 0) * 100).toFixed(2)}` : fmtPct(r.summary[k])
  }
  const maxAcc = Math.max(0.0001, ...filtered.map((r) => grouped(r)?.best_val_acc?.mean ?? r.summary.best_val_acc ?? 0))
  const ranked = sort === 'best_val_acc' || sort === 'test_acc' || sort === 'best_val_loss' || sort === 'shift_acc'
  const anyShift = filtered.some((r) => r.summary.shift_acc != null)

  return (
    <div className="page">
      <div className="row wrap" style={{ marginBottom: 14, gap: 12 }}>
        <h2>Leaderboard</h2>
        <GuideButton lesson="app.leaderboard" />
        <div className="seg">
          <button className={ds === 'all' ? 'on' : ''} onClick={() => setDs('all')}>All</button>
          {dsWithRuns.map((d) => <button key={d.id} className={ds === d.id ? 'on' : ''} onClick={() => setDs(d.id)}>{d.name}</button>)}
        </div>
        {access.has('datasets.custom') && (
          <div className="seg" title="By model: one row per model design, with its score on every dataset it was trained on and the average">
            <button className={!byModel ? 'on' : ''} onClick={() => setView('runs')}>Runs</button>
            <button className={byModel ? 'on' : ''} onClick={() => setView('models')}>By model</button>
          </div>
        )}
        <span className="spacer" />
        {access.has('hub') && <button onClick={() => setHubOpen(true)}>🤗 Hugging Face</button>}
        <input placeholder="Search…" value={q} onChange={(e) => setQ(e.target.value)} style={{ width: 160 }} />
        <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)}>
          {SORTS.map((s) => <option key={s.key} value={s.key}>Sort: {s.label}</option>)}
        </select>
        {canCombine && picked.length > 0 && <>
          <button className="primary" disabled={picked.length < 2} title={picked.length < 2 ? 'Tick at least two models' : ''} onClick={() => setCombining(true)}>🍲 Combine {picked.length} models…</button>
          <button className="ghost sm" onClick={() => setPicked([])}>Clear</button>
        </>}
        <label className="row small muted"><input type="checkbox" checked={showTrials} onChange={(e) => setShowTrials(e.target.checked)} /> all trials & seed runs</label>
        <label className="row small muted"><input type="checkbox" checked={showFailed} onChange={(e) => setShowFailed(e.target.checked)} /> failed/pruned</label>
      </div>

      <div className={`lb-layout ${selected ? 'with-detail' : ''}`}>
        <div className="panel lb-table" data-tour="leaderboard">
          {byModel ? (
            <ModelTable rows={rows.filter((r) => !q || r.name.toLowerCase().includes(q.toLowerCase()))} datasets={datasets} dsFilter={ds} selected={selected} onSelect={(id) => setSelected(selected === id ? null : id)} />
          ) : filtered.length === 0 ? (
            <div className="empty" style={{ margin: 20 }}>No runs yet. Build a model and train it, and it will show up here.</div>
          ) : (
            <table className="tbl">
              <thead>
                <tr>
                  {canCombine && <th style={{ width: 24 }} title="Tick models to combine them into a soup or an ensemble" />}
                  <th style={{ width: 36 }}>#</th>
                  <th>Model</th>
                  {ds === 'all' && <th>Dataset</th>}
                  <th>Architecture</th>
                  <th className="right sortable" onClick={() => setSort('params')}>Params</th>
                  <th className="sortable" onClick={() => setSort('best_val_acc')}>Best val acc</th>
                  <th className="right sortable" onClick={() => setSort('test_acc')}>Test acc</th>
                  {anyShift && <th className="right sortable" onClick={() => setSort('shift_acc')} title="Average accuracy on shifted test data: the same kind of images made noisy, blurred, rotated and so on, plus a second real dataset where one exists">Shifted</th>}
                  <th className="right sortable" onClick={() => setSort('best_val_loss')}>Val loss</th>
                  <th className="right">Epochs</th>
                  <th className="right sortable" onClick={() => setSort('duration')}>Time</th>
                  <th className="right sortable" onClick={() => setSort('created_at')}>When</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r, i) => (
                  <tr key={r.id} className={`clickable ${selected === r.id ? 'selected' : ''}`} onClick={() => setSelected(selected === r.id ? null : r.id)}>
                    {canCombine && (
                      <td onClick={(e) => e.stopPropagation()}>
                        {r.has_checkpoint && r.status !== 'running' && !grouped(r) && (
                          <input type="checkbox" checked={picked.includes(r.id)} onChange={(e) => setPicked((p) => (e.target.checked ? [...p, r.id] : p.filter((x) => x !== r.id)))} title="Tick to combine with other models" />
                        )}
                      </td>
                    )}
                    <td className={`rank r${i + 1}`}>{ranked && i < 3 && r.summary.best_val_acc != null ? MEDALS[i] : i + 1}</td>
                    <td>
                      <div style={{ fontWeight: 600 }}>{grouped(r) ? r.group!.name : r.name}</div>
                      <div className="row" style={{ gap: 4, marginTop: 2 }}>
                        {r.status !== 'completed' && <span className={`pill ${r.status === 'running' ? 'accent' : r.status === 'failed' ? 'bad' : 'warn'}`}>{r.status}</span>}
                        {r.study?.best && <span className="pill accent" title={r.study.kind === 'autopilot' ? 'The best model the autopilot found' : 'Best trial of an Optuna study'}>★ {r.study.kind === 'autopilot' ? 'autopilot' : 'optuna'} best</span>}
                        {r.combine && <span className="pill blue" title="Made by combining trained models">{COMBINE_ICON[r.combine.kind]} {combineLabel(r.combine)}</span>}
                        {r.scale && <span className="pill" title="One point of a scaling study (see the Scaling tab)">📈 {r.scale.axis === 'width' ? `${r.scale.factor}× width` : `${(r.scale.factor * 100).toFixed(0)}% data`}</span>}
                        {r.suite && <span className="pill" title={`Trained on ${r.suite.n} datasets in one go. Switch to “By model” to compare them.`}>🎒 {r.suite.index + 1} of {r.suite.n}</span>}
                        {r.hub && <span className="pill blue" title={r.hub.repo}>🤗 {r.hub.imported_at ? 'imported' : 'published'}</span>}
                        {r.init_from && <span className="pill" title={`Started from the trained weights of “${r.init_from.name ?? r.init_from.run_id}”`}>fine-tuned</span>}
                        {r.group && (grouped(r)
                          ? <span className="pill blue" title={`Average of ${r.group.completed} runs with seeds ${r.group.seeds.join(', ')}. Tick “all trials & seed runs” to see each one.`}>× {r.group.completed} seeds</span>
                          : <span className="pill">seed {r.group.seeds[r.group.index]}</span>)}
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
                        <span className="mono nowrap" style={{ minWidth: 62, fontWeight: 650 }}>{pm(r, 'best_val_acc')}</span>
                        <div className="bar grow"><div style={{ width: `${((grouped(r)?.best_val_acc?.mean ?? r.summary.best_val_acc ?? 0) / maxAcc) * 100}%` }} /></div>
                      </div>
                    </td>
                    <td className="num">{pm(r, 'test_acc')}</td>
                    {anyShift && <td className="num">{grouped(r) ? '-' : fmtPct(r.summary.shift_acc)}</td>}
                    <td className="num">{fmtLoss(r.summary.best_val_loss)}</td>
                    <td className="num">{r.combine ? '-' : `${r.summary.epochs_done ?? 0}/${String(r.training.epochs)}`}</td>
                    <td className="num">{fmtDur(r.summary.duration)}</td>
                    <td className="num muted">{fmtAgo(r.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        {selected && <RunDetailPanel id={selected} onClose={() => setSelected(null)} onChanged={load} onMade={(id) => { load(); setSelected(id) }} />}
      </div>
      {hubOpen && <HubModal onClose={() => setHubOpen(false)} onImported={(id) => { setHubOpen(false); setView('runs'); load(); setSelected(id) }} />}
      {combining && (
        <CombineModal runs={rows.filter((r) => picked.includes(r.id))} onClose={() => setCombining(false)}
          onDone={(id) => { setCombining(false); setPicked([]); setShowTrials(false); load(); setSelected(id) }} />
      )}
    </div>
  )
}

/** One row per model design: its best score on each dataset, and the average. */
function ModelTable({ rows, datasets, dsFilter, selected, onSelect }: { rows: RunRow[]; datasets: DatasetInfo[]; dsFilter: string; selected: string | null; onSelect: (id: string) => void }) {
  const { blocks: specs } = useStore()
  const [kind, setKind] = useState<'all' | 'image' | 'tokens' | 'vector'>('all')
  const kindOf = (id: string) => datasets.find((d) => d.id === id)?.input_kind
  const usable = rows.filter((r) => r.status === 'completed' && r.summary.best_val_acc != null && !r.combine && !r.variation && (kind === 'all' || kindOf(r.dataset) === kind))
  // Columns: every dataset any model was trained on (the dataset tabs narrow it to one).
  const cols = datasets.filter((d) => usable.some((r) => r.dataset === d.id) && (dsFilter === 'all' || d.id === dsFilter))
  const groups = new Map<string, RunRow[]>()
  usable.forEach((r) => groups.set(r.arch, [...(groups.get(r.arch) ?? []), r]))
  const table = [...groups.values()].map((rs) => {
    const best: Record<string, RunRow> = {}
    rs.forEach((r) => { if (!best[r.dataset] || (r.summary.best_val_acc ?? 0) > (best[r.dataset].summary.best_val_acc ?? 0)) best[r.dataset] = r })
    const have = cols.filter((c) => best[c.id])
    const avg = have.length ? have.reduce((a, c) => a + (best[c.id].summary.best_val_acc ?? 0), 0) / have.length : null
    // Name the group after its most common run name.
    const names = new Map<string, number>()
    rs.forEach((r) => { const n = (r.suite?.name ?? r.group?.name ?? r.name).replace(/ · (seed|trial|autopilot) \d+$/, ''); names.set(n, (names.get(n) ?? 0) + 1) })
    const name = [...names.entries()].sort((a, b) => b[1] - a[1])[0][0]
    return { key: rs[0].arch, name, best, have: have.length, avg, runs: rs.length, types: rs[0].block_types, params: Object.values(best).map((r) => r.summary.params ?? 0) }
  }).filter((g) => g.have > 0)
  // Models tested on more datasets come first: an average over one dataset is not comparable with one over four.
  table.sort((a, b) => b.have - a.have || (b.avg ?? 0) - (a.avg ?? 0))
  const top: Record<string, number> = {}
  cols.forEach((c) => { top[c.id] = Math.max(...table.map((g) => g.best[c.id]?.summary.best_val_acc ?? 0)) })
  return (
    <>
      <div className="row small" style={{ padding: '8px 10px', gap: 10 }}>
        <span className="muted">Each row is one model design (the same blocks with the same settings). Cells show its best validation accuracy on that dataset.</span>
        <span className="spacer" />
        <div className="seg">
          {([['all', 'All data'], ['image', '🖼️ Images'], ['tokens', '📝 Text'], ['vector', '🔢 Tables']] as const).map(([k, l]) => <button key={k} className={kind === k ? 'on' : ''} onClick={() => setKind(k)}>{l}</button>)}
        </div>
      </div>
      {table.length === 0 ? <div className="empty" style={{ margin: 20 }}>No finished runs to compare yet.</div> : (
        <table className="tbl">
          <thead>
            <tr>
              <th style={{ width: 36 }}>#</th><th>Model</th><th>Architecture</th><th className="right">Params</th>
              <th className="right" title="Average of the best validation accuracy over the datasets shown that this model was trained on">Average</th>
              <th className="right" title="How many of the datasets shown this model has been trained on">Datasets</th>
              {cols.map((c) => <th key={c.id} className="right">{c.name}</th>)}
            </tr>
          </thead>
          <tbody>
            {table.map((g, i) => (
              <tr key={g.key}>
                <td className={`rank r${i + 1}`}>{i < 3 && g.have === table[0].have && g.have > 1 ? MEDALS[i] : i + 1}</td>
                <td><div style={{ fontWeight: 600 }}>{g.name}</div><div className="tiny muted">{g.runs} run{g.runs === 1 ? '' : 's'}</div></td>
                <td><Chips types={g.types} specs={specs} /></td>
                <td className="num">{Math.min(...g.params) === Math.max(...g.params) ? fmtNum(g.params[0]) : `${fmtNum(Math.min(...g.params))} to ${fmtNum(Math.max(...g.params))}`}</td>
                <td className="num" style={{ fontWeight: 700 }}>{fmtPct(g.avg)}</td>
                <td className="num muted">{g.have} of {cols.length}</td>
                {cols.map((c) => {
                  const r = g.best[c.id]
                  return (
                    <td key={c.id} className={`num ${r ? 'clickable-cell' : ''} ${r && selected === r.id ? 'cell-selected' : ''}`} onClick={() => r && onSelect(r.id)} title={r ? `${r.name}: open this run` : 'Not trained on this dataset'}>
                      {r ? <span className={r.summary.best_val_acc === top[c.id] ? 'ok' : ''} style={r.summary.best_val_acc === top[c.id] ? { fontWeight: 700 } : undefined}>{fmtPct(r.summary.best_val_acc)}</span> : <span className="faint">-</span>}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  )
}

/** Log in to Hugging Face and import models other people published from Lego AI. */
function HubModal({ onClose, onImported }: { onClose: () => void; onImported: (runId: string) => void }) {
  const { toast, refreshBlocks } = useStore()
  const { flag } = useLearn()
  const [st, setSt] = useState<HubStatus | null>(null)
  const [token, setToken] = useState('')
  const [repo, setRepo] = useState('')
  const [found, setFound] = useState<HubModel[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [code, setCode] = useState<Record<string, string> | null>(null)
  useEffect(() => { api.hubStatus().then(setSt).catch(() => setSt({ logged_in: false, user: null })) }, [])
  const go = async (fn: () => Promise<void>) => { setBusy(true); setErr(null); try { await fn() } catch (e) { setErr((e as Error).message) } finally { setBusy(false) } }
  const doImport = (allow = false) => go(async () => {
    const r = await api.hubImport(repo.trim(), allow)
    if (r.needs_confirm) { setCode(r.custom_blocks ?? {}); return }
    await refreshBlocks()
    flag('hubImport')
    toast(`Imported “${r.name}”`, 'ok')
    onImported(r.run_id!)
  })
  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal" style={{ width: 'min(720px, 100%)' }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-head"><h2>🤗 Hugging Face</h2><GuideButton lesson="app.hub" /><span className="spacer" /><button className="ghost" onClick={onClose}>✕</button></div>
        <div className="modal-body col" style={{ gap: 16 }}>
          {err && <div className="error-box">{err}</div>}
          <div className="col" style={{ gap: 8 }}>
            <h4>Your account</h4>
            {!st ? <span className="spinner" /> : st.logged_in ? (
              <div className="row"><span className="grow">Logged in as <b>{st.user}</b>. You can publish a model from its page on the leaderboard.</span><button className="sm" disabled={busy} onClick={() => go(async () => setSt(await api.hubLogout()))}>Log out</button></div>
            ) : <>
              <div className="small muted">Importing public models needs no account. To publish your own, create a free account at huggingface.co, then make an access token with <b>write</b> permission at <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">huggingface.co/settings/tokens ↗</a> and paste it here.</div>
              <div className="row">
                <input className="grow" type="password" autoComplete="off" placeholder="hf_…" value={token} onChange={(e) => setToken(e.target.value)} />
                <button className="primary" disabled={busy || !token.trim()} onClick={() => go(async () => { setSt(await api.hubLogin(token)); setToken('') })}>Log in</button>
              </div>
              <div className="tiny faint">The token is checked and then stored on this computer by the official Hugging Face library, the same place its command-line tools use. A token is like a password: never share it.</div>
              {st.error && <div className="small warn">{st.error}</div>}
            </>}
          </div>
          <div className="col" style={{ gap: 8 }}>
            <h4>Import a model</h4>
            <div className="small muted">Models published from Lego AI carry their full recipe and weights, so they arrive on your leaderboard ready to test in Explore, retrain or fine-tune.</div>
            <div className="row">
              <input className="grow" placeholder="owner/name" value={repo} onChange={(e) => { setRepo(e.target.value); setCode(null) }} onKeyDown={(e) => e.key === 'Enter' && repo.trim() && doImport()} />
              <button className="primary" disabled={busy || !repo.trim()} onClick={() => doImport()}>Import</button>
              <button disabled={busy} onClick={() => go(async () => setFound(await api.hubModels(repo.includes('/') ? '' : repo)))}>Browse published models</button>
            </div>
            {code && (
              <div className="error-box">
                <b>This model contains custom block code.</b> Importing it will run this Python on your computer. Only continue if you trust the author and have read it.
                {Object.entries(code).map(([f, src]) => <pre key={f} className="json" style={{ maxHeight: 200, marginTop: 6 }}>{`# ${f}\n${src}`}</pre>)}
                <div className="row" style={{ marginTop: 6 }}><button className="danger sm" disabled={busy} onClick={() => doImport(true)}>I trust this code: import it</button><button className="sm" onClick={() => setCode(null)}>Cancel</button></div>
              </div>
            )}
            {found && (found.length === 0 ? <div className="small muted">No published Lego AI models found{repo && !repo.includes('/') ? ` matching “${repo}”` : ''} yet. Be the first: publish one of yours.</div> : (
              <table className="tbl small">
                <tbody>{found.map((m) => <tr key={m.repo} className="clickable" onClick={() => { setRepo(m.repo); setCode(null) }}><td className="mono">{m.repo}</td><td className="num muted">{m.downloads ?? 0} downloads</td><td className="num muted">{m.updated}</td></tr>)}</tbody>
              </table>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

/** Publish one run to the Hugging Face Hub. */
function Publish({ run, onChanged }: { run: RunDetail; onChanged: () => void }) {
  const { flag } = useLearn()
  const [st, setSt] = useState<HubStatus | null>(null)
  const [name, setName] = useState(() => `lego-${run.name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 50) || 'model'}`)
  const [priv, setPriv] = useState(false)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [url, setUrl] = useState<string | null>(run.hub?.published_at ? run.hub.url : null)
  useEffect(() => { api.hubStatus().then(setSt).catch(() => setSt(null)) }, [])
  const go = async () => {
    setBusy(true); setErr(null)
    try { const r = await api.publish(run.id, name, priv); setUrl(r.url); flag('publish'); onChanged() } catch (e) { setErr((e as Error).message) } finally { setBusy(false) }
  }
  return (
    <div className="col" style={{ gap: 6 }}>
      <h4>🤗 Publish to Hugging Face</h4>
      {url && <div className="small">Published at <a href={url} target="_blank" rel="noreferrer">{url.replace('https://', '')} ↗</a></div>}
      {st && !st.logged_in ? <div className="small muted">Log in first: press <b>🤗 Hugging Face</b> at the top of the leaderboard.</div> : (
        <div className="row wrap small">
          <span className="muted">{st?.user ?? '…'} /</span>
          <input style={{ width: 210 }} value={name} onChange={(e) => setName(e.target.value.replace(/[^A-Za-z0-9._-]/g, '-'))} />
          <label className="row muted" title="Only you can see a private model"><input type="checkbox" checked={priv} onChange={(e) => setPriv(e.target.checked)} /> private</label>
          <button disabled={busy || !st?.logged_in || !name} onClick={go}>{busy ? 'Uploading…' : url ? 'Publish again' : 'Publish'}</button>
        </div>
      )}
      {err && <div className="error-box">{err}</div>}
      <div className="tiny faint">Uploads the recipe (config), the trained weights, a model card with its scores and, for built-in datasets, a standalone training script. {priv ? '' : 'Anyone will be able to see and download it.'}</div>
    </div>
  )
}

const RECIPES: { id: string; title: string; icon: string; what: string; needs: string }[] = [
  { id: 'soup', icon: '🍲', title: 'Soup', what: 'Average the weights of all of them into one model. Costs the same to run as a single model.', needs: 'same' },
  { id: 'greedy', icon: '🥣', title: 'Greedy soup', what: 'Start with the best model and add the others one at a time, keeping each only if the soup does not get worse.', needs: 'same' },
  { id: 'ensemble', icon: '👥', title: 'Ensemble', what: 'Keep every model, ask them all, and average their answers. Any mix of architectures works.', needs: 'any' },
  { id: 'gate', icon: '🧑‍⚖️', title: 'Mixture of experts', what: 'Like an ensemble, plus a small trained gate that decides how much to trust each expert for each input.', needs: 'any' },
]

/** Run a combine job and wait for its result. */
function useCombine(onFlag: (kind: string) => void) {
  const [state, setState] = useState<{ busy: boolean; message: string; result: Combined | null; error: string | null }>({ busy: false, message: '', result: null, error: null })
  const run = useCallback(async (kind: string, ids: string[]) => {
    setState({ busy: true, message: 'Starting…', result: null, error: null })
    try {
      const { job } = await api.combine(kind, ids)
      streamJob(job.id, (e) => {
        if (e.type === 'status') setState((s) => ({ ...s, message: e.message as string }))
        if (e.type === 'combined') { setState((s) => ({ ...s, result: e as unknown as Combined })); onFlag(e.kind as string) }
        if (e.type === 'error') setState((s) => ({ ...s, error: e.message as string }))
      }, () => setState((s) => ({ ...s, busy: false })))
    } catch (e) {
      setState({ busy: false, message: '', result: null, error: (e as Error).message })
    }
  }, [onFlag])
  return { ...state, run }
}

function CombineResult({ c }: { c: Combined }) {
  const up = c.val_acc >= c.best_member
  return (
    <div className="col" style={{ gap: 10 }}>
      <div className="kpis">
        <div className="kpi"><div className="k">Val acc</div><div className={`v ${up ? 'ok' : 'bad'}`}>{fmtPct(c.val_acc)}</div></div>
        <div className="kpi"><div className="k">{c.kind === 'monosoup' ? 'Original model' : 'Best single model'}</div><div className="v">{fmtPct(c.best_member)}</div></div>
        <div className="kpi"><div className="k">Test acc</div><div className="v">{fmtPct(c.test_acc)}</div></div>
        {c.shift_acc != null && <div className="kpi"><div className="k">Shifted test</div><div className="v">{fmtPct(c.shift_acc)}</div>{c.shift_before != null && <div className={`tiny ${c.shift_acc >= c.shift_before ? 'ok' : 'bad'}`}>was {fmtPct(c.shift_before)}</div>}</div>}
      </div>
      <div className={up ? 'note-box' : 'error-box'}>{c.note}</div>
    </div>
  )
}

function CombineModal({ runs, onClose, onDone }: { runs: RunRow[]; onClose: () => void; onDone: (id: string) => void }) {
  const { flag } = useLearn()
  const [kind, setKind] = useState('soup')
  const job = useCombine(useCallback((k: string) => flag(k), [flag]))
  const sameData = new Set(runs.map((r) => r.dataset)).size === 1
  const sameArch = new Set(runs.map((r) => JSON.stringify(r.block_types))).size === 1
  const sameStart = sameArch && new Set(runs.map((r) => r.init_from?.run_id ?? `seed ${String(r.training.seed)}`)).size === 1
  const hasEnsemble = runs.some((r) => r.combine?.kind === 'ensemble')
  const recipe = RECIPES.find((x) => x.id === kind)!
  const blocked = !sameData ? 'These models were trained on different datasets.' : recipe.needs === 'same' && !sameArch ? 'A soup needs models with exactly the same architecture. Choose an ensemble instead.' : recipe.needs === 'same' && hasEnsemble ? 'An ensemble has no single set of weights to average.' : null
  return (
    <div className="modal-back" onClick={() => !job.busy && onClose()}>
      <div className="modal" style={{ width: 'min(760px, 100%)' }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2>Combine {runs.length} models</h2>
          <GuideButton lesson="adv.soup" />
          <span className="spacer" />
          <button className="ghost" disabled={job.busy} onClick={onClose}>✕</button>
        </div>
        <div className="modal-body col" style={{ gap: 14 }}>
          <table className="tbl small">
            <tbody>{runs.map((r) => <tr key={r.id}><td>{r.name}</td><td className="num">{fmtNum(r.summary.params)}</td><td className="num">{fmtPct(r.summary.best_val_acc)}</td></tr>)}</tbody>
          </table>
          {job.result ? <>
            <CombineResult c={job.result} />
            <div className="row"><span className="spacer" /><button className="primary" onClick={() => onDone(job.result!.run_id)}>Show “{job.result.name}” →</button></div>
          </> : <>
            <div className="recipe-grid">
              {RECIPES.map((x) => (
                <div key={x.id} className={`template-card ${kind === x.id ? 'chosen' : ''}`} onClick={() => !job.busy && setKind(x.id)}>
                  <h3>{x.icon} {x.title}</h3>
                  <div className="small muted">{x.what}</div>
                </div>
              ))}
            </div>
            {blocked && <div className="error-box">{blocked}</div>}
            {!blocked && recipe.needs === 'same' && !sameStart && (
              <div className="note-box" style={{ background: '#2a1a0b', borderColor: '#5a3514', color: '#fed7aa' }}>
                Heads up: these models did not start from the same weights (different seeds). Averaging such models usually gives a far worse one. Try it to see why, then open a model and use <b>Cook a soup</b> for a recipe that works.
              </div>
            )}
            {job.error && <div className="error-box">{job.error}</div>}
            <div className="row">
              {job.busy && <span className="small muted"><span className="spinner" style={{ width: 10, height: 10 }} /> {job.message}</span>}
              <span className="spacer" />
              <button className="primary" disabled={job.busy || !!blocked} onClick={() => job.run(kind, runs.map((r) => r.id))}>{recipe.icon} Make the {recipe.title.toLowerCase()}</button>
            </div>
          </>}
        </div>
      </div>
    </div>
  )
}

/** How a model holds up on shifted test data: one tile per shifted set, with an example of what it looks like. */
function Robustness({ run, onMeasured }: { run: RunDetail; onMeasured: (s: ShiftScores) => void }) {
  const { toast } = useStore()
  const { openLesson } = useLearn()
  const [sets, setSets] = useState<ShiftInfo[] | null>(null)
  const [busy, setBusy] = useState(false)
  const ds = run.config.dataset.id
  useEffect(() => {
    api.datasetShifts(ds, 3).then(setSets).catch(() => setSets([]))
  }, [ds])
  if (!sets?.length) return null
  const sc = run.summary.shifts
  const measure = async () => {
    setBusy(true)
    try { onMeasured(await api.measureShifts(run.id)) } catch (e) { toast((e as Error).message, 'error') } finally { setBusy(false) }
  }
  return (
    <div className="col" style={{ gap: 8 }}>
      <div className="row">
        <h4 className="grow">Shifted test data</h4>
        {sc && <span className="small">average <b>{fmtPct(sc.acc)}</b> <span className="muted">vs {fmtPct(sc.clean)} unchanged</span></span>}
        <button className="ghost sm" title="What is this?" onClick={() => openLesson('adv.robust')}>?</button>
      </div>
      <div className="shift-grid">
        {sets.map((s) => {
          const acc = s.id === 'clean' ? sc?.clean : sc?.each[s.id]
          const drop = sc && acc != null ? sc.clean - acc : 0
          return (
            <div key={s.id} className="shift-tile" title={`${s.description} (${s.n.toLocaleString()} images)`}>
              <TensorImage data={s.image} shape={s.shape} size={52} />
              <div className="tiny muted">{s.name}{s.real && ' ★'}</div>
              <div className={`mono small ${s.id !== 'clean' && drop > 0.1 ? 'bad' : s.id !== 'clean' && drop > 0.03 ? 'warn' : ''}`}>{acc == null ? '-' : fmtPct(acc, 1)}</div>
            </div>
          )
        })}
      </div>
      {!sc && run.has_checkpoint && run.status !== 'running' && (
        <div className="row small"><button disabled={busy} onClick={measure}>{busy ? 'Measuring…' : 'Measure this model'}</button><span className="muted">This run was trained before shifted tests were automatic.</span></div>
      )}
      {sc && <div className="tiny faint">The same test images, changed in ways the model never saw in training. ★ is a different real dataset with the same classes. A big drop means the model learned something narrower than it looks.</div>}
    </div>
  )
}

/** Advanced actions that start from one trained model. */
function AdvancedActions({ run, onMade }: { run: RunDetail; onMade: (id: string) => void }) {
  const { attachJob, setTab, toast, session, loadConfig, setDraft } = useStore()
  const { access, flag } = useLearn()
  const [n, setN] = useState(4)
  const [ep, setEp] = useState(2)
  const [rounds, setRounds] = useState(8)
  const [arch, setArch] = useState(true)
  const mono = useCombine(useCallback((k: string) => flag(k), [flag]))
  const busy = session?.status === 'running'
  const go = async (fn: () => Promise<void>) => { try { await fn() } catch (e) { toast((e as Error).message, 'error') } }
  if (!access.has('advanced.combine') && !access.has('advanced.autopilot')) return null
  return (
    <div className="col" style={{ gap: 8 }}>
      <div className="row"><h4 className="grow">Advanced</h4><GuideButton lesson="adv.soup" /></div>
      {access.has('advanced.combine') && <>
        <div className="row wrap small">
          <button title="Start a new training run from this model’s weights, on the same or a different dataset" onClick={() => {
            loadConfig(run.config)
            setDraft((d) => ({ ...d, name: `${run.name} · fine-tuned`, init_from: { run_id: run.id, name: run.name } }), { undoable: false })
            flag('finetune')
            setTab('train')
            toast('Loaded. Pick the dataset and settings on the Train page: training will start from this model’s weights.', 'ok')
          }}>🎯 Fine-tune…</button>
          <span className="muted">keep training these weights, on this dataset or another one</span>
        </div>
        <div className="row wrap small">
          <button disabled={busy} title="Fine-tune several variations of this model, then average their weights" onClick={() => go(async () => {
            const { job } = await api.cook(run.id, { n, epochs: ep, lr_scale: 0.3 })
            attachJob(job.id, 'cook', `Soup from ${run.name}`)
            setTab('train')
          })}>🍲 Cook a soup</button>
          <span className="muted">from</span><NumInput int style={{ width: 52 }} value={n} min={2} max={8} emptyValue={4} onChange={(v) => setN(v ?? 4)} />
          <span className="muted">variations,</span><NumInput int style={{ width: 52 }} value={ep} min={1} max={50} emptyValue={2} onChange={(v) => setEp(v ?? 2)} />
          <span className="muted">more epochs each</span>
        </div>
        <div className="row wrap small">
          <button disabled={mono.busy} title="“Model soups need only one ingredient”: re-weight the strong and weak directions of this one model’s update" onClick={() => mono.run('monosoup', [run.id])}>🥄 MonoSoup</button>
          <span className="muted">{mono.busy ? <><span className="spinner" style={{ width: 10, height: 10 }} /> {mono.message}</> : 'a soup from this one model, no extra training'}</span>
        </div>
        {mono.error && <div className="error-box">{mono.error}</div>}
        {mono.result && <>
          <CombineResult c={mono.result} />
          <div className="row"><span className="spacer" /><button className="sm" onClick={() => onMade(mono.result!.run_id)}>Show the MonoSoup →</button></div>
        </>}
      </>}
      {access.has('advanced.autopilot') && run.status === 'completed' && !run.combine && (
        <div className="row wrap small">
          <button disabled={busy} title="Keep trying small changes to this model and its training settings, keeping whatever improves validation accuracy" onClick={() => go(async () => {
            const { job } = await api.autopilot(run.id, { rounds, architecture: arch })
            attachJob(job.id, 'autopilot', `Autopilot · ${run.name}`)
            setTab('train')
          })}>🚀 Autopilot</button>
          <span className="muted">for</span><NumInput int style={{ width: 52 }} value={rounds} min={1} max={100} emptyValue={8} onChange={(v) => setRounds(v ?? 8)} />
          <span className="muted">rounds</span>
          <label className="row muted" title="Also try wider or narrower layers and more or fewer repeats"><input type="checkbox" checked={arch} onChange={(e) => setArch(e.target.checked)} /> may change the architecture</label>
        </div>
      )}
      {busy && <div className="tiny faint">Another job is running. Cooking and the autopilot can start when it finishes.</div>}
    </div>
  )
}

function RunDetailPanel({ id, onClose, onChanged, onMade }: { id: string; onClose: () => void; onChanged: () => void; onMade: (id: string) => void }) {
  const { blocks: specs, datasets, loadConfig, setTab, attachJob, toast } = useStore()
  const { access, flag } = useLearn()
  const [run, setRun] = useState<RunDetail | null>(null)
  const [epochs, setEpochs] = useState(0)
  const [showJson, setShowJson] = useState(false)
  const [name, setName] = useState('')

  useEffect(() => {
    setRun(null)
    api.run(id).then((r) => {
      setRun(r)
      setName(r.name)
      // A paused run resumes to its own target; a finished one gets some extra epochs by default.
      const target = r.config.training.epochs as number
      const done = r.summary.epochs_done ?? 0
      setEpochs(done < target ? target : done + Math.max(1, Math.ceil(target / 2)))
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
          {run.status === 'stopped' && s.paused_at != null && <span>paused at epoch {s.paused_at.toFixed(2)}</span>}
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
          {access.has('builder') && <button className="primary" onClick={() => { loadConfig(run.config); setTab('build'); toast(`Loaded “${run.name}” into the builder`, 'ok') }}>Open in builder</button>}
          {access.has('config.io') && run.combine && run.has_checkpoint && run.combine.kind !== 'ensemble' && <a className="btn" href={`/api/runs/${run.id}/checkpoint?which=best`} download>Weights (.pt)</a>}
          {access.has('config.io') && !run.combine && <><button onClick={() => act(async () => { const r = await api.retrain(run.id); attachJob(r.job.id, 'train', run.name, r.run.id); r.notes.forEach((n) => toast(n)); if (r.env_diffs.length) toast(`Environment differs: ${r.env_diffs.join('; ')}`, 'error'); setTab('train') }, 'Retraining from scratch with the identical config')}
            title="Train the exact same config again and check the result matches">
            Retrain from scratch
          </button>
          <a className="btn" href={`/api/runs/${run.id}/config`} download onClick={() => flag('export')}>Download config</a>
          {run.has_checkpoint && <a className="btn" href={`/api/runs/${run.id}/checkpoint?which=best`} download>Weights (.pt)</a>}
          <a className="btn" href={`/api/runs/${run.id}/export`} download onClick={() => flag('exportPython')} title="This model as plain PyTorch: a standalone train.py plus a script to use the trained model">Python scripts</a></>}
        </div>
        {run.combine && (
          <div className="col" style={{ gap: 8 }}>
            <h4>{COMBINE_ICON[run.combine.kind]} {run.combine.kind === 'monosoup' ? 'MonoSoup of' : run.combine.kind === 'soup' ? `${run.combine.method === 'greedy' ? 'Greedy soup' : 'Soup'} made from` : `${run.combine.method === 'gate' ? 'Mixture of experts' : 'Ensemble'} of`}</h4>
            <table className="tbl small">
              <tbody>
                {run.combine.members.map((m) => (
                  <tr key={m.id} style={m.used ? undefined : { opacity: 0.5 }}>
                    <td>{m.name}</td><td className="num">{fmtPct(m.val_acc)}</td><td className="small muted">{m.used ? '' : 'left out'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="note-box">{run.combine.note}</div>
            {run.combine.layers && (
              <details className="small">
                <summary className="muted" style={{ cursor: 'pointer' }}>Per-layer weights ({run.combine.layers.length} layers)</summary>
                <table className="tbl small" style={{ marginTop: 6 }}>
                  <thead><tr><th>Layer</th><th className="right" title="How many directions carry most of the update (its effective rank), out of all of them">strong directions</th><th className="right">kept at</th><th className="right">weak kept at</th></tr></thead>
                  <tbody>{run.combine.layers.map((l) => <tr key={l.layer}><td className="mono">{l.layer}</td><td className="num">{l.rank} of {l.of}</td><td className="num">{(l.lam_high * 100).toFixed(0)}%</td><td className="num">{(l.lam_low * 100).toFixed(0)}%</td></tr>)}</tbody>
                </table>
              </details>
            )}
          </div>
        )}
        {run.config.init_from && (
          <div className="small muted">
            Fine-tuned from the trained weights of “{run.config.init_from.name ?? run.config.init_from.run_id}”.
            {s.init_from && <> {s.init_from.fresh.length === 0 ? 'Every layer was carried over.' : `Carried over everything except ${s.init_from.fresh.length} layer${s.init_from.fresh.length === 1 ? '' : 's'} (${s.init_from.fresh.slice(0, 3).map((f) => f.replace(/^blocks\./, '')).join(', ')}${s.init_from.fresh.length > 3 ? ', …' : ''}), which started fresh.`}</>}
          </div>
        )}
        {run.hub?.imported_at && <div className="small muted">Imported from <a href={run.hub.url} target="_blank" rel="noreferrer">{run.hub.repo} ↗</a>. Its scores are the ones its author measured.</div>}
        {access.has('hub') && run.has_checkpoint && run.status === 'completed' && run.combine?.kind !== 'ensemble' && <Publish key={run.id} run={run} onChanged={onChanged} />}
        <Robustness run={run} onMeasured={(sh) => { setRun((r) => (r ? { ...r, summary: { ...r.summary, shift_acc: sh.acc, shifts: sh } } : r)); onChanged() }} />
        {run.has_checkpoint && run.status !== 'running' && run.combine?.kind !== 'ensemble' && <AdvancedActions run={run} onMade={onMade} />}
        {access.has('train.resume') && run.resumable !== false && run.has_checkpoint && run.status !== 'running' && (
          <div className="row">
            <button onClick={() => act(async () => { const r = await api.resume(run.id, epochs); attachJob(r.job.id, 'train', run.name, run.id, { resumed: true }); setTab('train') }, 'Resuming from checkpoint')}>
              Resume from checkpoint
            </button>
            <span className="small muted">continue to</span>
            <NumInput int style={{ width: 70 }} value={epochs} min={(s.epochs_done ?? 0) + 1} emptyValue={epochs} onChange={(n) => setEpochs(n ?? epochs)} />
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
                {s.train_examples != null && <tr><td className="muted">training examples</td><td className="mono">{s.train_examples.toLocaleString()}{run.config.dataset.train_subset ? ' (subset)' : ''}</td></tr>}
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
