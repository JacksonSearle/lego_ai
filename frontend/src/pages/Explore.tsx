import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import { Difficulty } from '../components/Difficulty'
import { ImageInput } from '../components/ImageInput'
import { NumInput } from '../components/NumInput'
import { TensorImage } from '../components/TensorImage'
import { GuideButton } from '../learn/GuideButton'
import { useLearn } from '../learn/LearnContext'
import { useStore } from '../store'
import type { DatasetInfo, Evaluation, InputPrediction, Prediction, Predictions, RunRow, Samples } from '../types'
import { fmtNum, fmtPct } from '../util'

type Tab = 'data' | 'model' | 'chat'
const CLASS_COLORS = ['#60a5fa', '#fb923c', '#4ade80', '#f472b6', '#a78bfa', '#facc15', '#2dd4bf', '#f87171', '#94a3b8', '#c084fc']

/** Runs whose trained weights can be loaded. */
function useUsableRuns() {
  const [runs, setRuns] = useState<RunRow[]>([])
  const { session } = useStore()
  useEffect(() => {
    api.runs().then((rs) => setRuns(rs.filter((r) => r.has_checkpoint && r.status !== 'running')))
  }, [session?.status])
  return runs
}

export function Explore() {
  const [tab, setTab] = useState<Tab>(() => (sessionStorage.getItem('lego-ai/explore-tab') as Tab) || 'data')
  useEffect(() => sessionStorage.setItem('lego-ai/explore-tab', tab), [tab])
  return (
    <div className="explore">
      <div className="explore-head">
        <h2>Explore</h2>
        <div className="seg" data-tour="explore-tabs">
          <button className={tab === 'data' ? 'on' : ''} onClick={() => setTab('data')}>🗂️ Look at the data</button>
          <button className={tab === 'model' ? 'on' : ''} onClick={() => setTab('model')}>🔬 Test a model</button>
          <button className={tab === 'chat' ? 'on' : ''} onClick={() => setTab('chat')}>💬 Chat</button>
        </div>
        <GuideButton lesson="app.explore" />
      </div>
      <div className="explore-body">
        {tab === 'data' && <DataBrowser />}
        {tab === 'model' && <ModelTester openChat={() => setTab('chat')} />}
        {tab === 'chat' && <Chat />}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Shared bits
// ---------------------------------------------------------------------------
const className = (names: string[] | undefined, c: number | string) => (typeof c === 'number' ? names?.[c] ?? String(c) : c)

function ExampleCard({ ex, names, size = 64 }: { ex: Partial<Prediction>; names?: string[]; size?: number }) {
  const hasPred = ex.pred != null
  const right = hasPred && ex.pred === ex.label
  const conf = ex.top?.[0]?.p
  if (ex.text != null) {
    return (
      <div className={`text-ex ${hasPred ? (right ? 'right' : 'wrong') : ''}`}>
        <div className="row tiny" style={{ marginBottom: 4 }}>
          <span className="pill">{typeof ex.label === 'number' ? className(names, ex.label) : 'continues with…'}</span>
          {hasPred && <span className={right ? 'ok' : 'bad'}>model: {className(names, ex.pred!)} {conf != null && `(${Math.round(conf * 100)}%)`} {right ? '✓' : '✗'}</span>}
        </div>
        <div className="small" style={{ whiteSpace: 'pre-wrap' }}>{ex.text}</div>
        {typeof ex.label === 'string' && <div className="small muted" style={{ whiteSpace: 'pre-wrap', marginTop: 4 }}>→ {ex.label}</div>}
      </div>
    )
  }
  return (
    <div className={`pred ${hasPred ? (right ? 'right' : 'wrong') : ''}`} title={ex.index != null ? `example #${ex.index}` : undefined}>
      {ex.image && ex.shape && <TensorImage data={ex.image} shape={ex.shape} size={size} />}
      {ex.vector && <div className="mono tiny">({ex.vector.map((v) => v.toFixed(2)).join(', ')})</div>}
      {hasPred ? (
        <>
          <div className="small"><b>{className(names, ex.pred!)}</b> {right ? '✓' : <span className="bad">✗</span>}</div>
          {!right && <div className="tiny bad">was {className(names, ex.label!)}</div>}
          {conf != null && <div className="tiny muted">{Math.round(conf * 100)}% sure</div>}
        </>
      ) : <div className="small">{className(names, ex.label!)}</div>}
    </div>
  )
}

function ProbBars({ probs, names, pred }: { probs: number[]; names: string[]; pred: number }) {
  return (
    <div className="prob-bars">
      {probs.map((p, i) => (
        <div key={i} className="prob-row">
          <span className={`small ${i === pred ? '' : 'muted'}`} style={{ fontWeight: i === pred ? 700 : 400 }}>{names[i] ?? i}</span>
          <div className="bar"><div style={{ width: `${p * 100}%`, background: i === pred ? 'var(--accent)' : 'var(--blue)' }} /></div>
          <span className="mono tiny" style={{ width: 44, textAlign: 'right' }}>{(p * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}

function Scatter({ examples, size = 320 }: { examples: { vector?: number[]; label: number | string; pred?: number }[]; size?: number }) {
  const pts = examples.filter((e) => e.vector)
  const max = Math.max(1, ...pts.flatMap((e) => e.vector!.map(Math.abs))) * 1.08
  const P = (v: number) => size / 2 + (v / max) * (size / 2)
  return (
    <svg width={size} height={size} style={{ background: '#0a0e16', borderRadius: 10, border: '1px solid var(--border)' }}>
      <line x1={0} y1={size / 2} x2={size} y2={size / 2} stroke="#1c2438" />
      <line x1={size / 2} y1={0} x2={size / 2} y2={size} stroke="#1c2438" />
      {pts.map((e, i) => (
        <circle key={i} cx={P(e.vector![0])} cy={size - P(e.vector![1])} r={4} fill={CLASS_COLORS[Number(e.label) % 10]}
          stroke={e.pred != null && e.pred !== e.label ? '#f87171' : 'none'} strokeWidth={2} />
      ))}
    </svg>
  )
}

// ---------------------------------------------------------------------------
// 1. Look at the data
// ---------------------------------------------------------------------------
function DataBrowser() {
  const { datasets, draft } = useStore()
  const { access, flag } = useLearn()
  const allowed = datasets.filter((d) => !access.datasets || access.datasets.has(d.id))
  const [dsId, setDsId] = useState(() => sessionStorage.getItem('lego-ai/explore-ds') || draft.dataset)
  const ds = allowed.find((d) => d.id === dsId) ?? allowed[0]
  const [split, setSplit] = useState('train')
  const [n, setN] = useState(36)
  const [seed, setSeed] = useState(0)
  const [label, setLabel] = useState<number | null>(null)
  const [data, setData] = useState<Samples | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => { if (ds) sessionStorage.setItem('lego-ai/explore-ds', ds.id) }, [ds])
  useEffect(() => setLabel(null), [ds?.id])
  useEffect(() => {
    if (!ds) return
    setBusy(true)
    api.samples(ds.id, { n, split, seed, label }).then((d) => { setData(d); flag('sawData') }).catch(() => setData(null)).finally(() => setBusy(false))
  }, [ds, n, split, seed, label, flag])

  if (!ds) return null
  const counts = data?.class_counts
  const maxCount = counts ? Math.max(...counts) : 1
  return (
    <div className="explore-cols">
      <aside className="panel explore-side">
        <div className="panel-body col" style={{ gap: 12 }}>
          <label className="field"><span className="lbl">Dataset</span>
            <select value={ds.id} onChange={(e) => setDsId(e.target.value)}>
              {allowed.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </label>
          <div className="small muted">{ds.description}</div>
          <Difficulty dataset={ds} />
          <label className="field"><span className="lbl">Which part</span>
            <select value={split} onChange={(e) => setSplit(e.target.value)}>
              <option value="train">Training set (what the model learns from)</option>
              <option value="val">Validation set (checked every epoch)</option>
              <option value="test">Test set (the final exam)</option>
            </select>
          </label>
          <div className="row">
            <label className="field grow"><span className="lbl">How many</span>
              <select value={n} onChange={(e) => setN(Number(e.target.value))}>{[12, 24, 36, 60, 96].map((v) => <option key={v} value={v}>{v} examples</option>)}</select>
            </label>
            <button style={{ alignSelf: 'flex-end' }} onClick={() => setSeed(seed + 1)} disabled={busy}>🎲 Show others</button>
          </div>
          {counts && (
            <div>
              <h4 style={{ marginBottom: 6 }}>Classes <span className="faint" style={{ textTransform: 'none', letterSpacing: 0 }}>(click to filter)</span></h4>
              {counts.map((cnt, c) => (
                <div key={c} className={`class-row ${label === c ? 'on' : ''}`} onClick={() => setLabel(label === c ? null : c)}>
                  <span className="small" style={{ width: 74, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{data!.class_names[c] ?? c}</span>
                  <div className="bar grow"><div style={{ width: `${(cnt / maxCount) * 100}%`, background: CLASS_COLORS[c % 10] }} /></div>
                  <span className="mono tiny muted" style={{ width: 42, textAlign: 'right' }}>{fmtNum(cnt)}</span>
                </div>
              ))}
              <div className="tiny faint" style={{ marginTop: 4 }}>{fmtNum(data!.total)} examples in this part. Roughly equal bars mean the classes are balanced.</div>
            </div>
          )}
        </div>
      </aside>
      <section className="explore-main">
        <div className="row" style={{ marginBottom: 10 }}>
          <h3 className="grow">{data ? `${data.examples.length} random examples from the ${split === 'val' ? 'validation' : split === 'train' ? 'training' : 'test'} set` : 'Loading…'}{label != null && data && <> · only “{data.class_names[label] ?? label}”</>}</h3>
          {busy && <span className="spinner" />}
        </div>
        {data && <ExampleGrid kind={data.kind} examples={data.examples} names={data.class_names} ds={ds} />}
      </section>
    </div>
  )
}

function ExampleGrid({ kind, examples, names, ds }: { kind: string; examples: Partial<Prediction>[]; names?: string[]; ds?: DatasetInfo }) {
  if (!examples.length) return <div className="empty">Nothing to show.</div>
  if (kind === 'vector') {
    return (
      <div className="row" style={{ alignItems: 'flex-start', gap: 18 }}>
        <Scatter examples={examples as { vector?: number[]; label: number }[]} />
        <div className="small muted" style={{ maxWidth: 320 }}>Each dot is one example: its two numbers are its position, its color is its class. {examples.some((e) => e.pred != null) && 'A red ring marks a point the model got wrong.'} {ds?.id === 'spirals' && 'The three arms wind around each other, which is why a straight line cannot separate them.'}</div>
      </div>
    )
  }
  if (kind === 'tokens') return <div className="col" style={{ gap: 8 }}>{examples.map((e, i) => <ExampleCard key={i} ex={e} names={names} />)}</div>
  const big = (ds?.input_shape[1] ?? 28) > 28
  return <div className="pred-grid" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${big ? 96 : 84}px, 1fr))` }}>{examples.map((e, i) => <ExampleCard key={i} ex={e} names={names} size={big ? 80 : 64} />)}</div>
}

// ---------------------------------------------------------------------------
// 2. Test a model
// ---------------------------------------------------------------------------
function RunPicker({ runs, value, onChange, filter }: { runs: RunRow[]; value: string; onChange: (id: string) => void; filter?: (r: RunRow) => boolean }) {
  const { datasets } = useStore()
  const list = filter ? runs.filter(filter) : runs
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} style={{ maxWidth: '100%' }}>
      {!list.some((r) => r.id === value) && <option value="">Choose a trained model…</option>}
      {list.map((r) => (
        <option key={r.id} value={r.id}>
          {r.name} · {datasets.find((d) => d.id === r.dataset)?.name ?? r.dataset} · {fmtPct(r.summary.best_val_acc, 1)}
        </option>
      ))}
    </select>
  )
}

function ModelTester({ openChat }: { openChat: () => void }) {
  const runs = useUsableRuns()
  const { datasets, session, setTab } = useStore()
  const { flag, access } = useLearn()
  const [runId, setRunId] = useState(() => sessionStorage.getItem('lego-ai/explore-run') || '')
  const [split, setSplit] = useState('test')
  const [limit, setLimit] = useState(2000)
  const [cell, setCell] = useState<{ true: number | null; pred: number | null } | null>(null)
  const [ev, setEv] = useState<Evaluation | null>(null)
  const [gen, setGen] = useState<Predictions | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  // Default to the model that was just trained, otherwise the newest one.
  useEffect(() => {
    if (runs.length && !runs.some((r) => r.id === runId)) setRunId(runs.find((r) => r.id === session?.runId)?.id ?? runs[0].id)
  }, [runs, runId, session?.runId])
  useEffect(() => { if (runId) sessionStorage.setItem('lego-ai/explore-run', runId) }, [runId])
  useEffect(() => setCell(null), [runId, split, limit])

  const run = runs.find((r) => r.id === runId)
  const ds = datasets.find((d) => d.id === run?.dataset)
  const isLM = ds?.task === 'lm'

  useEffect(() => {
    if (!run || !ds) return
    setErr(null)
    setBusy(true)
    if (isLM) {
      setEv(null)
      api.predict(run.id, 1, Math.floor(Math.random() * 1000)).then(setGen).catch((e) => setErr((e as Error).message)).finally(() => setBusy(false))
      return
    }
    setGen(null)
    api.evaluate(run.id, { split, limit, true: cell?.true ?? null, pred: cell?.pred ?? null })
      .then((r) => { setEv(r); flag('mistakes') })
      .catch((e) => { setEv(null); setErr((e as Error).message) })
      .finally(() => setBusy(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run?.id, split, limit, cell, isLM])

  if (!runs.length) {
    return <div className="empty" style={{ margin: 30 }}><h3 style={{ color: 'var(--text)' }}>No trained models yet</h3><p>Train a model first, then come back here to see exactly what it gets right and wrong.</p></div>
  }
  const names = ev?.class_names ?? ds?.class_names ?? []
  const k = ev?.confusion.length ?? 0
  const maxOff = ev ? Math.max(1, ...ev.confusion.flatMap((row, i) => row.filter((_, j) => i !== j))) : 1
  const cellLabel = cell && ev
    ? cell.pred == null ? `everything that really is “${names[cell.true!]}”`
      : cell.true === cell.pred ? `“${names[cell.true!]}” it got right` : `“${names[cell.true!]}” it called “${names[cell.pred]}”`
    : 'its most confident mistakes'

  return (
    <div className="explore-cols">
      <aside className="panel explore-side">
        <div className="panel-body col" style={{ gap: 12 }}>
          <label className="field"><span className="lbl">Model</span><RunPicker runs={runs} value={runId} onChange={setRunId} /></label>
          {run && <div className="small muted">{ds?.name} · {fmtNum(run.summary.params)} weights · best validation accuracy {fmtPct(run.summary.best_val_acc)}</div>}
          {!isLM && <>
            <div className="hp-grid">
              <label className="field"><span className="lbl">Tested on</span>
                <select value={split} onChange={(e) => setSplit(e.target.value)}>
                  <option value="test">Test set</option><option value="val">Validation set</option><option value="train">Training set</option>
                </select>
              </label>
              <label className="field"><span className="lbl">How many</span>
                <select value={limit} onChange={(e) => setLimit(Number(e.target.value))}>
                  <option value={500}>500 examples</option><option value={2000}>2,000 examples</option><option value={0}>all of them</option>
                </select>
              </label>
            </div>
            {ev && (
              <div className="kpis" style={{ gridTemplateColumns: '1fr 1fr' }}>
                <div className="kpi"><div className="k">Accuracy</div><div className="v ok">{fmtPct(ev.accuracy)}</div></div>
                <div className="kpi"><div className="k">Mistakes</div><div className="v bad">{ev.wrong_total} <span className="tiny muted">of {fmtNum(ev.n)}</span></div></div>
              </div>
            )}
            {ev && (
              <div>
                <h4 style={{ marginBottom: 6 }}>Accuracy per class <span className="faint" style={{ textTransform: 'none', letterSpacing: 0 }}>(click to see them)</span></h4>
                {ev.per_class.map((c) => (
                  <div key={c.cls} className={`class-row ${cell?.true === c.cls && cell.pred == null ? 'on' : ''}`} onClick={() => setCell(cell?.true === c.cls && cell.pred == null ? null : { true: c.cls, pred: null })}>
                    <span className="small" style={{ width: 74, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{names[c.cls] ?? c.cls}</span>
                    <div className="bar grow"><div style={{ width: `${(c.acc ?? 0) * 100}%`, background: (c.acc ?? 0) > 0.95 ? 'var(--green)' : (c.acc ?? 0) > 0.8 ? 'var(--accent)' : 'var(--red)' }} /></div>
                    <span className="mono tiny" style={{ width: 42, textAlign: 'right' }}>{c.acc == null ? '-' : fmtPct(c.acc, 0)}</span>
                  </div>
                ))}
              </div>
            )}
          </>}
          {err && <div className="error-box">{err}</div>}
        </div>
      </aside>

      <section className="explore-main col" style={{ gap: 16 }}>
        {isLM && (
          <div className="panel"><div className="panel-body col" style={{ gap: 10 }}>
            <h3>What it writes</h3>
            <div className="small muted">This model predicts text one character at a time. Given the start of a passage from the test set, here is how it carries on:</div>
            {gen?.prompt != null ? <div className="gen-box"><span className="muted">{gen.prompt}</span><b>{gen.generated}</b></div> : <span className="spinner" />}
            <div className="row"><button className="primary" onClick={openChat}>💬 Chat with this model</button></div>
          </div></div>
        )}

        {!isLM && ev && k > 0 && k <= 20 && (
          <div className="panel"><div className="panel-body">
            <div className="row" style={{ marginBottom: 8 }}>
              <h3 className="grow">What it mixes up</h3>
              {busy && <span className="spinner" />}
              {cell && <button className="sm" onClick={() => setCell(null)}>Show all mistakes</button>}
            </div>
            <div className="row" style={{ alignItems: 'flex-start', gap: 18, flexWrap: 'wrap' }}>
              <table className="confusion" data-tour="confusion">
                <thead><tr><th className="tiny muted">true ↓ / guessed →</th>{names.slice(0, k).map((nm, j) => <th key={j} title={nm}>{nm.slice(0, 5)}</th>)}</tr></thead>
                <tbody>
                  {ev.confusion.map((row, i) => (
                    <tr key={i}>
                      <th title={names[i]}>{(names[i] ?? String(i)).slice(0, 9)}</th>
                      {row.map((v, j) => {
                        const on = cell?.true === i && cell.pred === j
                        const bg = i === j ? `rgba(74, 222, 128, ${v ? 0.12 + 0.25 * (v / Math.max(1, row.reduce((a, b) => a + b, 0))) : 0})` : `rgba(248, 113, 113, ${v ? 0.15 + 0.75 * (v / maxOff) : 0})`
                        return <td key={j} className={on ? 'on' : ''} style={{ background: bg }} onClick={() => v > 0 && setCell(on ? null : { true: i, pred: j })} title={`${v} “${names[i]}” guessed as “${names[j]}”`}>{v || ''}</td>
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="small muted" style={{ maxWidth: 300 }}>
                Each row is the true answer, each column is what the model said. Green (the diagonal) is correct. <b style={{ color: 'var(--red)' }}>Red</b> cells are mistakes: the brighter, the more often it happens. Click any cell to see those examples.
              </div>
            </div>
          </div></div>
        )}

        {!isLM && ev && (
          <div className="panel"><div className="panel-body">
            <div className="row" style={{ marginBottom: 10 }}>
              <h3 className="grow">{ev.selected ? `${Math.min(ev.selected, ev.examples.length)} of ${ev.selected}: ${cellLabel}` : cell ? 'None like that' : '🎉 No mistakes on these examples'}</h3>
            </div>
            <ExampleGrid kind={ev.kind} examples={ev.examples} names={names} ds={ds} />
            {!cell && ev.examples.length > 0 && <div className="tiny faint" style={{ marginTop: 8 }}>Sorted by how sure the model was. A mistake at 99% confidence usually means the example is odd, mislabeled, or exposes something the model has not learned.</div>}
          </div></div>
        )}

        {!isLM && ev && !cell && ev.unsure.length > 0 && (
          <div className="panel"><div className="panel-body">
            <h3 style={{ marginBottom: 10 }}>What it is least sure about</h3>
            <ExampleGrid kind={ev.kind} examples={ev.unsure} names={names} ds={ds} />
          </div></div>
        )}

        {!isLM && run && ds && <TryYourOwn run={run} ds={ds} />}
        {!isLM && run && access.has('leaderboard.page') && <div className="row"><button className="ghost" onClick={() => setTab('leaderboard')}>See this model on the leaderboard →</button></div>}
      </section>
    </div>
  )
}

function TryYourOwn({ run, ds }: { run: RunRow; ds: DatasetInfo }) {
  const { flag } = useLearn()
  const [res, setRes] = useState<InputPrediction | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [text, setText] = useState('')
  const [vec, setVec] = useState<number[]>(() => ds.input_shape.length === 1 ? Array(ds.input_shape[0]).fill(0) : [])
  useEffect(() => { setRes(null); setErr(null) }, [run.id])
  const ask = useCallback((input: { image?: number[]; text?: string; vector?: number[] }) => {
    setErr(null)
    api.predictInput(run.id, input).then((r) => { setRes(r); flag('ownInput') }).catch((e) => { setRes(null); setErr((e as Error).message) })
  }, [run.id, flag])

  return (
    <div className="panel" data-tour="try-own"><div className="panel-body">
      <h3 style={{ marginBottom: 4 }}>Try it on your own {ds.input_kind === 'image' ? 'image' : ds.input_kind === 'tokens' ? 'text' : 'numbers'}</h3>
      <div className="small muted" style={{ marginBottom: 12 }}>
        {ds.input_kind === 'image' ? (ds.input_shape[0] === 1 ? 'Draw something, or drop in a photo, and see what the model makes of it.' : 'Drop in a photo and see what the model makes of it.') : ds.input_kind === 'tokens' ? 'Type a headline or a sentence and see which class the model picks.' : 'Pick a point and see which class the model assigns.'}
      </div>
      <div className="try-grid">
        <div>
          {ds.input_kind === 'image' && <ImageInput key={run.id} shape={ds.input_shape} allowDraw={ds.input_shape[0] === 1} onPixels={(image) => ask({ image })} />}
          {ds.input_kind === 'tokens' && (
            <div className="col">
              <textarea rows={4} value={text} placeholder="e.g. Stocks rally as tech earnings beat forecasts" onChange={(e) => setText(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) ask({ text }) }} />
              <div className="row"><button className="primary" disabled={!text.trim()} onClick={() => ask({ text })}>Ask the model</button><span className="tiny faint">⌘↵</span></div>
            </div>
          )}
          {ds.input_kind === 'vector' && (
            <div className="row wrap">
              {vec.map((v, i) => <NumInput key={i} style={{ width: 80 }} value={v} emptyValue={0} onChange={(nv) => setVec(vec.map((o, j) => (j === i ? nv ?? 0 : o)))} />)}
              <button className="primary" onClick={() => ask({ vector: vec })}>Ask the model</button>
            </div>
          )}
        </div>
        <div>
          {err && <div className="error-box">{err}</div>}
          {!res && !err && <div className="empty small">The model’s answer appears here.</div>}
          {res && (
            <div className="col" style={{ gap: 10 }}>
              <div className="row" style={{ gap: 14 }}>
                {res.input.image && res.input.shape && (
                  <div style={{ textAlign: 'center' }}>
                    <TensorImage data={res.input.image} shape={res.input.shape} size={84} />
                    <div className="tiny faint">what the model sees</div>
                  </div>
                )}
                <div>
                  <div className="tiny muted" style={{ fontWeight: 700, letterSpacing: '0.06em' }}>THE MODEL SAYS</div>
                  <div style={{ fontSize: 30, fontWeight: 800, color: 'var(--accent)' }}>{res.class_names[res.pred] ?? res.pred}</div>
                  <div className="small muted">{(res.probs[res.pred] * 100).toFixed(1)}% sure</div>
                </div>
              </div>
              <ProbBars probs={res.probs} names={res.class_names} pred={res.pred} />
              {res.input.unknown && res.input.unknown.length > 0 && <div className="tiny warn">Words it has never seen (treated as “unknown”): {res.input.unknown.join(', ')}</div>}
              {res.input.truncated && <div className="tiny muted">Only the first {ds.seq_len} words fit in its input.</div>}
            </div>
          )}
        </div>
      </div>
    </div></div>
  )
}

// ---------------------------------------------------------------------------
// 3. Chat with a text model
// ---------------------------------------------------------------------------
interface Msg { role: 'user' | 'model'; text: string; model?: { id: string; name: string }; temperature?: number }
interface ChatState { id: string; title: string; messages: Msg[] }
const CHATS_KEY = 'lego-ai/chats'
const STARTERS = ['ROMEO:\n', 'To be, or not to be, ', 'KING HENRY:\nMy lords, ', 'First Citizen:\nWe are ']

const loadChats = (): ChatState[] => {
  try {
    return JSON.parse(localStorage.getItem(CHATS_KEY) ?? '[]')
  } catch {
    return []
  }
}

function Chat() {
  const runs = useUsableRuns()
  const { datasets, setTab } = useStore()
  const { flag, access } = useLearn()
  const lmRuns = useMemo(() => runs.filter((r) => datasets.find((d) => d.id === r.dataset)?.task === 'lm'), [runs, datasets])
  const [chats, setChats] = useState<ChatState[]>(loadChats)
  const [current, setCurrent] = useState<string | null>(() => loadChats()[0]?.id ?? null)
  const [modelId, setModelId] = useState(() => localStorage.getItem('lego-ai/chat-model') || '')
  const [temperature, setTemperature] = useState(0.8)
  const [length, setLength] = useState(200)
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [seen, setSeen] = useState<{ context: string; seq_len: number; dropped: string[] } | null>(null)
  const end = useRef<HTMLDivElement>(null)

  useEffect(() => localStorage.setItem(CHATS_KEY, JSON.stringify(chats.slice(0, 30))), [chats])
  useEffect(() => { if (modelId) localStorage.setItem('lego-ai/chat-model', modelId) }, [modelId])
  useEffect(() => {
    if (lmRuns.length && !lmRuns.some((r) => r.id === modelId)) setModelId(lmRuns[0].id)
  }, [lmRuns, modelId])
  const chat = chats.find((c) => c.id === current) ?? null
  // Braces matter: scrollIntoView returns a Promise in current browsers, and an effect must not return one.
  useEffect(() => {
    end.current?.scrollIntoView({ block: 'end' })
  }, [chat?.messages.length, busy])

  const model = lmRuns.find((r) => r.id === modelId)
  const newChat = () => {
    const c: ChatState = { id: Math.random().toString(36).slice(2), title: 'New chat', messages: [] }
    setChats((cs) => [c, ...cs])
    setCurrent(c.id)
    setSeen(null)
    return c
  }
  const patch = (id: string, fn: (c: ChatState) => ChatState) => setChats((cs) => cs.map((c) => (c.id === id ? fn(c) : c)))

  /** The model has no notion of turns: it simply continues the whole conversation as one text. */
  const transcript = (msgs: Msg[]) => msgs.map((m) => m.text).join('')

  const reply = async (c: ChatState, msgs: Msg[], seed: number) => {
    if (!model) return
    setBusy(true)
    setErr(null)
    try {
      const r = await api.generate(model.id, { prompt: transcript(msgs), max_new: length, temperature, seed })
      setSeen({ context: r.context, seq_len: r.seq_len, dropped: r.dropped })
      patch(c.id, (x) => ({ ...x, messages: [...msgs, { role: 'model', text: r.text, model: { id: model.id, name: model.name }, temperature }] }))
      flag('chat')
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const send = (text: string) => {
    if (!text || !model || busy) return
    const c = chat ?? newChat()
    const msgs: Msg[] = [...c.messages, { role: 'user', text }]
    patch(c.id, (x) => ({ ...x, messages: msgs, title: x.messages.length ? x.title : text.trim().slice(0, 32) || 'New chat' }))
    setInput('')
    reply(c, msgs, Math.floor(Math.random() * 1e6))
  }
  const regenerate = () => {
    if (!chat || busy) return
    const msgs = chat.messages[chat.messages.length - 1]?.role === 'model' ? chat.messages.slice(0, -1) : chat.messages
    if (!msgs.length) return
    patch(chat.id, (x) => ({ ...x, messages: msgs }))
    reply(chat, msgs, Math.floor(Math.random() * 1e6))
  }

  if (!lmRuns.length) {
    return (
      <div className="empty" style={{ margin: 30 }}>
        <h3 style={{ color: 'var(--text)' }}>No text-generating models yet</h3>
        <p>Chat works with language models trained on <b>Tiny Shakespeare</b>. {access.datasets && !access.datasets.has('shakespeare') ? 'Unlock Embeddings in the Skill Tree to get that dataset.' : 'Build one (for example the “Char LSTM” or “Tiny GPT” template), train it, and it will appear here.'}</p>
        {access.has('builder') && <button onClick={() => setTab('build')}>Open the Builder</button>}
      </div>
    )
  }

  return (
    <div className="chat-layout">
      <aside className="chat-list">
        <button className="primary" style={{ width: '100%', justifyContent: 'center' }} onClick={newChat}>+ New chat</button>
        {chats.map((c) => (
          <div key={c.id} className={`chat-item ${c.id === current ? 'on' : ''}`} onClick={() => { setCurrent(c.id); setSeen(null) }}>
            <span className="grow">{c.title}</span>
            <button className="ghost sm" title="Rename" onClick={(e) => { e.stopPropagation(); const t = prompt('Chat name', c.title); if (t) patch(c.id, (x) => ({ ...x, title: t })) }}>✎</button>
            <button className="ghost sm" title="Delete" onClick={(e) => { e.stopPropagation(); setChats((cs) => cs.filter((x) => x.id !== c.id)); if (current === c.id) setCurrent(null) }}>✕</button>
          </div>
        ))}
        {!chats.length && <div className="small muted" style={{ padding: 8 }}>Your chats are kept here.</div>}
      </aside>

      <section className="chat-main">
        <div className="chat-bar">
          <label className="row small" style={{ gap: 6, minWidth: 0 }}>Model
            <RunPicker runs={lmRuns} value={modelId} onChange={setModelId} />
          </label>
          <label className="row small" style={{ gap: 6 }} title="Low = safe and repetitive. High = creative and chaotic. 0 always picks the most likely next character.">
            Creativity <input type="range" min={0} max={1.5} step={0.1} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} style={{ width: 90, padding: 0 }} />
            <span className="mono tiny" style={{ width: 22 }}>{temperature.toFixed(1)}</span>
          </label>
          <label className="row small" style={{ gap: 6 }}>Length
            <select value={length} onChange={(e) => setLength(Number(e.target.value))}>{[80, 200, 400, 800].map((v) => <option key={v} value={v}>{v} characters</option>)}</select>
          </label>
        </div>

        <div className="chat-scroll">
          {(!chat || !chat.messages.length) && (
            <div className="chat-intro">
              <h3>Talk to {model?.name}</h3>
              <p className="muted">This model was trained only to guess the next character of Shakespeare. It does not understand questions: it reads what you type and <b>continues the text</b> in the style it learned. Better-trained models write more convincing lines. Try a starter:</p>
              <div className="row wrap" style={{ justifyContent: 'center' }}>
                {STARTERS.map((s) => <button key={s} onClick={() => send(s)}>{s.replace('\n', ' ↵ ')}</button>)}
              </div>
            </div>
          )}
          {chat?.messages.map((m, i) => (
            <div key={i} className={`chat-msg ${m.role}`}>
              <div className="tiny muted" style={{ marginBottom: 2 }}>{m.role === 'user' ? 'You' : `${m.model?.name ?? 'Model'} · creativity ${m.temperature?.toFixed(1)}`}</div>
              <div className="chat-bubble">{m.text}</div>
            </div>
          ))}
          {busy && <div className="chat-msg model"><div className="chat-bubble"><span className="spinner" /> writing…</div></div>}
          {err && <div className="error-box">{err}</div>}
          <div ref={end} />
        </div>

        {seen && (
          <div className="tiny muted chat-seen">
            The model only sees the last {seen.seq_len} characters: <span className="mono" style={{ color: 'var(--text)' }}>…{seen.context.slice(-48).replace(/\n/g, '↵')}</span>
            {seen.dropped.length > 0 && <> · characters it does not know were skipped: <span className="mono">{seen.dropped.join(' ')}</span></>}
          </div>
        )}
        <div className="chat-input">
          <textarea rows={2} value={input} placeholder="Type the start of a line, then Enter…" onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input) } }} />
          <div className="col" style={{ gap: 6 }}>
            <button className="primary" disabled={!input || busy} onClick={() => send(input)}>Send</button>
            <button className="sm" disabled={busy || !chat?.messages.length} onClick={regenerate} title="Throw away the last answer and write a different one">↻ Again</button>
          </div>
        </div>
      </section>
    </div>
  )
}
