import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import { GuideButton } from '../learn/GuideButton'
import { useStore } from '../store'
import type { RunRow, ScalePlanRow } from '../types'
import { fmtDur, fmtLoss, fmtNum, fmtPct } from '../util'

type Axis = 'width' | 'data'
type Metric = 'val_loss' | 'val_err' | 'test_loss' | 'test_err'
const METRICS: { id: Metric; label: string }[] = [
  { id: 'val_loss', label: 'Validation loss' },
  { id: 'val_err', label: 'Validation error' },
  { id: 'test_loss', label: 'Test loss' },
  { id: 'test_err', label: 'Test error' },
]
const FACTORS: Record<Axis, number[]> = { width: [0.25, 0.5, 1, 2, 4], data: [0.03, 0.1, 0.3, 1] }

interface Point { x: number; y: number; label: string; runId: string }
interface Fit { a: number; b: number; r2: number }

/** Straight-line fit in log-log space: y ≈ a · x^(−b), with how well it fits (R²). */
function fitLogLog(pts: Point[]): Fit | null {
  const use = pts.filter((p) => p.x > 0 && p.y > 0)
  if (use.length < 3) return null
  const lx = use.map((p) => Math.log(p.x)), ly = use.map((p) => Math.log(p.y))
  const n = use.length
  const mx = lx.reduce((s, v) => s + v, 0) / n, my = ly.reduce((s, v) => s + v, 0) / n
  let sxx = 0, sxy = 0, syy = 0
  for (let i = 0; i < n; i++) {
    sxx += (lx[i] - mx) ** 2
    sxy += (lx[i] - mx) * (ly[i] - my)
    syy += (ly[i] - my) ** 2
  }
  if (sxx === 0) return null
  const slope = sxy / sxx
  return { a: Math.exp(my - slope * mx), b: -slope, r2: syy === 0 ? 1 : (sxy * sxy) / (sxx * syy) }
}

const fmtX = (v: number) => (v >= 1e6 ? `${+(v / 1e6).toPrecision(3)}M` : v >= 1e3 ? `${+(v / 1e3).toPrecision(3)}k` : `${+v.toPrecision(3)}`)

/** Log-log scatter with the fitted power law drawn through it. */
function LogLogPlot({ pts, fit, xLabel, yLabel, pct }: { pts: Point[]; fit: Fit | null; xLabel: string; yLabel: string; pct: boolean }) {
  const W = 620, H = 360, L = 64, R = 20, T = 16, B = 46
  const use = pts.filter((p) => p.x > 0 && p.y > 0)
  if (!use.length) return <div className="empty">Points appear here as each size finishes training.</div>
  const pad = (lo: number, hi: number) => (lo === hi ? [lo / 2, hi * 2] : [lo / (hi / lo) ** 0.12, hi * (hi / lo) ** 0.12])
  const [x0, x1] = pad(Math.min(...use.map((p) => p.x)), Math.max(...use.map((p) => p.x)))
  const [y0, y1] = pad(Math.min(...use.map((p) => p.y)), Math.max(...use.map((p) => p.y)))
  const sx = (v: number) => L + ((Math.log(v) - Math.log(x0)) / (Math.log(x1) - Math.log(x0))) * (W - L - R)
  const sy = (v: number) => H - B - ((Math.log(v) - Math.log(y0)) / (Math.log(y1) - Math.log(y0))) * (H - T - B)
  const ticks = (lo: number, hi: number) => {
    const out: number[] = []
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) for (const m of [1, 2, 5]) {
      const v = m * 10 ** e
      if (v >= lo && v <= hi) out.push(v)
    }
    // Keep the axis readable when the range spans many decades.
    return out.length > 9 ? out.filter((v) => /^1/.test(v.toExponential())) : out
  }
  const fy = (v: number) => (pct ? `${+(v * 100).toPrecision(2)}%` : `${+v.toPrecision(2)}`)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: 760, display: 'block' }} role="img" aria-label="Scaling curve">
      {ticks(x0, x1).map((v) => (
        <g key={`x${v}`}>
          <line x1={sx(v)} x2={sx(v)} y1={T} y2={H - B} stroke="var(--border)" />
          <text x={sx(v)} y={H - B + 16} textAnchor="middle" fontSize="11" fill="var(--muted)">{fmtX(v)}</text>
        </g>
      ))}
      {ticks(y0, y1).map((v) => (
        <g key={`y${v}`}>
          <line x1={L} x2={W - R} y1={sy(v)} y2={sy(v)} stroke="var(--border)" />
          <text x={L - 8} y={sy(v) + 4} textAnchor="end" fontSize="11" fill="var(--muted)">{fy(v)}</text>
        </g>
      ))}
      <text x={(L + W - R) / 2} y={H - 8} textAnchor="middle" fontSize="12" fill="var(--muted)">{xLabel} (log scale)</text>
      <text x={14} y={(T + H - B) / 2} textAnchor="middle" fontSize="12" fill="var(--muted)" transform={`rotate(-90 14 ${(T + H - B) / 2})`}>{yLabel} (log scale)</text>
      <clipPath id="scale-plot-area"><rect x={L} y={T} width={W - L - R} height={H - T - B} /></clipPath>
      {fit && (
        <line x1={sx(x0)} x2={sx(x1)} y1={sy(fit.a * x0 ** -fit.b)} y2={sy(fit.a * x1 ** -fit.b)}
          stroke="var(--accent)" strokeWidth="2" strokeDasharray="6 5" clipPath="url(#scale-plot-area)" />
      )}
      <polyline points={use.map((p) => `${sx(p.x)},${sy(p.y)}`).join(' ')} fill="none" stroke="#60a5fa" strokeWidth="1.5" opacity="0.6" />
      {use.map((p) => (
        <g key={p.runId}>
          <circle cx={sx(p.x)} cy={sy(p.y)} r="6" fill="#60a5fa" stroke="#0b1220" strokeWidth="2"><title>{`${p.label}: ${fy(p.y)}`}</title></circle>
          <text x={sx(p.x)} y={sy(p.y) - 11} textAnchor="middle" fontSize="10.5" fill="var(--text)">{p.label}</text>
        </g>
      ))}
    </svg>
  )
}

/** What the fit says, in plain words. */
function verdict(fit: Fit | null, n: number, axis: Axis, what: string): { tone: string; text: string } {
  const thing = axis === 'width' ? 'weights' : 'training examples'
  if (!fit) return { tone: 'muted', text: n < 3 ? 'A fit needs at least three finished sizes.' : 'Not enough usable points to fit a trend.' }
  if (fit.b <= 0.02) return { tone: 'warn', text: `${what} does not fall as the ${axis === 'width' ? 'model grows' : 'data grows'}. More ${thing} are not what this setup needs: ${axis === 'width' ? 'it is probably limited by the amount of data or the training time' : 'the model is probably too small to use more data'}.` }
  const double = (1 - 2 ** -fit.b) * 100
  const halve = 2 ** (1 / fit.b)
  const cost = halve > 1e6 ? 'an astronomical number of' : `about ${halve >= 100 ? Math.round(halve).toLocaleString() : +halve.toPrecision(2)} times the`
  if (fit.r2 >= 0.92) return { tone: 'ok', text: `Yes: the points lie close to a straight line on this log-log plot, which is what a power law looks like. Every doubling of the ${thing} cuts ${what.toLowerCase()} by about ${double.toFixed(0)}%. Halving it would take ${cost} ${thing}.` }
  if (fit.r2 >= 0.7) return { tone: 'warn', text: `Roughly. The trend is downward (about ${double.toFixed(0)}% lower per doubling) but the points bend away from a straight line. Curves often flatten at the large end when something else becomes the limit: ${axis === 'width' ? 'the data or the training time' : 'the size of the model'}.` }
  return { tone: 'bad', text: 'No clean scaling law here: the points do not line up. With so few short runs, luck (the random seed) can be as large as the effect of size. Try more epochs or more sizes.' }
}

export function Scaling() {
  const { draft, inspect, dataset, toConfig, attachJob, session, toast, setTab } = useStore()
  const [axis, setAxis] = useState<Axis>('width')
  const [chosen, setChosen] = useState<Record<Axis, number[]>>({ width: [0.25, 0.5, 1, 2, 4], data: [0.03, 0.1, 0.3, 1] })
  const [plan, setPlan] = useState<ScalePlanRow[] | null>(null)
  const [planErr, setPlanErr] = useState<string | null>(null)
  const [rows, setRows] = useState<RunRow[]>([])
  const [study, setStudy] = useState<string | null>(null)
  const [metric, setMetric] = useState<Metric>('val_loss')
  const [busy, setBusy] = useState(false)

  const load = useCallback(() => api.runs().then((r) => setRows(r.filter((x) => x.scale))).catch(() => {}), [])
  const live = session?.kind === 'scale' ? session : null
  // Reload finished points as the live study produces them.
  useEffect(() => { load() }, [load, live?.points.length, live?.status])
  useEffect(() => { if (live?.scale) setStudy(live.scale.id) }, [live?.scale])

  const valid = !!inspect?.ok && draft.blocks.length > 0
  const factors = chosen[axis]
  const key = JSON.stringify([axis, factors, draft.dataset, draft.seq_len, draft.train_subset, draft.blocks, valid])
  useEffect(() => {
    if (!valid || factors.length === 0) { setPlan(null); return }
    let alive = true
    const h = setTimeout(() => {
      api.scalingPlan(toConfig(), axis, factors).then((p) => { if (alive) { setPlan(p); setPlanErr(null) } }).catch((e) => { if (alive) { setPlan(null); setPlanErr((e as Error).message) } })
    }, 300)
    return () => { alive = false; clearTimeout(h) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  const studies = useMemo(() => {
    const by = new Map<string, RunRow[]>()
    rows.forEach((r) => by.set(r.scale!.id, [...(by.get(r.scale!.id) ?? []), r]))
    return [...by.entries()].map(([id, rs]) => ({ id, runs: rs.sort((a, b) => a.scale!.index - b.scale!.index), tag: rs[0].scale!, created: Math.max(...rs.map((r) => r.created_at)), dataset: rs[0].dataset }))
      .sort((a, b) => b.created - a.created)
  }, [rows])
  const cur = studies.find((s) => s.id === study) ?? studies[0]

  const value = (r: RunRow): number | undefined => {
    const s = r.summary
    return metric === 'val_loss' ? s.best_val_loss : metric === 'val_err' ? (s.best_val_acc == null ? undefined : 1 - s.best_val_acc) : metric === 'test_loss' ? s.test_loss : s.test_acc == null ? undefined : 1 - s.test_acc
  }
  const pts: Point[] = (cur?.runs ?? []).filter((r) => r.status === 'completed' && value(r) != null).map((r) => ({
    x: cur!.tag.axis === 'width' ? (r.summary.params ?? 0) : (r.summary.train_examples ?? r.scale!.n_train),
    y: value(r)!, label: cur!.tag.axis === 'width' ? `${r.scale!.factor}×` : fmtX(r.summary.train_examples ?? r.scale!.n_train), runId: r.id,
  }))
  const fit = fitLogLog(pts)
  const mLabel = METRICS.find((m) => m.id === metric)!.label
  const v = verdict(fit, pts.length, cur?.tag.axis ?? 'width', mLabel)
  const running = session?.status === 'running'
  const buildable = (plan ?? []).filter((p) => !p.error)

  const start = async () => {
    setBusy(true)
    try {
      const { job } = await api.scaling(toConfig(), axis, factors)
      attachJob(job.id, 'scale', `Scaling · ${draft.name}`)
      toast('Scaling study started. Each size trains in turn: watch the live charts on the Train tab.', 'ok')
    } catch (e) {
      toast((e as Error).message, 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="page">
      <div className="row wrap" style={{ marginBottom: 14, gap: 12 }}>
        <h2>Scaling curves</h2>
        <GuideButton lesson="adv.scaling" />
        <span className="small muted">Train one design at several sizes and see whether bigger keeps paying off.</span>
      </div>
      <div className="scaling-layout">
        <div className="panel">
          <div className="panel-head"><h3 className="grow">New study</h3></div>
          <div className="panel-body col" style={{ gap: 12 }}>
            <div className="small">
              Model: <b>{draft.name}</b> on {dataset?.name}
              <div className="muted">{valid ? `${fmtNum(inspect?.total_params)} weights at 1×` : 'The model in the Builder is not valid yet.'} · <a onClick={() => setTab('build')} style={{ cursor: 'pointer' }}>edit in Builder</a></div>
            </div>
            <div>
              <div className="lbl small muted" style={{ marginBottom: 4 }}>What to scale</div>
              <div className="seg" style={{ width: '100%' }}>
                <button className={axis === 'width' ? 'on' : ''} style={{ flex: 1 }} onClick={() => setAxis('width')}>Model size</button>
                <button className={axis === 'data' ? 'on' : ''} style={{ flex: 1 }} onClick={() => setAxis('data')}>Amount of data</button>
              </div>
              <div className="tiny faint" style={{ marginTop: 4 }}>
                {axis === 'width' ? 'Every layer is made wider or narrower by the same factor. The data and training settings stay the same.' : 'The model stays the same and trains on a fraction of the training set.'}
              </div>
            </div>
            <div>
              <div className="lbl small muted" style={{ marginBottom: 4 }}>{axis === 'width' ? 'Width factors' : 'Fractions of the training set'}</div>
              <div className="row wrap" style={{ gap: 6 }}>
                {FACTORS[axis].map((f) => {
                  const on = factors.includes(f)
                  return <button key={f} className={`sm ${on ? 'primary' : ''}`} onClick={() => setChosen((c) => ({ ...c, [axis]: on ? c[axis].filter((x) => x !== f) : [...c[axis], f].sort((a, b) => a - b) }))}>{axis === 'width' ? `${f}×` : `${f * 100}%`}</button>
                })}
              </div>
            </div>
            {planErr && <div className="error-box">{planErr}</div>}
            {plan && (
              <table className="tbl small">
                <thead><tr><th>Size</th><th className="right">Weights</th><th className="right">Examples</th></tr></thead>
                <tbody>
                  {plan.map((p) => (
                    <tr key={p.factor} title={p.error ?? ''}>
                      <td>{p.label}</td>
                      <td className="num">{p.error ? <span className="bad">{p.params > 1e8 ? 'too big' : 'does not build'}</span> : fmtNum(p.params)}</td>
                      <td className="num">{fmtNum(p.n_train)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            <div className="tiny faint">Each size is a full training run with the settings from the Train tab ({String(draft.training.epochs)} epochs). Sizes that round to the same model are trained once.</div>
            <button className="primary" disabled={!valid || busy || running || buildable.length < 2} onClick={start}>
              {running ? 'A job is already running' : `▶ Train ${buildable.length} sizes`}
            </button>
            {live && live.status === 'running' && <div className="small muted"><span className="spinner" style={{ width: 10, height: 10 }} /> {live.message} · <a onClick={() => setTab('train')} style={{ cursor: 'pointer' }}>live charts</a></div>}
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <h3 className="grow">{cur ? `${cur.tag.name} · ${cur.tag.axis === 'width' ? 'model size' : 'amount of data'}` : 'Results'}</h3>
            {studies.length > 1 && (
              <select value={cur?.id} onChange={(e) => setStudy(e.target.value)}>
                {studies.map((s) => <option key={s.id} value={s.id}>{s.tag.name} · {s.tag.axis === 'width' ? 'size' : 'data'} · {s.runs.length} points</option>)}
              </select>
            )}
            <select value={metric} onChange={(e) => setMetric(e.target.value as Metric)}>
              {METRICS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
            </select>
          </div>
          <div className="panel-body col" style={{ gap: 12 }}>
            {!cur ? (
              <div className="empty">No scaling studies yet. Pick the sizes on the left and press Train.</div>
            ) : <>
              <LogLogPlot pts={pts} fit={fit} xLabel={cur.tag.axis === 'width' ? 'Weights' : 'Training examples'} yLabel={mLabel} pct={metric.endsWith('err')} />
              <div className={v.tone === 'ok' ? 'note-box' : v.tone === 'bad' ? 'error-box' : 'note-box'}>
                <b>Does it follow a scaling law?</b> {v.text}
                {fit && <div className="mono small" style={{ marginTop: 6 }}>{mLabel.toLowerCase()} ≈ {+fit.a.toPrecision(3)} × N<sup>−{fit.b.toFixed(3)}</sup> · R² = {fit.r2.toFixed(3)} · {pts.length} points</div>}
              </div>
              <table className="tbl">
                <thead><tr><th>Size</th><th className="right">Weights</th><th className="right">Examples</th><th className="right">Val acc</th><th className="right">Val loss</th><th className="right">Test acc</th><th className="right">Time</th></tr></thead>
                <tbody>
                  {cur.runs.map((r) => (
                    <tr key={r.id}>
                      <td>{cur.tag.axis === 'width' ? `${r.scale!.factor}× width` : `${(r.scale!.factor * 100).toFixed(0)}% of data`} {r.status !== 'completed' && <span className={`pill ${r.status === 'running' ? 'accent' : 'warn'}`}>{r.status}</span>}</td>
                      <td className="num">{fmtNum(r.summary.params)}</td>
                      <td className="num">{fmtNum(r.summary.train_examples ?? r.scale!.n_train)}</td>
                      <td className="num">{fmtPct(r.summary.best_val_acc)}</td>
                      <td className="num">{fmtLoss(r.summary.best_val_loss)}</td>
                      <td className="num">{fmtPct(r.summary.test_acc)}</td>
                      <td className="num">{fmtDur(r.summary.duration)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="tiny faint">Every point is an ordinary run on the Leaderboard. One seed per size: differences smaller than the seed-to-seed spread are noise.</div>
            </>}
          </div>
        </div>
      </div>
    </div>
  )
}
