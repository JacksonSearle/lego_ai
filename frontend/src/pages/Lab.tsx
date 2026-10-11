import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import { Brick } from '../components/Brick'
import { LineChart } from '../components/LineChart'
import { TensorImage } from '../components/TensorImage'
import { useLearn } from '../learn/LearnContext'
import { newId, useStore } from '../store'
import type { BlockInst, Predictions, RunRow } from '../types'
import { fmtDur, fmtNum, fmtPct, MEDALS } from '../util'

export const WIDTHS = [
  { id: 'too_small', label: 'Too small', n: 2, emoji: '🐜', note: 'Only 2 neurons: every digit has to be squeezed through just 2 numbers. That isn’t enough room to tell 10 digits apart, so many get mixed up.' },
  { id: 'small', label: 'Small', n: 16, emoji: '🐭', note: '16 neurons already capture the main strokes of each digit. Not bad for such a tiny model!' },
  { id: 'medium', label: 'Medium', n: 64, emoji: '🐕', note: '64 neurons: a sweet spot, fast to train and quite accurate.' },
  { id: 'big', label: 'Big', n: 512, emoji: '🐘', note: '512 neurons: more room to learn subtle shapes, a bit more accurate, but 8× the weights of Medium.' },
  { id: 'too_big', label: 'Too big', n: 4096, emoji: '🐳', note: '4,096 neurons and 3.2 million weights. Slower, and barely better than Big, and with longer training such a model can start memorizing instead of learning (overfitting). Bigger isn’t always better.' },
]

const labBlocks = (n: number, layers: number): BlockInst[] => [
  { id: newId(), type: 'flatten', params: {}, repeat: 1 },
  { id: newId(), type: 'linear', params: { out_features: n, activation: 'relu', bias: true }, repeat: layers },
  { id: newId(), type: 'classifier', params: { pool: 'auto', dropout: 0 }, repeat: 1 },
]

export function Lab() {
  const { blocks: specs, draft, setDraft, session, attachJob, effectiveTraining, toast, datasets } = useStore()
  const { access, openLesson, progress } = useLearn()
  const [width, setWidth] = useState<string | null>(() => sessionStorage.getItem('lego-ai/lab-width'))
  const [layers, setLayers] = useState(() => Number(sessionStorage.getItem('lego-ai/lab-layers') || 1))
  const [board, setBoard] = useState<RunRow[]>([])
  const [preds, setPreds] = useState<Predictions | null>(null)
  const [predRun, setPredRun] = useState<string | null>(null)
  const mnist = datasets.find((d) => d.id === 'mnist')
  const w = WIDTHS.find((x) => x.id === width)

  useEffect(() => { if (width) sessionStorage.setItem('lego-ai/lab-width', width); sessionStorage.setItem('lego-ai/lab-layers', String(layers)) }, [width, layers])

  const loadBoard = () => api.runs().then((rs) => setBoard(rs.filter((r) => r.dataset === 'mnist' && r.status === 'completed' && r.summary.best_val_acc != null)))
  useEffect(() => { loadBoard() }, [session?.status])

  const running = session?.status === 'running'
  const lastEp = session?.epochPts[session.epochPts.length - 1]
  const lastStep = session?.steps[session.steps.length - 1]
  const done = session?.status === 'completed' && session.summary

  useEffect(() => {
    if (done && session?.runId && session.runId !== predRun) {
      setPredRun(session.runId)
      api.predict(session.runId, 12, Math.floor(Math.random() * 1000)).then(setPreds).catch(() => setPreds(null))
    }
  }, [done, session?.runId, predRun])

  const train = async (wid: string, nLayers = layers) => {
    const opt = WIDTHS.find((x) => x.id === wid)!
    setWidth(wid)
    setPreds(null)
    if (running && session) await api.stopJob(session.jobId)
    const blocks = labBlocks(opt.n, nLayers)
    const name = `${opt.label} hidden layer (${opt.n})${nLayers > 1 ? ` ×${nLayers}` : ''}`
    setDraft((d) => ({ ...d, name, dataset: 'mnist', train_subset: null, blocks, init_from: undefined, origin: 'lab', edited: false }))
    try {
      const cfg = { name, dataset: { id: 'mnist', train_subset: null }, architecture: { blocks }, training: effectiveTraining(draft.training) }
      const { job, run } = await api.train(cfg)
      attachJob(job.id, 'train', name, run.id, { width: wid })
    } catch (e) {
      toast((e as Error).message, 'error')
    }
  }

  const accSeries = useMemo(() => [
    { label: 'training accuracy', color: '#60a5fa', points: (session?.steps ?? []).map((s) => ({ x: s.epoch, y: s.acc })), width: 1.8 },
  ], [session?.steps])

  const acc = lastEp?.val_acc ?? lastStep?.acc
  const progressFrac = session?.totalSteps && lastStep ? Math.min(1, lastStep.step / session.totalSteps) : 0
  const ranked = [...board].sort((a, b) => (b.summary.best_val_acc ?? 0) - (a.summary.best_val_acc ?? 0)).slice(0, 12)
  const layerCount = access.has('lab.layers') ? layers : 1
  // 784 → n → … → n → 10, with biases
  const weights = w ? 784 * w.n + w.n + (layerCount - 1) * (w.n * w.n + w.n) + w.n * 10 + 10 : 0

  return (
    <div className="page lab">
      <div className="lab-hero">
        <div className="grow">
          <div className="tiny muted" style={{ fontWeight: 700, letterSpacing: '0.08em' }}>LAB 1</div>
          <h1>Teach a computer to read handwriting</h1>
          <p className="muted" style={{ margin: '6px 0 0', maxWidth: 680 }}>
            Your model will look at 55,000 handwritten digits and learn to tell which digit each one is. Pick how big its “brain” (the hidden layer) should be. Training starts straight away.
          </p>
        </div>
        <div className="row wrap">
          <button onClick={() => openLesson('start.mnist')}>📖 What is MNIST?</button>
          <button onClick={() => openLesson('start.nn')}>📖 What’s a neural network?</button>
        </div>
      </div>

      <div className="lab-grid">
        {/* ------------ the model ------------ */}
        <div className="panel">
          <div className="panel-head"><h3 className="grow">Your model</h3><span className="tiny muted">data flows up ↑</span></div>
          <div className="panel-body lab-model">
            <div className="io-node" style={{ padding: '6px 10px' }}>
              <div className="io-label">OUTPUT</div>
              <div className="small">a score for each digit 0–9</div>
            </div>
            <div className="lab-arrow">↑</div>
            {specs.classifier && <Brick spec={specs.classifier} inst={{ id: 'h', type: 'classifier', params: {}, repeat: 1 }} compact />}
            <div className="lab-arrow">↑</div>
            <div className="lab-hidden" data-tour="lab-widths">
              {specs.linear && <Brick spec={specs.linear} inst={{ id: 'l', type: 'linear', params: { out_features: w?.n ?? 64, activation: 'relu' }, repeat: layerCount }} compact />}
              <Neurons n={w?.n ?? 0} />
              <div className="lab-label">How wide should the hidden layer be?</div>
              <div className="width-options">
                {WIDTHS.map((o) => (
                  <button key={o.id} className={`width-opt ${width === o.id ? 'on' : ''} ${progress.stats.widths.includes(o.id) ? 'tried' : ''}`} onClick={() => train(o.id)}>
                    <span style={{ fontSize: 20 }}>{o.emoji}</span>
                    <b>{o.label}</b>
                    <span className="tiny mono muted">{fmtNum(o.n)} neurons</span>
                  </button>
                ))}
              </div>
              {access.has('lab.layers') ? (
                <div data-tour="lab-layers" style={{ marginTop: 12 }}>
                  <div className="lab-label">How many hidden layers? <button className="ghost sm" onClick={() => openLesson('start.layers')}>📖 why?</button></div>
                  <div className="seg">
                    {[1, 2, 3, 4].map((n) => <button key={n} className={layers === n ? 'on' : ''} onClick={() => { setLayers(n); if (width) train(width, n) }}>{n}</button>)}
                  </div>
                  {w && <div className={`tiny ${weights > 10e6 ? 'warn' : 'muted'}`} style={{ marginTop: 6 }}>
                    ≈ {fmtNum(weights)} weights{weights > 10e6 ? ': that’s huge! Expect a few minutes of training on a CPU.' : ''}
                  </div>}
                </div>
              ) : <div className="tiny faint" style={{ marginTop: 10 }}>🔒 Level 2: stack more than one hidden layer</div>}
            </div>
            <div className="lab-arrow">↑</div>
            {specs.flatten && <Brick spec={specs.flatten} inst={{ id: 'f', type: 'flatten', params: {}, repeat: 1 }} compact />}
            <div className="tiny muted" style={{ textAlign: 'center', margin: '4px 0' }}>unrolls 28×28 pixels into a list of 784 numbers</div>
            <div className="lab-arrow">↑</div>
            <div className="io-node" style={{ padding: '6px 10px' }}>
              <div className="io-label">INPUT · {mnist?.name}</div>
              <div className="small">a 28×28 picture of a digit</div>
            </div>
          </div>
        </div>

        {/* ------------ results ------------ */}
        <div className="col" style={{ gap: 14 }}>
          <div className="panel">
            <div className="panel-body">
              {!session && !width && (
                <div className="lab-empty">
                  <div style={{ fontSize: 42 }}>👈</div>
                  <h3>Pick a size for the hidden layer to start training</h3>
                  <p className="muted small">Not sure? Start with <b>Medium</b>, then try the extremes and compare.</p>
                </div>
              )}
              {session && (
                <>
                  <div className="row">
                    <div className="grow">
                      <div className="tiny muted" style={{ fontWeight: 700, letterSpacing: '0.06em' }}>{running ? 'TRAINING…' : session.status === 'completed' ? 'RESULT' : session.status.toUpperCase()}</div>
                      <div className="lab-acc" style={{ color: acc == null ? 'var(--muted)' : acc > 0.95 ? 'var(--green)' : acc > 0.85 ? 'var(--accent)' : 'var(--orange)' }}>
                        {acc == null ? '…' : fmtPct(acc, 1)}
                      </div>
                      <div className="small muted">{lastEp ? 'of digits it had never seen were read correctly' : running ? 'correct on the digits it’s practicing on' : ''}</div>
                    </div>
                    <div className="right small muted">
                      <div>{session.name}</div>
                      {session.params != null && <div>{fmtNum(session.params)} weights</div>}
                      {session.summary?.duration != null && <div>trained in {fmtDur(session.summary.duration)}</div>}
                    </div>
                  </div>
                  <div className="progress" style={{ marginTop: 10 }}><div style={{ width: `${(session.status === 'completed' ? 1 : progressFrac) * 100}%` }} /></div>
                  {session.error && <div className="error-box" style={{ marginTop: 10 }}>{session.error}</div>}
                  {session.steps.length > 1 && <div style={{ marginTop: 12 }}><LineChart series={accSeries} percent height={150} xLabel="epoch (one full pass over the data)" /></div>}
                  {done && w && <div className="note-box" style={{ marginTop: 12 }}>{w.emoji} <b>{w.label}:</b> {w.note}</div>}
                </>
              )}
            </div>
          </div>

          {preds?.examples && (
            <div className="panel">
              <div className="panel-head"><h3 className="grow">What did it guess?</h3><span className="small muted">{preds.examples.filter((e) => e.pred === e.label).length} / {preds.examples.length} right</span></div>
              <div className="panel-body pred-grid">
                {preds.examples.map((e, i) => (
                  <div key={i} className={`pred ${e.pred === e.label ? 'right' : 'wrong'}`}>
                    {e.image && e.shape && <TensorImage data={e.image} shape={e.shape} size={52} />}
                    <div className="small"><b>{e.pred}</b> {e.pred === e.label ? '✓' : <span className="bad">✗ (was {e.label})</span>}</div>
                    <div className="tiny muted">{Math.round(e.top[0].p * 100)}% sure</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ------------ leaderboard ------------ */}
        <div className="panel">
          <div className="panel-head"><h3 className="grow">🏆 MNIST leaderboard</h3><span className="tiny muted">validation accuracy</span></div>
          <div className="panel-body" style={{ padding: 8 }}>
            {ranked.length === 0 && <div className="small muted" style={{ padding: 8 }}>Your trained models will be ranked here.</div>}
            {ranked.map((r, i) => (
              <div key={r.id} className={`lab-lb-row ${r.id === session?.runId ? 'me' : ''}`}>
                <span className="mono">{i < 3 ? MEDALS[i] : `${i + 1}.`}</span>
                <span className="grow small">{r.name}<div className="tiny faint">{fmtNum(r.summary.params)} weights</div></span>
                <span className="mono small" style={{ fontWeight: 650 }}>{fmtPct(r.summary.best_val_acc, 1)}</span>
              </div>
            ))}
            {board.length > 12 && <div className="tiny faint" style={{ padding: 6 }}>+ {board.length - 12} more</div>}
          </div>
        </div>
      </div>
    </div>
  )
}

function Neurons({ n }: { n: number }) {
  const shown = Math.min(n, 64)
  if (!n) return null
  return (
    <div className="neurons" title={`${n} neurons`}>
      {Array.from({ length: shown }, (_, i) => <span key={i} />)}
      {n > shown && <em className="tiny muted">+{fmtNum(n - shown)}</em>}
    </div>
  )
}
