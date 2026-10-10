import { useEffect, useState } from 'react'
import { api } from '../api'
import { useLearn } from '../learn/LearnContext'
import type { EpochPoint, Hardware } from '../types'

/** Where an epoch's time went, and what that says about the bottleneck. */
export function bottleneck(ep: EpochPoint | undefined): { parts: { label: string; frac: number; color: string }[]; note: string } | null {
  if (!ep || ep.compute_seconds == null) return null
  const d = ep.data_seconds ?? 0
  const c = ep.compute_seconds ?? 0
  const e = ep.eval_seconds ?? 0
  const total = d + c + e
  if (total <= 0) return null
  const parts = [
    { label: 'model math', frac: c / total, color: '#60a5fa' },
    { label: 'preparing data', frac: d / total, color: '#f472b6' },
    { label: 'validation', frac: e / total, color: '#fb923c' },
  ]
  let note = 'Most of the time goes into the model’s own calculations: a faster processor (a GPU) or a smaller model is what would speed this up.'
  if (d / total > 0.3) note = 'Preparing the data is a large share of the time (augmentation runs on the CPU). A bigger batch size helps; a faster GPU would not.'
  else if (e / total > 0.45) note = 'Much of each epoch is spent on validation, because the training set is small compared with the 5,000 validation examples. That is normal for small subsets.'
  return { parts, note }
}

export function HardwarePanel({ device, lastEpoch }: { device: string; lastEpoch?: EpochPoint }) {
  const { openLesson } = useLearn()
  const [hw, setHw] = useState<Hardware | null>(null)
  useEffect(() => {
    api.hardware().then(setHw).catch(() => setHw(null))
  }, [])
  if (!hw) return <div className="small muted"><span className="spinner" /> Looking at your hardware…</div>
  const b = bottleneck(lastEpoch)
  const gpu = hw.gpus[0]
  return (
    <div className="col" style={{ gap: 12 }}>
      <div className="stat-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <div className="stat"><div className="k">Processor</div><div className="v" title={hw.cpu}>{hw.cpu}</div><div className="tiny muted">{hw.cores} cores</div></div>
        <div className="stat"><div className="k">Memory (RAM)</div><div className="v">{hw.ram_gb.toFixed(0)} GB</div><div className="tiny muted">{hw.ram_free_gb.toFixed(1)} GB free now</div></div>
        <div className="stat" style={{ gridColumn: '1 / -1' }}>
          <div className="k">Graphics processor (GPU)</div>
          {gpu ? <>
            <div className="v" title={gpu.name}>{gpu.name}</div>
            <div className="tiny muted">up to {gpu.memory_gb.toFixed(1)} GB{gpu.shared ? ' (shared with the rest of the computer)' : ''} · choose device “{gpu.device}” to use it</div>
          </> : <div className="v">none found</div>}
        </div>
      </div>
      <div className="small">
        This run is set to train on <b>{device}</b>.
        {gpu && device === 'cpu' && <> Your GPU (“{gpu.device}”) is usually several times faster for convolutional and transformer models. The CPU is fine, and exactly reproducible, for small ones.</>}
      </div>
      {b && (
        <div>
          <h4 style={{ marginBottom: 6 }}>Where the last epoch’s time went</h4>
          <div className="time-bar">{b.parts.map((p) => <div key={p.label} style={{ width: `${p.frac * 100}%`, background: p.color }} title={`${p.label}: ${(p.frac * 100).toFixed(0)}%`} />)}</div>
          <div className="row wrap tiny muted" style={{ gap: 10, marginTop: 4 }}>
            {b.parts.map((p) => <span key={p.label}><span style={{ color: p.color }}>■</span> {p.label} {(p.frac * 100).toFixed(0)}%</span>)}
          </div>
          <div className="small muted" style={{ marginTop: 6 }}>{b.note}</div>
        </div>
      )}
      {hw.notes.length > 0 && (
        <ul className="small muted" style={{ margin: 0, paddingLeft: 18, lineHeight: 1.6 }}>
          {hw.notes.map((n) => <li key={n}>{n}</li>)}
        </ul>
      )}
      <button onClick={() => openLesson('train.hardware')}>📖 When and how to get a bigger GPU</button>
    </div>
  )
}
