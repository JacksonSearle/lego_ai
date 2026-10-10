import { useEffect, useState } from 'react'
import { api } from '../api'
import { useStore } from '../store'
import type { Estimate } from '../types'

/** "about 40 s", "about 12 min", "about 3.5 h", "about 2 days" */
export function fmtDur(s: number | null | undefined) {
  if (s == null) return '-'
  if (s < 1) return 'under 1 s'
  if (s < 90) return `${Math.round(s)} s`
  if (s < 5400) return `${Math.round(s / 60)} min`
  if (s < 48 * 3600) return `${(s / 3600).toFixed(1)} h`
  return `${(s / 86400).toFixed(1)} days`
}

const fmtGB = (g: number) => (g < 0.1 ? `${Math.max(1, Math.round(g * 1024))} MB` : `${g.toFixed(1)} GB`)

/** Memory and time the current draft needs, here and on well-known GPUs. Null until the model is valid. */
export function useEstimate(): Estimate | null {
  const { draft, inspect, effectiveTraining } = useStore()
  const [est, setEst] = useState<Estimate | null>(null)
  const t = effectiveTraining(draft.training)
  const ready = !!inspect?.ok && inspect.flops_per_sample != null
  const key = JSON.stringify([
    ready, draft.dataset, draft.seq_len ?? null, draft.train_subset, inspect?.total_params, inspect?.act_bytes_per_sample, inspect?.flops_per_sample,
    t.batch_size, t.optimizer, t.epochs, t.device,
  ])
  useEffect(() => {
    if (!ready || !inspect) {
      setEst(null)
      return
    }
    let live = true
    const h = setTimeout(() => {
      api.estimate({
        dataset: draft.dataset, seq_len: draft.seq_len, train_subset: draft.train_subset,
        params: inspect.total_params, act_bytes: inspect.act_bytes_per_sample ?? 0, flops: inspect.flops_per_sample ?? 0,
        batch_size: Number(t.batch_size) || 128, optimizer: String(t.optimizer ?? 'adamw'), epochs: Number(t.epochs) || 1, device: String(t.device ?? 'cpu'),
      }).then((e) => { if (live) setEst(e) }).catch(() => { if (live) setEst(null) })
    }, 300)
    return () => { live = false; clearTimeout(h) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  return est
}

export const verdictTone = (e: Estimate) => (e.verdict === 'too_big' ? 'bad' : e.verdict === 'ok' ? 'ok' : 'warn')
export const verdictLabel = (e: Estimate) => (e.verdict === 'too_big' ? '⚠ too big for this computer' : e.verdict === 'slow' ? '⚠ very slow here' : e.verdict === 'long' ? '⏱ slow here' : '✓ fits here')

/** Can this computer train the model, and if not, what hardware would it take? */
export function SizeAdvisor({ est, compact }: { est: Estimate | null; compact?: boolean }) {
  if (!est) return null
  const tone = verdictTone(est)
  const box = tone === 'bad' ? 'error-box' : 'note-box'
  const showGpus = !compact || est.verdict !== 'ok'
  return (
    <div className="col" style={{ gap: 8 }}>
      <div className={box} style={tone === 'warn' ? { background: '#2a1a0b', borderColor: '#5a3514', color: '#fed7aa' } : undefined}>
        {est.headline}
        {est.faster_here && <> Switching the device to “{est.faster_here}” would be noticeably faster.</>}
      </div>
      <table className="tbl size-table">
        <thead>
          <tr><th>Where</th><th title={`Weights, gradients and optimizer state (${fmtGB(est.fixed_gb)}) plus ${est.per_sample_mb.toFixed(2)} MB for every example in a batch`}>Memory</th><th>One epoch</th></tr>
        </thead>
        <tbody>
          {est.here.map((h) => (
            <tr key={h.device} className={h.selected ? 'selected' : ''}>
              <td>{h.label}{h.selected && <span className="tiny muted"> · selected</span>}</td>
              <td className={h.fits ? '' : 'bad'} title={`${fmtGB(h.capacity_gb)} usable`}>
                {h.fits ? `✓ ${fmtGB(h.need_gb)} of ${fmtGB(h.capacity_gb)}` : h.max_batch >= 1 ? `✗ fits at batch ≤ ${h.max_batch}` : '✗ does not fit'}
              </td>
              <td className="mono">{h.fits ? fmtDur(h.epoch_seconds) : '-'}</td>
            </tr>
          ))}
          {showGpus && est.gpus.map((g) => (
            <tr key={g.name}>
              <td title={g.note}>{g.count && g.count > 1 ? `${g.count} × ` : ''}{g.name} <span className="tiny muted">{g.memory_gb} GB</span></td>
              <td className={g.count ? '' : 'bad'}>{g.count == null ? '✗ too big even for one example' : g.count > 1 ? `needs ${g.count} of them` : '✓ one is enough'}</td>
              <td className="mono">{g.count ? fmtDur(g.epoch_seconds) : '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {!compact && (
        <div className="tiny faint">
          Rough estimates from the model’s size and a quick speed test of this computer. Real times can differ by 2× or more. GPU rows are typical cards you can rent or use for free online.
        </div>
      )}
    </div>
  )
}
