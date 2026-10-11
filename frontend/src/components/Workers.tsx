import { useEffect, useState } from 'react'
import { api } from '../api'
import { useLearn } from '../learn/LearnContext'
import type { Worker } from '../types'

/** Choose where a run trains: this computer, or another one (a "worker") such as a rented cloud GPU. */
export function Workers({ value, onChange }: { value: string; onChange: (id: string) => void }) {
  const { openLesson } = useLearn()
  const [list, setList] = useState<Worker[] | null>(null)
  const [adding, setAdding] = useState(false)
  const [name, setName] = useState('')
  const [url, setUrl] = useState('')
  const [token, setToken] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  useEffect(() => { api.workers().then(setList).catch(() => setList([])) }, [])
  // A worker that was removed or went away cannot stay selected.
  useEffect(() => { if (list && value && !list.some((w) => w.id === value && w.ok)) onChange('') }, [list, value, onChange])
  const add = async () => {
    setBusy(true); setErr(null)
    try {
      setList(await api.addWorker(name, url, token))
      setAdding(false); setName(''); setUrl(''); setToken('')
    } catch (e) { setErr((e as Error).message) } finally { setBusy(false) }
  }
  return (
    <div className="col" style={{ gap: 8 }}>
      <div className="row"><h4 className="grow">Where to train</h4><button className="ghost sm" onClick={() => openLesson('adv.cloud')}>📖 Cloud GPUs</button></div>
      <label className={`ds-card ${value === '' ? 'chosen-card' : ''}`} style={{ cursor: 'pointer' }}>
        <div className="row"><input type="radio" checked={value === ''} onChange={() => onChange('')} /><b>This computer</b></div>
      </label>
      {(list ?? []).map((w) => (
        <label key={w.id} className={`ds-card ${value === w.id ? 'chosen-card' : ''}`} style={{ cursor: w.ok ? 'pointer' : 'default', opacity: w.ok ? 1 : 0.6 }}>
          <div className="row">
            <input type="radio" disabled={!w.ok} checked={value === w.id} onChange={() => onChange(w.id)} />
            <b className="grow">☁ {w.name}</b>
            <button className="sm ghost danger" onClick={async (e) => { e.preventDefault(); setList(await api.deleteWorker(w.id)) }}>Remove</button>
          </div>
          <div className="tiny muted">
            {w.ok ? <>{w.gpu ? `${w.gpu}${w.gpu_gb ? `, ${w.gpu_gb.toFixed(0)} GB` : ''}` : `no GPU (${w.cpu})`} · trains on “{w.device}”{w.same_version ? '' : ` · runs version ${w.version}, which differs from this copy`}</> : <span className="bad">{w.error}</span>}
          </div>
        </label>
      ))}
      {value !== '' && <div className="tiny faint">The run trains there with live charts here, then its results and weights are copied back to your leaderboard. Works for a single run (not Optuna, seeds or several datasets).</div>}
      {adding ? (
        <div className="ds-card">
          <input placeholder="A name for it, for example “Rented A100”" value={name} onChange={(e) => setName(e.target.value)} />
          <input placeholder="Address, for example http://123.45.67.89:8000" value={url} onChange={(e) => setUrl(e.target.value)} />
          <input type="password" autoComplete="off" placeholder="Its access token (the LEGO_TOKEN it was started with)" value={token} onChange={(e) => setToken(e.target.value)} />
          {err && <div className="error-box">{err}</div>}
          <div className="row"><button className="primary sm" disabled={busy || !url.trim()} onClick={add}>{busy ? 'Connecting…' : 'Connect'}</button><button className="sm" onClick={() => setAdding(false)}>Cancel</button></div>
          <div className="tiny faint">The other computer runs Lego AI too, started with a token so that only you can use it. The Cloud GPUs lesson shows how.</div>
        </div>
      ) : <button className="sm" onClick={() => setAdding(true)}>+ Add another computer</button>}
    </div>
  )
}
