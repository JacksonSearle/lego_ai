import { useCallback, useEffect, useState } from 'react'
import { api, hostAdmin } from '../api'
import { Chips } from '../components/ArchView'
import { GuideButton } from '../learn/GuideButton'
import { useLearn } from '../learn/LearnContext'
import { useStore } from '../store'
import type { GlobalEntry, GlobalInfo, RunDetail, Track } from '../types'
import { fmtAgo, fmtDur, fmtNum, fmtPct, MEDALS } from '../util'

/** A track's limits as short readable phrases. */
export function trackRules(t: Track, datasetName: string): string[] {
  const out = [`dataset: ${datasetName}`]
  if (t.max_params) out.push(`at most ${fmtNum(t.max_params)} weights`)
  if (t.max_epochs) out.push(`at most ${t.max_epochs} epoch${t.max_epochs === 1 ? '' : 's'}`)
  if (t.max_train_examples) out.push(`at most ${fmtNum(t.max_train_examples)} training examples`)
  if (t.max_seconds) out.push(`trains in under ${fmtDur(t.max_seconds)} on the host’s computer`)
  out.push(t.allow_custom_blocks ? 'custom blocks allowed (the host reads the code first)' : 'built-in blocks only')
  return out
}

/** Enter (or change) the address of the shared leaderboard and the name you appear under. */
function Connect({ onDone, initial }: { onDone: () => void; initial?: { url: string; player: string } }) {
  const [url, setUrl] = useState(initial?.url ?? '')
  const [player, setPlayer] = useState(initial?.player ?? '')
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const go = async () => {
    setBusy(true); setErr(null)
    try {
      await api.saveSettings({ server_url: url, player })
      if (url) await api.globalInfo()
      onDone()
    } catch (e) { setErr((e as Error).message) } finally { setBusy(false) }
  }
  return (
    <div className="col" style={{ gap: 10, maxWidth: 560 }}>
      <div className="small muted">A shared leaderboard is run by one person (a teacher, a club, a friend) on their computer. Everyone else enters its address here. Your models are trained again on the host’s computer, so every score on the board was measured the same way.</div>
      <label className="field"><span className="lbl">Leaderboard address</span><input placeholder="https://… (ask whoever runs it)" value={url} onChange={(e) => setUrl(e.target.value)} /></label>
      <label className="field"><span className="lbl">Your name on the board</span><input maxLength={40} placeholder="A nickname is fine" value={player} onChange={(e) => setPlayer(e.target.value)} /></label>
      {err && <div className="error-box">{err}</div>}
      <div className="row"><button className="primary" disabled={busy || !url.trim()} onClick={go}>{busy ? 'Connecting…' : 'Connect'}</button>{initial?.url && <button onClick={async () => { await api.saveSettings({ server_url: '' }); onDone() }}>Disconnect</button>}</div>
    </div>
  )
}

/** The shared leaderboard: tracks with limits, entries the host has checked, and the queue. */
export function SharedBoard() {
  const { datasets, blocks: specs, loadConfig, setTab, toast, refreshBlocks } = useStore()
  const [info, setInfo] = useState<GlobalInfo | null>(null)
  const [state, setState] = useState<'loading' | 'setup' | 'ok' | 'error'>('loading')
  const [err, setErr] = useState('')
  const [track, setTrack] = useState<string>(() => localStorage.getItem('lego-ai/track') ?? '')
  const [board, setBoard] = useState<{ entries: GlobalEntry[]; waiting: GlobalEntry[] } | null>(null)
  const [editing, setEditing] = useState(false)
  const [server, setServer] = useState('')
  const [code, setCode] = useState<{ id: string; files: Record<string, string> } | null>(null)

  const load = useCallback(async () => {
    const s = await api.settings()
    setServer(s.server_url)
    if (!s.server_url) { setState('setup'); return }
    try {
      const i = await api.globalInfo()
      setInfo(i)
      setState('ok')
      setTrack((t) => (i.tracks.some((x) => x.id === t) ? t : i.tracks[0]?.id ?? ''))
    } catch (e) { setErr((e as Error).message); setState('error') }
  }, [])
  useEffect(() => { load() }, [load])
  useEffect(() => {
    if (state !== 'ok' || !track) return
    localStorage.setItem('lego-ai/track', track)
    let alive = true
    const get = () => api.globalBoard(track).then((b) => { if (alive) setBoard(b) }).catch(() => {})
    get()
    const h = setInterval(get, 8000)
    return () => { alive = false; clearInterval(h) }
  }, [state, track])

  const open = async (id: string, allow = false) => {
    try {
      const r = await api.globalOpen(id, allow)
      if (r.needs_confirm) { setCode({ id, files: r.custom_blocks ?? {} }); return }
      setCode(null)
      await refreshBlocks()
      loadConfig(r.config!)
      setTab('build')
      toast(`Loaded “${r.config!.name}” from the shared leaderboard. Train it yourself to see it work.`, 'ok')
    } catch (e) { toast((e as Error).message, 'error') }
  }

  if (state === 'loading') return <div className="panel-body"><span className="spinner" /></div>
  if (state === 'setup' || editing) {
    return <div className="panel-body col" style={{ gap: 12 }}><h3>🌍 Connect to a shared leaderboard</h3><Connect initial={{ url: server, player: info?.player ?? '' }} onDone={() => { setEditing(false); setState('loading'); load() }} /></div>
  }
  if (state === 'error') {
    return <div className="panel-body col" style={{ gap: 10 }}><div className="error-box">{err}</div><div className="row"><button onClick={() => { setState('loading'); load() }}>Try again</button><button onClick={() => setEditing(true)}>Change the address</button></div></div>
  }
  const t = info!.tracks.find((x) => x.id === track)
  const mine = new Set(Object.values(info!.mine).map((m) => m.id))
  const dsName = (id: string) => datasets.find((d) => d.id === id)?.name ?? id
  return (
    <>
      <div className="row wrap" style={{ padding: '10px 12px', gap: 10 }}>
        <b>🌍 {info!.name}</b>
        <div className="seg">{info!.tracks.map((x) => <button key={x.id} className={track === x.id ? 'on' : ''} onClick={() => { setBoard(null); setTrack(x.id) }}>{x.name}</button>)}</div>
        <span className="spacer" />
        <span className="small muted">{info!.queue} waiting to be checked · you are “{info!.player || 'unnamed'}”</span>
        <button className="sm ghost" onClick={() => setEditing(true)}>Settings</button>
      </div>
      {t && (
        <div className="note-box" style={{ margin: '0 12px 10px' }}>
          <b>{t.name}.</b> {t.description}
          <div className="row wrap" style={{ gap: 6, marginTop: 6 }}>{trackRules(t, dsName(t.dataset)).map((r) => <span key={r} className="pill">{r}</span>)}</div>
          <div className="tiny" style={{ marginTop: 6, opacity: 0.8 }}>To enter: train a model that obeys these limits, open it on your own leaderboard (Runs) and press “Submit to the shared leaderboard”. The host trains it again and lists the score it measures.</div>
        </div>
      )}
      {code && (
        <div className="error-box" style={{ margin: '0 12px 10px' }}>
          <b>This entry contains custom block code.</b> Opening it installs and runs this Python on your computer. Only continue if you have read it and trust it.
          {Object.entries(code.files).map(([f, src]) => <pre key={f} className="json" style={{ maxHeight: 180, marginTop: 6 }}>{`# ${f}\n${src}`}</pre>)}
          <div className="row" style={{ marginTop: 6 }}><button className="danger sm" onClick={() => open(code.id, true)}>I trust this code: open it</button><button className="sm" onClick={() => setCode(null)}>Cancel</button></div>
        </div>
      )}
      {!board ? <div className="panel-body"><span className="spinner" /></div> : board.entries.length + board.waiting.length === 0 ? (
        <div className="empty" style={{ margin: 20 }}>No entries on this track yet. Be the first.</div>
      ) : (
        <table className="tbl">
          <thead><tr><th style={{ width: 36 }}>#</th><th>Who</th><th>Model</th><th>Architecture</th><th className="right">Params</th><th className="right" title="Measured by the host on its own computer">Test acc</th><th className="right">Val acc</th><th className="right" title="Training time on the host’s computer">Time</th><th className="right">When</th><th /></tr></thead>
          <tbody>
            {board.entries.map((e, i) => (
              <tr key={e.id} className={mine.has(e.id) ? 'selected' : ''}>
                <td className={`rank r${i + 1}`}>{i < 3 ? MEDALS[i] : i + 1}</td>
                <td><b>{e.player}</b>{mine.has(e.id) && <span className="pill accent" style={{ marginLeft: 6 }}>you</span>}</td>
                <td>{e.name}</td>
                <td><Chips types={e.block_types} specs={specs} /></td>
                <td className="num">{fmtNum(e.params)}</td>
                <td className="num" style={{ fontWeight: 700 }} title={e.match === 'exact' ? 'The host got exactly the score the submitter reported' : e.match === 'close' ? `The submitter reported ${fmtPct(e.claimed_test_acc)}: close, as expected on different hardware` : e.match === 'different' ? `The submitter reported ${fmtPct(e.claimed_test_acc)}` : ''}>{fmtPct(e.test_acc)}{e.match === 'exact' && <span className="ok"> ✓</span>}</td>
                <td className="num">{fmtPct(e.val_acc)}</td>
                <td className="num">{fmtDur(e.seconds)}</td>
                <td className="num muted">{fmtAgo(e.submitted_at)}</td>
                <td><button className="sm" title="Load this model’s recipe into the Builder" onClick={() => open(e.id)}>Open</button></td>
              </tr>
            ))}
            {board.waiting.map((e) => (
              <tr key={e.id} style={{ opacity: 0.6 }}>
                <td>·</td><td><b>{e.player}</b>{mine.has(e.id) && <span className="pill accent" style={{ marginLeft: 6 }}>you</span>}</td><td>{e.name}</td>
                <td><Chips types={e.block_types} specs={specs} /></td><td className="num">{fmtNum(e.params)}</td>
                <td className="num" colSpan={3}><span className="pill warn">{e.status === 'verifying' ? 'being checked now' : 'waiting to be checked'}</span></td>
                <td className="num muted">{fmtAgo(e.submitted_at)}</td><td />
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  )
}

/** In a run's detail panel: enter this run on a track of the shared leaderboard. */
export function SubmitRun({ run }: { run: RunDetail }) {
  const { flag } = useLearn()
  const [info, setInfo] = useState<GlobalInfo | null>(null)
  const [missing, setMissing] = useState(false)
  const [track, setTrack] = useState('')
  const [player, setPlayer] = useState('')
  const [problems, setProblems] = useState<string[] | null>(null)
  const [status, setStatus] = useState<GlobalEntry | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const ds = run.config.dataset.id
  useEffect(() => {
    api.globalInfo().then((i) => {
      setInfo(i)
      setPlayer(i.player)
      const fit = i.tracks.filter((t) => t.dataset === ds)
      setTrack(fit[0]?.id ?? '')
      const sub = i.mine[run.id]
      if (sub) api.globalSubmission(sub.id).then(setStatus).catch(() => {})
    }).catch(() => setMissing(true))
  }, [run.id, ds])
  useEffect(() => {
    setProblems(null)
    if (track) api.globalCheck(run.id, track).then((c) => setProblems(c.problems)).catch((e) => setErr((e as Error).message))
  }, [track, run.id])
  if (missing) return <div className="small muted">🌍 To enter this model on a shared leaderboard, connect to one first: Leaderboard → Shared.</div>
  if (!info) return null
  const fit = info.tracks.filter((t) => t.dataset === ds)
  const submit = async () => {
    setBusy(true); setErr(null)
    try {
      const r = await api.globalSubmit(run.id, track, player)
      flag('submit')
      setStatus(await api.globalSubmission(r.id))
    } catch (e) { setErr((e as Error).message) } finally { setBusy(false) }
  }
  return (
    <div className="col" style={{ gap: 6 }}>
      <h4>🌍 Shared leaderboard: {info.name}</h4>
      {status ? (
        <div className={status.status === 'rejected' ? 'error-box' : 'note-box'}>
          {status.status === 'verified' ? <>✓ Checked by the host: <b>{fmtPct(status.test_acc)}</b> test accuracy on the host’s computer{status.match === 'exact' ? ', exactly what you measured.' : status.match === 'close' ? ` (you measured ${fmtPct(status.claimed_test_acc)}: small differences between computers are normal).` : '.'}</>
            : status.status === 'rejected' ? <>Not accepted: {status.reason}</>
            : <>Submitted. It is waiting for the host to train it again and confirm the score. Check back on Leaderboard → Shared.</>}
        </div>
      ) : fit.length === 0 ? <div className="small muted">This leaderboard has no track for this dataset. Its tracks: {info.tracks.map((t) => t.name).join(', ')}.</div> : <>
        <div className="row wrap small">
          <select value={track} onChange={(e) => setTrack(e.target.value)}>{fit.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}</select>
          <span className="muted">as</span>
          <input style={{ width: 150 }} maxLength={40} placeholder="your name" value={player} onChange={(e) => setPlayer(e.target.value)} />
          <button disabled={busy || !problems || problems.length > 0 || !player.trim() || run.status !== 'completed'} onClick={submit}>{busy ? 'Sending…' : 'Submit'}</button>
        </div>
        {problems && problems.length > 0 && <div className="small warn">Does not fit this track yet:<ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>{problems.map((p) => <li key={p}>{p}</li>)}</ul></div>}
        {problems && problems.length === 0 && <div className="tiny faint">Fits the rules. Only the recipe is sent (blocks and settings), not your weights: the host trains it again from scratch.</div>}
      </>}
      {err && <div className="error-box">{err}</div>}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Host admin (only in a copy started with LEGO_HOST=1)
// ---------------------------------------------------------------------------
interface QueueRow extends GlobalEntry { claimed: { test_acc?: number | null }; client: { version?: string; ip?: string }; custom_blocks: Record<string, string> }
interface Feedback { id: string; at: number; text: string; player: string; contact: string; kind: string; version: string; context: Record<string, string> }

export function HostAdmin() {
  const { meta, datasets, toast } = useStore()
  const [view, setView] = useState<'queue' | 'tracks' | 'feedback'>('queue')
  const [queue, setQueue] = useState<{ submissions: QueueRow[]; busy: boolean } | null>(null)
  const [fb, setFb] = useState<Feedback[]>([])
  const [tracks, setTracks] = useState<Track[]>([])
  const [needToken, setNeedToken] = useState(false)
  const [token, setToken] = useState('')
  const [showCode, setShowCode] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setQueue(await hostAdmin('GET', '/queue'))
      setFb(await hostAdmin('GET', '/feedback'))
      setNeedToken(false)
    } catch (e) { if ((e as { status?: number }).status === 401) setNeedToken(true) }
  }, [])
  useEffect(() => {
    load()
    fetch('/host/info').then((r) => r.json()).then((i) => setTracks(i.tracks)).catch(() => {})
    const h = setInterval(load, 4000)
    return () => clearInterval(h)
  }, [load])
  const act = async (fn: () => Promise<unknown>, msg?: string) => { try { await fn(); if (msg) toast(msg, 'ok'); load() } catch (e) { toast((e as Error).message, 'error') } }

  if (needToken) {
    return (
      <div className="page"><div className="panel" style={{ maxWidth: 520 }}><div className="panel-body col" style={{ gap: 10 }}>
        <h3>Host admin</h3>
        <div className="small muted">Open this page with the address the host printed when it started (it ends in ?token=…), or enter the admin token (LEGO_ADMIN_TOKEN).</div>
        <div className="row"><input className="grow" type="password" autoComplete="off" value={token} onChange={(e) => setToken(e.target.value)} /><button className="primary" onClick={() => { sessionStorage.setItem('lego-ai/admin-token', token); setToken(''); load() }}>Unlock</button></div>
      </div></div></div>
    )
  }
  const pending = queue?.submissions.filter((s) => s.status === 'pending') ?? []
  const setT = (i: number, k: keyof Track, v: unknown) => setTracks((ts) => ts.map((t, j) => (j === i ? { ...t, [k]: v } : t)))
  const num = (v: string) => (v.trim() === '' ? null : Math.max(0, Number(v)) || null)
  return (
    <div className="page">
      <div className="row wrap" style={{ marginBottom: 14, gap: 12 }}>
        <h2>Host: {meta?.host_name}</h2>
        <GuideButton lesson="adv.shared" />
        <div className="seg">
          <button className={view === 'queue' ? 'on' : ''} onClick={() => setView('queue')}>Submissions{pending.length ? ` (${pending.length} waiting)` : ''}</button>
          <button className={view === 'tracks' ? 'on' : ''} onClick={() => setView('tracks')}>Tracks</button>
          <button className={view === 'feedback' ? 'on' : ''} onClick={() => setView('feedback')}>Feedback{fb.length ? ` (${fb.length})` : ''}</button>
        </div>
        <span className="spacer" />
        {view === 'queue' && <button className="primary" disabled={!pending.length || queue?.busy} onClick={() => act(() => hostAdmin('POST', '/verify-all'), 'Working through the queue: each entry is trained here in turn')}>{queue?.busy ? 'Checking…' : `Check all ${pending.filter((s) => !Object.keys(s.custom_blocks).length).length} waiting`}</button>}
      </div>
      {view === 'queue' && (
        <div className="panel lb-table">
          {!queue ? <div className="panel-body"><span className="spinner" /></div> : queue.submissions.length === 0 ? <div className="empty" style={{ margin: 20 }}>No submissions yet. Share this computer’s address with your players.</div> : (
            <table className="tbl">
              <thead><tr><th>Who</th><th>Model</th><th>Track</th><th className="right">Params</th><th className="right">They measured</th><th className="right">This computer</th><th>Status</th><th className="right">When</th><th /></tr></thead>
              <tbody>
                {queue.submissions.map((s) => {
                  const hasCode = Object.keys(s.custom_blocks).length > 0
                  return (
                    <tr key={s.id}>
                      <td><b>{s.player}</b><div className="tiny faint">{s.client.ip} · v{s.client.version}</div></td>
                      <td>{s.name}{hasCode && <div><button className="sm danger" onClick={() => setShowCode(showCode === s.id ? null : s.id)}>⚠ contains code: read it</button></div>}
                        {showCode === s.id && Object.entries(s.custom_blocks).map(([f, src]) => <pre key={f} className="json" style={{ maxHeight: 240, maxWidth: 520 }}>{`# ${f}\n${src}`}</pre>)}
                      </td>
                      <td className="small">{s.track}</td>
                      <td className="num">{fmtNum(s.params)}</td>
                      <td className="num">{fmtPct(s.claimed.test_acc)}</td>
                      <td className="num" style={{ fontWeight: 700 }}>{fmtPct(s.test_acc)}{s.seconds != null && <div className="tiny faint">{fmtDur(s.seconds)}</div>}</td>
                      <td><span className={`pill ${s.status === 'verified' ? 'ok' : s.status === 'rejected' ? 'bad' : 'warn'}`}>{s.status}</span>{s.match && <span className="tiny muted"> {s.match}</span>}{s.reason && <div className="tiny warn" style={{ maxWidth: 260 }}>{s.reason}</div>}</td>
                      <td className="num muted">{fmtAgo(s.submitted_at)}</td>
                      <td className="nowrap">
                        {s.status === 'pending' && <button className="sm" disabled={queue.busy} title={hasCode ? 'Runs the custom code on this computer' : 'Train it here and record the score'} onClick={() => (!hasCode || confirm('This submission contains Python code that will run on this computer. Have you read it?')) && act(() => hostAdmin('POST', `/submissions/${s.id}/verify`, { allow_code: hasCode }))}>Check</button>}
                        {s.status !== 'rejected' && <button className="sm ghost" onClick={() => { const reason = prompt('Reason shown to the submitter', 'Removed by the host.'); if (reason != null) act(() => hostAdmin('POST', `/submissions/${s.id}/reject`, { reason })) }}>Reject</button>}
                        <button className="sm ghost danger" onClick={() => confirm('Delete this submission for good?') && act(() => hostAdmin('DELETE', `/submissions/${s.id}`))}>✕</button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>
      )}
      {view === 'tracks' && (
        <div className="panel"><div className="panel-body col" style={{ gap: 12 }}>
          <div className="small muted">A track is a leaderboard with limits. Tight limits on small datasets keep every entry quick to check and make it a contest of ideas, not of hardware. Leave a limit empty for “no limit”.</div>
          {tracks.map((t, i) => (
            <div key={i} className="ds-card">
              <div className="row wrap">
                <input style={{ width: 150 }} placeholder="id" value={t.id} onChange={(e) => setT(i, 'id', e.target.value)} />
                <input className="grow" placeholder="Name" value={t.name} onChange={(e) => setT(i, 'name', e.target.value)} />
                <select value={t.dataset} onChange={(e) => setT(i, 'dataset', e.target.value)}>{datasets.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}</select>
                <button className="sm ghost danger" onClick={() => setTracks((ts) => ts.filter((_, j) => j !== i))}>Remove</button>
              </div>
              <input placeholder="Description" value={t.description} onChange={(e) => setT(i, 'description', e.target.value)} />
              <div className="row wrap small">
                {([['max_params', 'max weights'], ['max_epochs', 'max epochs'], ['max_train_examples', 'max training examples'], ['max_seconds', 'max seconds here']] as const).map(([k, l]) => (
                  <label key={k} className="row muted">{l}<input style={{ width: 90 }} value={t[k] ?? ''} onChange={(e) => setT(i, k, num(e.target.value))} /></label>
                ))}
                <label className="row muted" title="Custom blocks are Python code that will run on this computer when you check an entry"><input type="checkbox" checked={t.allow_custom_blocks} onChange={(e) => setT(i, 'allow_custom_blocks', e.target.checked)} /> allow custom blocks</label>
              </div>
            </div>
          ))}
          <div className="row">
            <button onClick={() => setTracks((ts) => [...ts, { id: `track-${ts.length + 1}`, name: 'New track', dataset: 'mnist', description: '', max_params: 100000, max_epochs: 3, max_train_examples: null, max_seconds: 180, allow_custom_blocks: false }])}>+ Add a track</button>
            <span className="spacer" />
            <button className="primary" onClick={() => act(async () => setTracks(await hostAdmin('PUT', '/tracks', tracks)), 'Tracks saved')}>Save tracks</button>
          </div>
        </div></div>
      )}
      {view === 'feedback' && (
        <div className="col" style={{ gap: 10 }}>
          {fb.length === 0 && <div className="empty">No feedback yet. It arrives here when a player presses the Feedback button in their copy.</div>}
          {fb.map((f) => (
            <div key={f.id} className="panel"><div className="panel-body col" style={{ gap: 6 }}>
              <div className="row small"><span className="pill">{f.kind}</span><b>{f.player || 'anonymous'}</b><span className="muted">{f.contact}</span><span className="spacer" /><span className="muted">v{f.version} · {f.context?.tab} · {f.context?.platform} · {fmtAgo(f.at)}</span></div>
              <div style={{ whiteSpace: 'pre-wrap' }}>{f.text}</div>
            </div></div>
          ))}
        </div>
      )}
    </div>
  )
}
