import { useEffect, useState } from 'react'
import { api } from '../api'
import { useLearn } from '../learn/LearnContext'
import { useStore } from '../store'
import type { UpdateInfo } from '../types'

const KINDS = [['bug', '🐞 Something broke'], ['confusing', '🤔 Something confused me'], ['idea', '💡 An idea'], ['praise', '👍 Something I liked']] as const

/** The feedback button in the top bar, and the "update available" notice. */
export function FeedbackButton() {
  const { tab, meta } = useStore()
  const { flag } = useLearn()
  const [open, setOpen] = useState(false)
  const [kind, setKind] = useState('idea')
  const [text, setText] = useState('')
  const [contact, setContact] = useState('')
  const [ctx, setCtx] = useState(true)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<{ sent: boolean; email: string | null } | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const send = async () => {
    setBusy(true); setErr(null)
    try {
      setResult(await api.feedback({ text, kind, contact, tab, include_context: ctx }))
      flag('feedback')
    } catch (e) { setErr((e as Error).message) } finally { setBusy(false) }
  }
  const close = () => { setOpen(false); if (result) { setText(''); setResult(null) } }
  return (
    <>
      <button className="ghost sm" onClick={() => setOpen(true)} title="Tell the makers what works and what does not">💬 Feedback</button>
      {open && (
        <div className="modal-back" onClick={close}>
          <div className="modal" style={{ width: 'min(560px, 100%)' }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-head"><h2>Send feedback</h2><span className="spacer" /><button className="ghost" onClick={close}>✕</button></div>
            <div className="modal-body col" style={{ gap: 12 }}>
              {result ? <>
                <div className="note-box">
                  {result.sent ? '✓ Sent. Thank you: this goes straight to the people making Lego AI.' : 'Saved on this computer. This copy is not connected to a shared server, so it could not be sent automatically.'}
                  {!result.sent && result.email && <> You can <a href={`mailto:${result.email}?subject=${encodeURIComponent('Lego AI feedback')}&body=${encodeURIComponent(text)}`}>email it instead</a>.</>}
                </div>
                <div className="row"><span className="spacer" /><button className="primary" onClick={close}>Done</button></div>
              </> : <>
                <div className="seg" style={{ flexWrap: 'wrap' }}>{KINDS.map(([k, l]) => <button key={k} className={kind === k ? 'on' : ''} onClick={() => setKind(k)}>{l}</button>)}</div>
                <textarea rows={6} autoFocus placeholder={kind === 'bug' ? 'What did you do, what did you expect, and what happened instead?' : 'What is on your mind?'} value={text} onChange={(e) => setText(e.target.value)} style={{ width: '100%', resize: 'vertical' }} />
                <input placeholder="Email or name if you would like a reply (optional)" value={contact} onChange={(e) => setContact(e.target.value)} />
                <label className="row small muted"><input type="checkbox" checked={ctx} onChange={(e) => setCtx(e.target.checked)} /> include which page I am on, the app version and my operating system (helps with bugs; nothing about your models or data)</label>
                {err && <div className="error-box">{err}</div>}
                <div className="row"><span className="tiny faint">Lego AI {meta?.version}</span><span className="spacer" /><button className="primary" disabled={busy || !text.trim()} onClick={send}>{busy ? 'Sending…' : 'Send'}</button></div>
              </>}
            </div>
          </div>
        </div>
      )}
    </>
  )
}

export function UpdateBanner() {
  const [u, setU] = useState<UpdateInfo | null>(null)
  const [hidden, setHidden] = useState(() => sessionStorage.getItem('lego-ai/update-hidden') ?? '')
  useEffect(() => { api.update().then(setU).catch(() => {}) }, [])
  if (!u?.available || hidden === u.latest) return null
  return (
    <div className="offline-banner" style={{ background: '#0e1d33', borderColor: '#1e3a66', color: '#cfe1ff' }}>
      ⬆️ Lego AI {u.latest} is available (you have {u.current}). Close this window and start Lego AI again to update. Your models and data are kept.
      {u.url && <> <a href={u.url} target="_blank" rel="noreferrer">What’s new ↗</a></>}
      <button className="sm ghost" style={{ marginLeft: 10 }} onClick={() => { sessionStorage.setItem('lego-ai/update-hidden', u.latest ?? ''); setHidden(u.latest ?? '') }}>Later</button>
    </div>
  )
}
