import { useEffect, useLayoutEffect, useState } from 'react'
import { useStore } from '../store'
import { AVATARS, LEVELS, MAX_LEVEL, QUESTS, levelFor, type Quest } from './curriculum'
import { useLearn } from './LearnContext'
import { LESSONS } from './lessons'

// ---------------------------------------------------------------------------
// Top bar: avatar, level, XP
// ---------------------------------------------------------------------------
export function ProfileBar() {
  const { progress, access, ready } = useLearn()
  const { setTab } = useStore()
  const [open, setOpen] = useState(false)
  if (!ready || !progress.mode) return null
  const lvl = levelFor(progress.xp)
  const next = LEVELS.find((l) => l.level === lvl.level + 1)
  const frac = next ? (progress.xp - lvl.xp) / (next.xp - lvl.xp) : 1
  return (
    <>
      <div className="profile-bar">
        <button className="avatar-btn" onClick={() => setOpen(true)} title="Profile, avatars & settings">{progress.avatar}</button>
        {access.unlocked ? (
          <span className="pill accent" title="Everything is unlocked">🔓 All unlocked</span>
        ) : (
          <div className="xp" title={`${progress.xp} XP${next ? ` — ${next.xp - progress.xp} to level ${next.level}` : ''}`}>
            <div className="row" style={{ gap: 6 }}>
              <span className="lvl">Lv {lvl.level}</span>
              <span className="small">{lvl.title}</span>
              <span className="spacer" />
              <span className="tiny mono muted">{progress.xp}{next ? ` / ${next.xp}` : ''} XP</span>
            </div>
            <div className="xp-track"><div style={{ width: `${frac * 100}%` }} /></div>
          </div>
        )}
        {!access.unlocked && access.skillPoints > 0 && (
          <button className="pill accent sp-btn" onClick={() => { sessionStorage.setItem('lego-ai/guide-view', 'skills'); setTab('guide') }} title="Spend skill points in the Skill Tree">⭐ {access.skillPoints}</button>
        )}
      </div>
      {open && <ProfileModal onClose={() => setOpen(false)} />}
    </>
  )
}

function ProfileModal({ onClose }: { onClose: () => void }) {
  const { progress, access, setAvatar, chooseMode, reset, setOverviewOpen } = useLearn()
  const { setTab } = useStore()
  const s = progress.stats
  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal" style={{ width: 'min(640px, 100%)' }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <span style={{ fontSize: 28 }}>{progress.avatar}</span>
          <div className="grow">
            <h2>{access.unlocked ? 'Expert mode' : `Level ${access.level} · ${levelFor(progress.xp).title}`}</h2>
            <div className="small muted">{access.unlocked ? 'Everything unlocked' : `${progress.xp} XP`}</div>
          </div>
          <button className="ghost" onClick={onClose}>✕</button>
        </div>
        <div className="modal-body col" style={{ gap: 16 }}>
          <div>
            <h4 style={{ marginBottom: 8 }}>Choose your avatar</h4>
            <div className="avatar-grid">
              {AVATARS.map((a) => {
                const ok = access.unlocked || access.level >= a.level
                return (
                  <button key={a.icon} className={`avatar-opt ${progress.avatar === a.icon ? 'on' : ''}`} disabled={!ok} onClick={() => setAvatar(a.icon)} title={ok ? a.name : `Unlocks at level ${a.level}`}>
                    <span style={{ fontSize: 26, filter: ok ? 'none' : 'grayscale(1) opacity(0.3)' }}>{a.icon}</span>
                    <span className="tiny muted">{ok ? a.name : `Lv ${a.level}`}</span>
                  </button>
                )
              })}
            </div>
          </div>
          <div className="kpis">
            <div className="kpi"><div className="k">Runs</div><div className="v">{s.runs}</div></div>
            <div className="kpi"><div className="k">Lessons</div><div className="v">{s.lessons.length}/{LESSONS.length}</div></div>
            <div className="kpi"><div className="k">Badges</div><div className="v">{Object.keys(progress.achievements).length}</div></div>
            <div className="kpi"><div className="k">Best MNIST</div><div className="v">{s.best.mnist ? `${(s.best.mnist * 100).toFixed(1)}%` : '—'}</div></div>
          </div>
          <div className="card col" style={{ gap: 8 }}>
            <h4>Learning mode</h4>
            {access.unlocked ? (
              <>
                <div className="small muted">You’re in expert mode: every block, page and setting is available and all badges are yours.</div>
                <div className="row wrap">
                  <button onClick={() => { onClose(); setOverviewOpen(true) }}>Replay the quick tour</button>
                  <button onClick={() => { if (confirm('Switch to guided learning? You’ll start at level 1 with features unlocking as you go.')) { reset(); setTimeout(() => chooseMode('guided'), 50); setTab('lab'); onClose() } }}>Switch to guided learning</button>
                </div>
              </>
            ) : (
              <>
                <div className="small muted">Already know your way around? Unlock every block, page and setting at once. You’ll get all badges and the XP bar goes away.</div>
                <div className="row wrap">
                  <button className="primary" onClick={() => { if (confirm('Unlock everything? This skips the guided path.')) { chooseMode('unlocked'); onClose(); setOverviewOpen(true) } }}>🔓 Unlock everything</button>
                  <button className="danger" onClick={() => { if (confirm('Reset all XP, levels, skills and badges? (Your trained models are kept.)')) { reset(); onClose() } }}>Reset progress</button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Rewards & level-up
// ---------------------------------------------------------------------------
export function Rewards() {
  const { rewards, dismissReward } = useLearn()
  return (
    <div className="rewards">
      {rewards.map((r) => (
        <div key={r.key} className={`reward ${r.kind}`} onClick={() => dismissReward(r.key)}>
          {r.kind === 'achievement' ? (
            <>
              <div className="reward-icon">{r.icon}</div>
              <div className="grow">
                <div className="tiny" style={{ fontWeight: 700, letterSpacing: '0.08em', color: 'var(--accent)' }}>ACHIEVEMENT UNLOCKED</div>
                <div style={{ fontWeight: 700 }}>{r.title}</div>
                <div className="small muted">{r.detail}</div>
              </div>
            </>
          ) : (
            <span className="grow small">{r.kind === 'quest' ? `✅ Quest complete: ${r.title}` : r.title}</span>
          )}
          {r.xp > 0 && <span className="xp-gain">+{r.xp} XP</span>}
        </div>
      ))}
    </div>
  )
}

export function LevelUpModal() {
  const { levelUp, closeLevelUp, openLesson } = useLearn()
  const { setTab } = useStore()
  if (!levelUp || !levelUp.level) return null
  const L = LEVELS.find((l) => l.level === levelUp.level)!
  const avatar = AVATARS.find((a) => a.level === L.level)
  const lessons = LESSONS.filter((l) => l.unlock?.level === L.level || (l.unlock?.feature && L.unlocks.includes(l.unlock.feature)))
  const firstTab = L.unlocks.includes('builder') ? 'build' : L.unlocks.includes('train.page') ? 'train' : L.unlocks.includes('skilltree') ? 'guide' : L.unlocks.includes('custom.page') ? 'custom' : null
  return (
    <div className="modal-back" onClick={closeLevelUp}>
      <div className="modal levelup" onClick={(e) => e.stopPropagation()}>
        <div className="levelup-burst">
          <div className="tiny" style={{ letterSpacing: '0.2em', fontWeight: 800, color: 'var(--accent)' }}>LEVEL UP!</div>
          <div className="levelup-num">{L.level}</div>
          <h2>{L.title}</h2>
        </div>
        <div className="modal-body col" style={{ gap: 12 }}>
          {L.news.length > 0 && (
            <div>
              <h4 style={{ marginBottom: 6 }}>Unlocked</h4>
              <ul className="unlock-list">{L.news.map((n) => <li key={n}>{n}</li>)}</ul>
            </div>
          )}
          {L.level >= 6 && <div className="note-box">⭐ +2 skill points to spend in the Skill Tree.</div>}
          {avatar && <div className="small">New avatar: <span style={{ fontSize: 22 }}>{avatar.icon}</span> {avatar.name}</div>}
          {lessons.length > 0 && (
            <div className="row wrap">
              {lessons.map((l) => <button key={l.id} className="sm" onClick={() => { closeLevelUp(); openLesson(l.id) }}>📖 {l.title}</button>)}
            </div>
          )}
          <div className="row">
            <span className="spacer" />
            {firstTab && <button onClick={() => { closeLevelUp(); setTab(firstTab) }}>Take me there</button>}
            <button className="primary" onClick={closeLevelUp}>Let’s go!</button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Spotlight: ring around a [data-tour] element
// ---------------------------------------------------------------------------
export function Spotlight() {
  const { spot, setSpot } = useLearn()
  const [rect, setRect] = useState<DOMRect | null>(null)
  useLayoutEffect(() => {
    if (!spot) return setRect(null)
    const measure = () => {
      const el = document.querySelector(`[data-tour="${spot.id}"]`)
      setRect(el ? el.getBoundingClientRect() : null)
    }
    measure()
    const h = setInterval(measure, 250)
    const t = setTimeout(() => setSpot(null), 20000)
    return () => { clearInterval(h); clearTimeout(t) }
  }, [spot, setSpot])
  if (!spot || !rect) return null
  const below = rect.bottom + 140 < window.innerHeight
  const tipTop = below ? rect.bottom + 12 : Math.max(8, rect.top - 120)
  const tipLeft = Math.min(Math.max(8, rect.left), window.innerWidth - 340)
  return (
    <>
      <div className="spot-ring" style={{ left: rect.left - 6, top: rect.top - 6, width: rect.width + 12, height: rect.height + 12 }} />
      {spot.text && (
        <div className="spot-tip" style={{ left: tipLeft, top: tipTop }}>
          <span className="grow">{spot.text}</span>
          <button className="ghost sm" onClick={() => setSpot(null)}>✕</button>
        </div>
      )}
    </>
  )
}

// ---------------------------------------------------------------------------
// Tutor: the quest checklist for the current level
// ---------------------------------------------------------------------------
export function Tutor() {
  const { progress, access, openLesson, setSpot, ready } = useLearn()
  const { setTab } = useStore()
  const [open, setOpen] = useState(() => localStorage.getItem('lego-ai/tutor-open') !== '0')
  const [showOld, setShowOld] = useState(false)
  useEffect(() => localStorage.setItem('lego-ai/tutor-open', open ? '1' : '0'), [open])
  if (!ready || progress.mode !== 'guided') return null
  const level = access.level
  const quests = QUESTS.filter((q) => q.level === level)
  const backlog = QUESTS.filter((q) => q.level < level && !progress.quests.includes(q.id))
  const done = quests.filter((q) => progress.quests.includes(q.id)).length
  const next = LEVELS.find((l) => l.level === level + 1)
  const show = (q: Quest) => {
    if (q.go?.lesson) openLesson(q.go.lesson)
    if (q.go?.tab) setTab(q.go.tab)
    if (q.go?.spot) setTimeout(() => setSpot({ id: q.go!.spot!, text: q.how }), 150)
  }
  return (
    <div className={`tutor ${open ? 'open' : ''}`}>
      <button className="tutor-head" onClick={() => setOpen(!open)}>
        <span>🎯</span>
        <b className="grow" style={{ textAlign: 'left' }}>Level {level} quests</b>
        <span className="tiny mono">{done}/{quests.length}</span>
        <span className="tiny">{open ? '▾' : '▴'}</span>
      </button>
      {open && (
        <div className="tutor-body">
          {quests.filter((q) => !progress.quests.includes(q.id)).map((q) => (
            <div key={q.id} className="quest">
              <span className="quest-check" />
              <div className="grow">
                <div className="quest-title">{q.title} <span className="tiny mono" style={{ color: 'var(--accent)' }}>+{q.xp}</span></div>
                <div className="tiny muted">{q.how}</div>
              </div>
              {q.go && <button className="sm" onClick={() => show(q)}>Show me</button>}
            </div>
          ))}
          {done > 0 && <div className="tiny ok" style={{ padding: '6px 2px' }}>✓ {done} completed this level</div>}
          {backlog.length > 0 && (
            <>
              <button className="ghost sm" style={{ width: '100%', justifyContent: 'flex-start' }} onClick={() => setShowOld(!showOld)}>
                {showOld ? '▾' : '▸'} {backlog.length} optional quest{backlog.length > 1 ? 's' : ''} from earlier levels
              </button>
              {showOld && backlog.map((q) => (
                <div key={q.id} className="quest">
                  <span className="quest-check" />
                  <div className="grow"><div className="quest-title">{q.title} <span className="tiny mono" style={{ color: 'var(--accent)' }}>+{q.xp}</span></div></div>
                  {q.go && <button className="sm" onClick={() => show(q)}>Show me</button>}
                </div>
              ))}
            </>
          )}
          {next ? (
            <div className="tiny muted tutor-next">
              <b>Level {next.level}</b> ({next.xp - progress.xp} XP to go): {next.news[0] ?? 'new avatar'}
            </div>
          ) : level >= MAX_LEVEL && <div className="tiny muted tutor-next">Max level — legend!</div>}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// First-run welcome
// ---------------------------------------------------------------------------
export function Onboarding() {
  const { progress, ready, chooseMode, openLesson, setOverviewOpen } = useLearn()
  const { setTab } = useStore()
  const [step, setStep] = useState(0)
  if (!ready || progress.mode) return null
  return (
    <div className="modal-back onboarding-back">
      <div className="modal onboarding">
        {step === 0 ? (
          <div className="modal-body">
            <div className="brand-logo big">{['#ef4444', '#facc15', '#3b82f6', '#22c55e'].map((c) => <span key={c} style={{ background: c }} />)}</div>
            <h1 style={{ textAlign: 'center', marginTop: 12 }}>Welcome to Lego AI</h1>
            <p className="muted" style={{ textAlign: 'center', maxWidth: 520, margin: '8px auto 24px' }}>Build neural networks out of blocks, train them, and learn how modern AI works along the way.</p>
            <div className="mode-cards">
              <button className="mode-card" onClick={() => setStep(1)}>
                <div style={{ fontSize: 40 }}>🐣</div>
                <h2>I’m new to AI</h2>
                <p className="small muted">Start simple. Learn step by step with lessons, quests and animations. New blocks and features unlock as you level up.</p>
                <span className="pill accent">Recommended for beginners</span>
              </button>
              <button className="mode-card" onClick={() => { chooseMode('unlocked'); setTab('build'); setOverviewOpen(true) }}>
                <div style={{ fontSize: 40 }}>🔓</div>
                <h2>Unlock everything</h2>
                <p className="small muted">I know my way around neural networks. Give me every block, setting and page now — plus a 1-minute tour and the guidebook.</p>
                <span className="pill">Expert mode</span>
              </button>
            </div>
          </div>
        ) : (
          <div className="modal-body col" style={{ gap: 14 }}>
            <h1>How this works</h1>
            <div className="how-grid">
              <div className="how"><span>🧪</span><div><b>Start in the Lab.</b> Your first model is one choice away. It trains in seconds.</div></div>
              <div className="how"><span>⭐</span><div><b>Earn XP</b> by trying things, reading lessons and hitting goals. Your level is at the top of the screen.</div></div>
              <div className="how"><span>🔓</span><div><b>Level up to unlock</b> the Builder, new blocks, training controls, datasets and more.</div></div>
              <div className="how"><span>🎯</span><div><b>Quests</b> (bottom left) tell you what to try next. Press <i>Show me</i> if you’re stuck.</div></div>
              <div className="how"><span>📖</span><div><b>The Guidebook</b> explains every idea and block with animations, code and the original papers.</div></div>
              <div className="how"><span>🏆</span><div><b>Achievements</b> reward milestones like “get 98% on MNIST”.</div></div>
            </div>
            <div className="row">
              <button className="ghost" onClick={() => setStep(0)}>← Back</button>
              <span className="spacer" />
              <button className="primary big" onClick={() => { chooseMode('guided'); setTab('lab'); openLesson('start.mnist') }}>First up: meet MNIST →</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Expert-mode quick tour
// ---------------------------------------------------------------------------
const TOUR = [
  { tab: 'build', spot: 'palette', title: 'Block palette', text: 'Every block, grouped by kind and colour. Filter by task (image / text). Drag onto the stack or click to add. Custom Python blocks appear here too.' },
  { tab: 'build', spot: 'stack', title: 'The stack', text: 'Data flows up from the input (bottom) to the output (top). Shapes are inferred and shown on the arrows. Drag to reorder, click to edit, drop on the bin to delete.' },
  { tab: 'build', spot: 'templates', title: 'Templates & JSON', text: 'Start from a proven architecture for each dataset, or import/export reproducible .lego.json configs.' },
  { tab: 'train', spot: 'hyperparams', title: 'Training', text: 'All hyperparameters, a batch-size memory checker, resume from checkpoints, and live loss/accuracy charts.' },
  { tab: 'train', spot: 'optuna', title: 'Optuna', text: 'Switch on to give ranges instead of values and run a pruned TPE search. Every trial is a reproducible run.' },
  { tab: 'leaderboard', spot: 'leaderboard', title: 'Leaderboard', text: 'Rank runs, inspect curves and configs, retrain from scratch (bit-exact reproduction check), resume, download weights.' },
  { tab: 'custom', spot: 'custom-new', title: 'Custom blocks', text: 'Write nn.Modules with @lego_block and they join the palette. Their source is embedded in run configs.' },
  { tab: 'history', spot: 'timeline', title: 'History of AI', text: 'A self-running tour of 12 landmark architectures on MNIST, with papers.' },
  { tab: 'guide', spot: 'guide-nav', title: 'Guidebook', text: 'Lessons for every page, action and block — animations, annotated PyTorch code, papers. Each page’s “?” button jumps to its guide.' },
]

export function OverviewTour() {
  const { overviewOpen, setOverviewOpen, update } = useLearn()
  const { setTab } = useStore()
  const { setSpot } = useLearn()
  const [i, setI] = useState(0)
  useEffect(() => {
    if (!overviewOpen) return
    const s = TOUR[i]
    setTab(s.tab)
    const h = setTimeout(() => setSpot({ id: s.spot }), 200)
    return () => clearTimeout(h)
  }, [i, overviewOpen, setTab, setSpot])
  if (!overviewOpen) return null
  const s = TOUR[i]
  const finish = () => {
    setOverviewOpen(false)
    setSpot(null)
    setI(0)
    update((p) => ({ ...p, tourDone: true }))
    setTab('guide')
  }
  return (
    <div className="tour-card">
      <div className="tiny muted">Quick tour · {i + 1} / {TOUR.length}</div>
      <h3 style={{ margin: '4px 0 6px' }}>{s.title}</h3>
      <div className="small">{s.text}</div>
      <div className="row" style={{ marginTop: 12 }}>
        <button className="ghost sm" onClick={finish}>Skip</button>
        <span className="spacer" />
        {i > 0 && <button className="sm" onClick={() => setI(i - 1)}>Back</button>}
        {i < TOUR.length - 1 ? <button className="sm primary" onClick={() => setI(i + 1)}>Next</button> : <button className="sm primary" onClick={finish}>Open the guidebook</button>}
      </div>
    </div>
  )
}
