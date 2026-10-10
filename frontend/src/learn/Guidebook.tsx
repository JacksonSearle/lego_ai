import { useEffect, useMemo, useState } from 'react'
import { Brick } from '../components/Brick'
import { newId, useStore } from '../store'
import { defaultParams } from '../util'
import { ACHIEVEMENTS, GRANTED_SKILLS, LEVELS, SKILLS, type Branch, type SkillNode } from './curriculum'
import { blockAllowed, featureLevel, skillForBlock, type Access } from './engine'
import { DiagramBoundary } from './diagrams/core'
import { useLearn } from './LearnContext'
import { LESSONS, SECTIONS, blockOfLesson, getLesson, type Lesson } from './lessons'

// ---------------------------------------------------------------------------
// Locking
// ---------------------------------------------------------------------------
export function lessonLock(l: Lesson, a: Access): string | null {
  if (a.unlocked) return null
  if (l.block) {
    const type = blockOfLesson(l)!
    if (blockAllowed(a, type)) return null
    const node = skillForBlock(type)
    if (!node) return null
    const grantedAt = LEVELS.find((lv) => lv.grants?.includes(node.id))?.level
    if (grantedAt) return `Unlocks at level ${grantedAt}`
    return a.has('skilltree') ? `Unlock “${node.title}” in the Skill Tree` : 'Unlocks with the Skill Tree (level 6)'
  }
  if (l.unlock?.level && a.level < l.unlock.level) return `Unlocks at level ${l.unlock.level}`
  if (l.unlock?.feature && !a.has(l.unlock.feature)) return `Unlocks at level ${featureLevel(l.unlock.feature)}`
  return null
}

const TAG_LABEL = { image: '🖼️ image', text: '📝 text', general: '🧱 general' } as const

// ---------------------------------------------------------------------------
// Broken-model challenge
// ---------------------------------------------------------------------------
export function useStartChallenge() {
  const { setDraft, setTab, toast } = useStore()
  const { update, setSpot, openLesson } = useLearn()
  return () => {
    setDraft((d) => ({
      ...d,
      name: 'Broken model (fix me!)',
      dataset: 'mnist',
      origin: 'challenge',
      edited: false,
      blocks: [
        { id: newId(), type: 'linear', params: { out_features: 128, activation: 'relu', bias: true }, repeat: 1 },
        { id: newId(), type: 'linear', params: { out_features: 7, activation: 'none', bias: true }, repeat: 1 },
      ],
    }))
    update((p) => ({ ...p, challenge: 'active' }))
    openLesson(null)
    setTab('build')
    setSpot({ id: 'stack', text: 'Two problems: the image reaches a Linear layer unflattened, and the model outputs 7 numbers instead of 10. Fix both!' })
    toast('Challenge started: make every block happy and the output match 10 classes.', 'info')
  }
}

// ---------------------------------------------------------------------------
// Lesson view
// ---------------------------------------------------------------------------
export function LessonView({ id, onNavigate }: { id: string; onNavigate?: (id: string) => void }) {
  const lesson = getLesson(id)
  const { blocks: specs, setDraft, setTab, toast } = useStore()
  const { progress, access, track, openLesson } = useLearn()
  const startChallenge = useStartChallenge()
  const [answer, setAnswer] = useState<number | null>(null)
  useEffect(() => setAnswer(null), [id])
  if (!lesson) return <div className="empty">Lesson not found.</div>
  const lock = lessonLock(lesson, access)
  const read = progress.stats.lessons.includes(lesson.id)
  const quizDone = progress.stats.quizzes.includes(lesson.id)
  const blockType = blockOfLesson(lesson)
  const spec = blockType ? specs[blockType] : undefined
  const idx = LESSONS.findIndex((l) => l.id === id)
  const prev = LESSONS[idx - 1]
  const next = LESSONS[idx + 1]

  if (lock) {
    return (
      <div className="lesson">
        <h1>🔒 {lesson.title}</h1>
        <p className="muted">{lesson.summary}</p>
        <div className="note-box">{lock}.</div>
      </div>
    )
  }

  const addBlock = () => {
    if (!blockType || !spec) return
    if (!access.has('builder')) return toast('The Builder unlocks at level 3.', 'error')
    setDraft((d) => {
      const head = d.blocks.findIndex((b) => b.type === 'classifier')
      const at = head >= 0 ? head : d.blocks.length
      const b = { id: newId(), type: blockType, params: defaultParams(spec), repeat: 1 }
      return { ...d, edited: true, blocks: [...d.blocks.slice(0, at), b, ...d.blocks.slice(at)] }
    })
    openLesson(null)
    setTab('build')
    toast(`Added ${spec.name} to your model`, 'ok')
  }

  const diagrams = lesson.diagrams ?? (lesson.diagram ? [lesson.diagram] : [])
  return (
    <article className="lesson">
      <div className="lesson-kicker">
        {SECTIONS.find((s) => s.id === lesson.section)?.icon} {SECTIONS.find((s) => s.id === lesson.section)?.title}
        {lesson.year && <span className="pill">{lesson.year}</span>}
        {lesson.tags?.map((t) => <span key={t} className={`pill tag-${t}`}>{TAG_LABEL[t]}</span>)}
        {read && <span className="pill ok">✓ read</span>}
      </div>
      <h1>{lesson.title}</h1>
      <p className="lesson-summary">{lesson.summary}</p>
      {spec && (
        <div style={{ width: 300, margin: '14px 0 6px', paddingRight: 40 }}>
          <Brick spec={spec} inst={{ id: 'x', type: spec.type, params: defaultParams(spec), repeat: 1 }} compact />
        </div>
      )}
      {diagrams.map((D, i) => <DiagramBoundary key={`${id}-${i}`}><D /></DiagramBoundary>)}
      <div className="lesson-body">{lesson.body}</div>

      {spec && spec.params.length > 0 && (
        <section>
          <h3>Its settings</h3>
          <div className="param-table">
            {spec.params.map((p) => (
              <div key={p.name} className="param-row">
                <div><b>{p.label}</b><div className="tiny faint mono">default: {String(p.default)}</div></div>
                <div className="small">{lesson.paramNotes?.[p.name] ?? p.help}{lesson.paramNotes?.[p.name] && p.help && <div className="tiny muted" style={{ marginTop: 3 }}>{p.help}</div>}</div>
              </div>
            ))}
          </div>
        </section>
      )}

      {lesson.code && (
        <section>
          <h3>{lesson.block ? 'In PyTorch' : 'The code, line by line'}</h3>
          <div className="code-walk">
            {lesson.code.map((c, i) => (
              <div key={i} className="code-line">
                <pre>{c.code || ' '}</pre>
                <div className="code-note">{c.note}</div>
              </div>
            ))}
          </div>
        </section>
      )}

      {lesson.paper && (
        <a className="paper-card" href={lesson.paper.url} target="_blank" rel="noreferrer">
          <div className="tiny muted" style={{ fontWeight: 700, letterSpacing: '0.06em' }}>📄 READ THE ORIGINAL PAPER ↗</div>
          <div style={{ fontWeight: 650, marginTop: 4 }}>{lesson.paper.title}</div>
          <div className="small muted">{lesson.paper.authors}, {lesson.paper.year}</div>
        </a>
      )}

      {lesson.quiz && (
        <section className="quiz">
          <h3>Quick check {quizDone && <span className="pill ok">✓ answered</span>}</h3>
          <p>{lesson.quiz.q}</p>
          <div className="quiz-options">
            {lesson.quiz.options.map((o, i) => {
              const picked = answer === i
              const cls = picked ? (i === lesson.quiz!.answer ? 'right' : 'wrong') : quizDone && i === lesson.quiz!.answer ? 'right' : ''
              return (
                <button key={i} className={`quiz-opt ${cls}`} onClick={() => { setAnswer(i); if (i === lesson.quiz!.answer) track({ type: 'quiz', id: lesson.id }) }}>
                  {o}
                </button>
              )
            })}
          </div>
          {answer !== null && (
            <div className={answer === lesson.quiz.answer ? 'note-box' : 'error-box'} style={{ marginTop: 8 }}>
              {answer === lesson.quiz.answer ? `Correct! ${lesson.quiz.explain}` : 'Not quite — have another look at the lesson and try again.'}
            </div>
          )}
        </section>
      )}

      <div className="row wrap lesson-actions">
        {lesson.action === 'challenge' && <button className="primary" onClick={startChallenge}>🧩 Start the challenge</button>}
        {lesson.action === 'addBlock' && spec && access.has('builder') && <button onClick={addBlock}>+ Add {spec.name} to my model</button>}
        {lesson.action && typeof lesson.action === 'object' && (
          <button onClick={() => { openLesson(null); setTab((lesson.action as { tab: string }).tab) }}>{(lesson.action as { label: string }).label} →</button>
        )}
        <span className="spacer" />
        {!read ? (
          <button className="primary" onClick={() => track({ type: 'lesson', id: lesson.id })}>✓ Got it{progress.mode === 'guided' ? ' (+10 XP)' : ''}</button>
        ) : <span className="small ok">✓ You’ve read this</span>}
      </div>

      {onNavigate && (
        <div className="row lesson-nav">
          {prev ? <button className="ghost" onClick={() => onNavigate(prev.id)}>← {prev.title}</button> : <span />}
          <span className="spacer" />
          {next && <button className="ghost" onClick={() => onNavigate(next.id)}>{next.title} →</button>}
        </div>
      )}
    </article>
  )
}

export function LessonModal() {
  const { lesson, openLesson } = useLearn()
  const [id, setId] = useState(lesson)
  useEffect(() => setId(lesson), [lesson])
  if (!lesson || !id) return null
  return (
    <div className="modal-back" onClick={() => openLesson(null)}>
      <div className="modal lesson-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <span className="small muted">📖 Guidebook</span>
          <span className="spacer" />
          <button className="ghost" onClick={() => openLesson(null)}>✕</button>
        </div>
        <div className="modal-body"><LessonView id={id} onNavigate={setId} /></div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Skill tree
// ---------------------------------------------------------------------------
const BRANCHES: { id: Branch; title: string; icon: string; color: string; desc: string }[] = [
  { id: 'image', title: 'Image', icon: '🖼️', color: '#f97316', desc: 'Seeing: from convolutions to vision transformers.' },
  { id: 'text', title: 'Text', icon: '📝', color: '#ec4899', desc: 'Reading & writing: from embeddings to transformers.' },
  { id: 'general', title: 'General', icon: '🧱', color: '#60a5fa', desc: 'Tools that help any model.' },
]

export function SkillTree() {
  const { access, unlockSkill } = useLearn()
  const { blocks: specs, toast } = useStore()
  const { openLesson } = useLearn()
  const [sel, setSel] = useState<SkillNode | null>(null)
  const granted = new Set(GRANTED_SKILLS(access.level))
  const state = (n: SkillNode) => (access.skills.has(n.id) ? 'owned' : n.requires.every((r) => access.skills.has(r)) ? 'open' : 'locked')
  const CW = 150, RH = 92

  return (
    <div className="skill-layout">
      <div>
        <div className="row" style={{ marginBottom: 12 }}>
          <h2 className="grow">Skill Tree</h2>
          {!access.unlocked && (access.has('skilltree')
            ? <span className={`pill ${access.skillPoints > 0 ? 'accent' : ''}`} data-tour="skillpoints">⭐ {access.skillPoints} skill point{access.skillPoints === 1 ? '' : 's'}</span>
            : <span className="pill">🔒 Opens at level 6</span>)}
        </div>
        <div className="skill-branches">
          {BRANCHES.map((b) => {
            const nodes = SKILLS.filter((s) => s.branch === b.id)
            const rows = Math.max(...nodes.map((n) => n.row)) + 1
            const cols = Math.max(...nodes.map((n) => n.col)) + 1
            const pos = (n: SkillNode) => ({ x: 20 + n.col * CW, y: 16 + n.row * RH })
            return (
              <div key={b.id} className="branch" style={{ borderColor: b.color + '55' }}>
                <div className="branch-head" style={{ color: b.color }}>{b.icon} {b.title}</div>
                <div className="tiny muted" style={{ marginBottom: 8 }}>{b.desc}</div>
                <div style={{ position: 'relative', height: rows * RH + 10, width: cols * CW + 20 }}>
                  <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                    {nodes.flatMap((n) => n.requires.map((r) => {
                      const from = nodes.find((x) => x.id === r)
                      if (!from) return null
                      const a = pos(from), c = pos(n)
                      return <line key={`${r}-${n.id}`} x1={a.x + 60} y1={a.y + 58} x2={c.x + 60} y2={c.y} stroke={access.skills.has(r) ? b.color : '#34405c'} strokeWidth={2} />
                    }))}
                  </svg>
                  {nodes.map((n) => {
                    const st = state(n)
                    const p = pos(n)
                    const cross = n.requires.filter((r) => !nodes.some((x) => x.id === r))
                    return (
                      <button key={n.id} className={`skill-node ${st} ${sel?.id === n.id ? 'sel' : ''}`} style={{ left: p.x, top: p.y, ['--bc' as string]: b.color }} onClick={() => setSel(n)}>
                        <div className="skill-year">{n.year}</div>
                        <div className="skill-title">{st === 'owned' ? '' : st === 'locked' ? '🔒 ' : ''}{n.title}</div>
                        {cross.length > 0 && <div className="tiny muted">+ {cross.map((c) => SKILLS.find((s) => s.id === c)?.title).join(', ')}</div>}
                        {granted.has(n.id) && !access.unlocked && <div className="tiny" style={{ color: b.color }}>free</div>}
                      </button>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>
      </div>
      <aside className="panel skill-detail">
        {sel ? (
          <div className="panel-body col" style={{ gap: 10 }}>
            <div className="tiny muted">{BRANCHES.find((b) => b.id === sel.branch)?.icon} {sel.branch} · {sel.year}</div>
            <h2>{sel.title}</h2>
            <p className="small muted" style={{ margin: 0 }}>{sel.blurb}</p>
            <h4>Unlocks</h4>
            <div className="col" style={{ gap: 12, paddingRight: 40 }}>
              {sel.blocks.map((t) => specs[t] && <Brick key={t} spec={specs[t]} inst={{ id: t, type: t, params: defaultParams(specs[t]), repeat: 1 }} compact />)}
            </div>
            {sel.datasets && <div className="small">Datasets: <b>{sel.datasets.join(', ')}</b></div>}
            {sel.requires.length > 0 && <div className="small muted">Requires: {sel.requires.map((r) => SKILLS.find((s) => s.id === r)?.title).join(', ')}</div>}
            <div className="row wrap">
              {state(sel) === 'owned' ? <span className="pill ok">✓ Unlocked</span> : (
                <button className="primary" disabled={access.unlocked} onClick={() => {
                  const err = unlockSkill(sel.id)
                  if (err) toast(err, 'error')
                  else toast(`Unlocked ${sel.title}! New blocks are in the Builder palette.`, 'ok')
                }}>⭐ Unlock (1 point)</button>
              )}
              <button onClick={() => openLesson(sel.lesson)}>📖 Lesson</button>
            </div>
          </div>
        ) : (
          <div className="panel-body small muted">
            Click a skill to see what it unlocks. Each costs one skill point; you earn 2 per level from level 6. Branches roughly follow history — older ideas first — so each new block shows what the previous ones were missing.
          </div>
        )}
      </aside>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Achievements
// ---------------------------------------------------------------------------
export function Achievements() {
  const { progress } = useLearn()
  const earned = Object.keys(progress.achievements).length
  return (
    <div>
      <div className="row" style={{ marginBottom: 14 }}>
        <h2 className="grow">Achievements</h2>
        <span className="pill accent">{earned} / {ACHIEVEMENTS.length}</span>
      </div>
      <div className="badge-grid">
        {ACHIEVEMENTS.map((a) => {
          const at = progress.achievements[a.id]
          return (
            <div key={a.id} className={`badge ${at ? 'earned' : ''}`}>
              <div className="badge-icon">{a.icon}</div>
              <div className="grow">
                <div style={{ fontWeight: 650 }}>{a.title}</div>
                <div className="small muted">{a.desc}</div>
              </div>
              <div className="right">
                {progress.mode === 'guided' && <div className="tiny mono" style={{ color: 'var(--accent)' }}>+{a.xp} XP</div>}
                {at && <div className="tiny ok">✓ {new Date(at).toLocaleDateString()}</div>}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// The page
// ---------------------------------------------------------------------------
export function Guidebook() {
  const { access, progress } = useLearn()
  const [view, setView] = useState<'lessons' | 'skills' | 'achievements'>(() => (sessionStorage.getItem('lego-ai/guide-view') as 'lessons') || 'lessons')
  const [id, setId] = useState(() => sessionStorage.getItem('lego-ai/guide-lesson') || 'start.welcome')
  const [filter, setFilter] = useState('')
  useEffect(() => { sessionStorage.setItem('lego-ai/guide-view', view); sessionStorage.setItem('lego-ai/guide-lesson', id) }, [view, id])
  const bySection = useMemo(() => SECTIONS.map((s) => ({
    ...s, lessons: LESSONS.filter((l) => l.section === s.id && (!filter || l.title.toLowerCase().includes(filter.toLowerCase()))),
  })), [filter])
  const readCount = progress.stats.lessons.length

  return (
    <div className="guide-layout">
      <aside className="guide-nav" data-tour="guide-nav">
        <div className="seg" style={{ width: '100%', marginBottom: 12 }}>
          <button className={view === 'lessons' ? 'on' : ''} onClick={() => setView('lessons')}>📖 Lessons</button>
          <button className={view === 'skills' ? 'on' : ''} onClick={() => setView('skills')} data-tour="skilltree">🌳 Skills</button>
          <button className={view === 'achievements' ? 'on' : ''} onClick={() => setView('achievements')}>🏆</button>
        </div>
        <input placeholder="Search lessons…" value={filter} onChange={(e) => setFilter(e.target.value)} style={{ width: '100%', marginBottom: 10 }} />
        <div className="tiny muted" style={{ marginBottom: 8 }}>{readCount} of {LESSONS.length} lessons read</div>
        {bySection.map((s) => s.lessons.length > 0 && (
          <div key={s.id} className="guide-section">
            <h4>{s.icon} {s.title}</h4>
            {s.lessons.map((l) => {
              const lock = lessonLock(l, access)
              const read = progress.stats.lessons.includes(l.id)
              return (
                <div key={l.id} className={`guide-item ${id === l.id && view === 'lessons' ? 'active' : ''} ${lock ? 'locked' : ''}`} onClick={() => { setId(l.id); setView('lessons') }} title={lock ?? l.summary}>
                  <span className="grow">{lock ? '🔒 ' : ''}{l.title}</span>
                  {read && <span className="ok tiny">✓</span>}
                  {!read && !lock && progress.mode === 'guided' && <span className="dot-new" />}
                </div>
              )
            })}
          </div>
        ))}
      </aside>
      <main className="guide-main">
        {view === 'lessons' && <LessonView id={id} onNavigate={setId} />}
        {view === 'skills' && <SkillTree />}
        {view === 'achievements' && <Achievements />}
      </main>
    </div>
  )
}
