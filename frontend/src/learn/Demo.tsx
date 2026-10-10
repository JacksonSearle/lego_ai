/**
 * Teacher demo: a narrated walkthrough of the whole app in about ten minutes.
 *
 * It runs in a temporary profile (so nobody's real progress is touched), jumps the learner's
 * level forward step by step to show what unlocks when, and really trains a few small models.
 * Leaving the demo switches back to the previous profile, restores the model that was in the
 * builder, and offers to delete the models the demo trained.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { newId, useStore, type Draft } from '../store'
import type { BlockInst } from '../types'
import { AVATARS, LEVELS, QUESTS, newProgress, type Progress } from './curriculum'
import { useLearn } from './LearnContext'

const KEY = 'lego-ai/demo'
const DRAFT_KEY = 'lego-ai/draft'

interface DemoState { returnTo: string; runIds: string[]; draft: string | null; step: number }

const readState = (): DemoState | null => {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? 'null')
  } catch {
    return null
  }
}

export async function startDemo() {
  const [profiles, runs] = await Promise.all([api.profiles(), api.runs()])
  const state: DemoState = { returnTo: profiles.active, runIds: runs.map((r) => r.id), draft: localStorage.getItem(DRAFT_KEY), step: 0 }
  await api.createProfile('Teacher demo', { demo: true, progress: { ...newProgress(), mode: 'guided' } })
  localStorage.setItem(KEY, JSON.stringify(state))
  localStorage.removeItem(DRAFT_KEY)
  sessionStorage.clear()
  location.hash = 'lab'
  location.reload()
}

const B = (type: string, params: Record<string, unknown> = {}, repeat = 1): BlockInst => ({ id: newId(), type, params, repeat })
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms))

function clickButton(match: (b: HTMLElement) => boolean) {
  const b = [...document.querySelectorAll<HTMLElement>('button')].find(match)
  b?.click()
  return !!b
}

/** A guided-mode progress as if the learner had just reached `level` with these skills. */
function progressAt(cur: Progress, level: number, skills: string[] = []): Progress {
  const L = LEVELS.find((l) => l.level === level)!
  return {
    ...cur,
    mode: 'guided',
    xp: Math.max(L.xp + 30, 0),
    quests: QUESTS.filter((q) => q.level < level).map((q) => q.id),
    skills,
    avatar: [...AVATARS].reverse().find((a) => a.level <= level)!.icon,
    levelSeen: level,
  }
}

interface Ctx {
  alive: () => boolean
  level: (n: number, skills?: string[]) => void
  tab: (t: string) => void
  model: (name: string, dataset: string, blocks: BlockInst[], training?: Record<string, unknown>, extra?: Partial<Draft>) => void
  lesson: (id: string | null) => void
  spot: (id: string | null, text?: string) => void
  guide: (v: 'lessons' | 'skills') => void
  unlockAll: () => void
}

interface Step { title: string; say: string; seconds: number; run: (c: Ctx) => Promise<void> | void }

const MLP = () => [B('flatten'), B('linear', { out_features: 128, activation: 'relu', bias: true }, 2), B('classifier')]
const CNN = () => [
  B('conv2d', { out_channels: 32, batch_norm: true }), B('maxpool2d'), B('conv2d', { out_channels: 64, batch_norm: true }), B('maxpool2d'),
  B('flatten'), B('linear', { out_features: 128 }), B('dropout', { p: 0.3 }), B('classifier'),
]

const STEPS: Step[] = [
  {
    title: 'Welcome to Lego AI', seconds: 25,
    say: 'Lego AI teaches how neural networks work by letting students build and train them from blocks. New learners start here at level 1 with almost everything locked. This demo fast-forwards through what a student unlocks over several sessions.',
    run: (c) => { c.level(1); c.tab('lab'); c.lesson(null) },
  },
  {
    title: 'Every idea has a lesson', seconds: 35,
    say: 'The first thing a student sees is this lesson about the MNIST handwritten-digit dataset: real examples, and what an image looks like to a computer (a grid of 784 numbers). Lessons end with a quick quiz and earn experience points.',
    run: (c) => { c.tab('lab'); c.lesson('start.mnist') },
  },
  {
    title: 'The Lab: a first model in one click', seconds: 40,
    say: 'The student makes one choice: how big the model’s hidden layer is. Training starts by itself and takes a few seconds. They see the accuracy, a learning curve, the model’s guesses on real digits, and a leaderboard of their models.',
    run: async (c) => {
      c.lesson(null); c.tab('lab'); await wait(900)
      if (c.alive()) clickButton((b) => b.classList.contains('width-opt') && b.textContent!.includes('Medium'))
    },
  },
  {
    title: 'Learning by comparing', seconds: 35,
    say: 'Now the “too small” model: only 2 neurons. It does far worse, and the page explains why. Trying the extremes teaches capacity better than a definition does. Notice the XP bar and achievement appear as they experiment.',
    run: async (c) => {
      c.tab('lab'); await wait(600)
      if (c.alive()) clickButton((b) => b.classList.contains('width-opt') && b.textContent!.includes('Too small'))
      await wait(9000)
      if (c.alive()) c.spot('xp', 'Experience points, level and achievements. Each level unlocks something new.')
    },
  },
  {
    title: 'Quests guide what to try next', seconds: 25,
    say: 'Each level has a short list of quests. “Show me” highlights exactly where to click, so students are never stuck. Completing quests is what earns the XP for the next level.',
    run: async (c) => {
      c.spot(null); await wait(300)
      if (c.alive()) clickButton((b) => b.classList.contains('quests-btn') && !b.classList.contains('on'))
    },
  },
  {
    title: 'Level 3: the Builder', seconds: 40,
    say: 'At level 3 the full builder opens. Blocks are dragged from the panel onto a stack that flows upward from the input to the output. Any block can be repeated, and the shape of the data after every block is shown on the arrows.',
    run: async (c) => {
      clickButton((b) => b.classList.contains('quests-btn') && b.classList.contains('on'))
      c.level(3); c.model('Two-layer network', 'mnist', MLP()); c.tab('build'); await wait(800)
      if (c.alive()) c.spot('palette', 'Only unlocked blocks are shown. More arrive as the student levels up.')
    },
  },
  {
    title: 'Level 4: shapes have to fit', seconds: 45,
    say: 'Students learn why tensor shapes must match by repairing a broken model. The red block explains the problem in plain words: an image is reaching a layer that needs a flat list. Watch it get fixed: a Flatten block goes underneath, and the output is set to 10 classes.',
    run: async (c) => {
      c.level(4)
      c.model('Broken model (fix me!)', 'mnist', [B('linear', { out_features: 128, activation: 'relu', bias: true }), B('linear', { out_features: 7, activation: 'none', bias: true })])
      c.tab('build'); await wait(700)
      if (c.alive()) c.spot('stack', 'Two problems: an unflattened image, and 7 outputs instead of 10.')
      await wait(14000)
      if (!c.alive()) return
      c.spot(null)
      c.model('Repaired model', 'mnist', [B('flatten'), B('linear', { out_features: 128, activation: 'relu', bias: true }), B('linear', { out_features: 10, activation: 'none', bias: true })])
    },
  },
  {
    title: 'Level 5: the Train page', seconds: 50,
    say: 'Training gets its own page with live loss and accuracy charts. Controls arrive gradually: first epochs and learning rate, later optimizers, batch size, schedules and more. Training can be paused and resumed exactly where it stopped.',
    run: async (c) => {
      c.spot(null); c.level(5); c.model('Two-layer network', 'mnist', MLP(), { epochs: 3, lr: 0.001 }); c.tab('train'); await wait(1200)
      if (c.alive()) clickButton((b) => b.classList.contains('big') && b.textContent!.includes('Train'))
    },
  },
  {
    title: 'The Guidebook: animations, code, papers', seconds: 45,
    say: 'Every block has a lesson: an animation of what it does, each setting explained, the matching PyTorch code annotated line by line, a link to the original paper and a quiz. This is the convolution lesson.',
    run: (c) => { c.level(6, ['conv', 'pooling']); c.guide('lessons'); sessionStorage.setItem('lego-ai/guide-lesson', 'block.conv2d'); c.tab('guide'); c.lesson('block.conv2d') },
  },
  {
    title: 'Level 6: the Skill Tree', seconds: 35,
    say: 'From level 6, students choose their own path. Skill points unlock new blocks along an image branch and a text branch, each in historical order, so they feel why each invention mattered: from convolutions to vision transformers, from embeddings to GPT-style transformers.',
    run: (c) => { c.lesson(null); c.level(6, ['conv', 'pooling']); c.guide('skills'); c.tab('guide') },
  },
  {
    title: 'A convolutional network', seconds: 60,
    say: 'With convolution unlocked, the student builds a small CNN and trains it. One epoch is enough to pass 98% on the digits their first model struggled with. That jump is the lesson.',
    run: async (c) => {
      c.level(7, ['conv', 'pooling', 'norms']); c.model('Simple CNN', 'mnist', CNN(), { epochs: 1, lr: 0.002 }); c.tab('build'); await wait(5000)
      if (!c.alive()) return
      c.tab('train'); await wait(1200)
      if (c.alive()) clickButton((b) => b.classList.contains('big') && b.textContent!.includes('Train'))
    },
  },
  {
    title: 'The Leaderboard', seconds: 30,
    say: 'Every model is saved and ranked. Students compare architectures, open any run to see its curves and settings, and can retrain a run from scratch to check they get exactly the same result: reproducibility as a habit.',
    run: (c) => { c.level(8, ['conv', 'pooling', 'norms']); c.tab('leaderboard') },
  },
  {
    title: 'Looking inside the model', seconds: 40,
    say: 'Accuracy is only a number. In Explore, students see which digits their model confuses, its most confident mistakes, and can draw a digit themselves to test it. Looking at examples is how real practitioners debug models.',
    run: (c) => { sessionStorage.setItem('lego-ai/explore-tab', 'model'); c.tab('explore') },
  },
  {
    title: 'Text models', seconds: 35,
    say: 'The text branch works the same way. This is a character-level LSTM for Shakespeare: an embedding, a recurrent layer and an output head. Later the student can swap the LSTM for transformer blocks and build a miniature GPT.',
    run: (c) => {
      c.level(8, ['conv', 'pooling', 'embedding', 'rnn', 'lstm'])
      c.model('Char LSTM', 'shakespeare', [B('embedding', { dim: 64 }), B('lstm', { hidden: 256, layers: 2, dropout: 0.1, bidirectional: false, return_sequences: true }), B('classifier')], {}, { seq_len: null })
      c.tab('build')
    },
  },
  {
    title: 'Automatic tuning with Optuna', seconds: 35,
    say: 'At level 10, students give ranges instead of exact values and let Optuna search for good hyperparameters, stopping weak trials early. It is the same tool used in research labs.',
    run: async (c) => {
      c.level(10, ['conv', 'pooling']); c.model('Two-layer network', 'mnist', MLP(), { epochs: 3 }); c.tab('train'); await wait(900)
      if (c.alive()) c.spot('optuna', 'Switch on, pick which settings to search, and run.')
    },
  },
  {
    title: 'Writing their own blocks', seconds: 30,
    say: 'At level 11, students write a block in Python with a line-by-line walkthrough. It appears in the builder next to the built-in ones. By this point they can read and modify real PyTorch.',
    run: (c) => { c.spot(null); c.level(11, ['conv', 'pooling']); c.tab('custom') },
  },
  {
    title: 'Everything unlocked', seconds: 30,
    say: 'Experienced users can skip the path and unlock everything at once: every block, dataset and setting, with a one-minute tour. That is the end of the demo. Press “Exit demo” to return to your own profile.',
    run: (c) => { c.unlockAll(); c.tab('build') },
  },
]

export function DemoPlayer() {
  const learn = useLearn()
  const store = useStore()
  const [state, setState] = useState<DemoState | null>(readState)
  const [verified, setVerified] = useState(false)
  const [step, setStep] = useState(() => readState()?.step ?? 0)
  const [playing, setPlaying] = useState(true)
  const [elapsed, setElapsed] = useState(0)
  const [exiting, setExiting] = useState(false)
  const [cleanup, setCleanup] = useState(true)
  const token = useRef(0)
  const L = useRef(learn)
  const S = useRef(store)
  L.current = learn
  S.current = store

  // Only play when the active profile really is the demo profile (the flag could be stale).
  useEffect(() => {
    if (!state) return
    api.profiles().then((p) => {
      if (p.profiles.find((x) => x.active)?.demo) setVerified(true)
      else {
        localStorage.removeItem(KEY)
        setState(null)
      }
    })
  }, [state])

  const active = !!state && verified && learn.ready && !!store.meta

  useEffect(() => {
    if (!active) return
    const mine = ++token.current
    setElapsed(0)
    localStorage.setItem(KEY, JSON.stringify({ ...state, step }))
    const ctx: Ctx = {
      alive: () => token.current === mine,
      level: (n, skills) => L.current.update((p) => progressAt(p, n, skills)),
      tab: (t) => S.current.setTab(t),
      model: (name, dataset, blocks, training, extra) =>
        S.current.setDraft((d) => ({ ...d, name, dataset, train_subset: null, seq_len: null, blocks, training: { ...S.current.meta!.defaults, ...training }, origin: 'template', edited: false, ...extra }), { undoable: false }),
      lesson: (id) => L.current.openLesson(id),
      spot: (id, text) => L.current.setSpot(id ? { id, text } : null),
      guide: (v) => L.current.setGuideView(v),
      unlockAll: () => L.current.chooseMode('unlocked'),
    }
    L.current.rewards.forEach((r) => L.current.dismissReward(r.key))
    L.current.closeLevelUp()
    Promise.resolve(STEPS[step].run(ctx)).catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, active])

  // Auto-advance, and never let a level-up popup sit in front of the demo.
  useEffect(() => {
    if (!active || !playing) return
    const h = setInterval(() => setElapsed((e) => e + 0.25), 250)
    return () => clearInterval(h)
  }, [active, playing, step])
  useEffect(() => {
    if (active && playing && elapsed >= STEPS[step].seconds && step < STEPS.length - 1) setStep(step + 1)
  }, [elapsed, active, playing, step])
  useEffect(() => {
    if (!active || !learn.levelUp) return
    const h = setTimeout(() => L.current.closeLevelUp(), 2500)
    return () => clearTimeout(h)
  }, [active, learn.levelUp])

  const exit = useCallback(async () => {
    if (!state) return
    setExiting(true)
    token.current++
    try {
      const jobs = await api.jobs()
      await Promise.all(jobs.map((j) => api.stopJob(j.id).catch(() => {})))
      const profiles = await api.profiles()
      const demo = profiles.profiles.find((p) => p.active && p.demo)
      const target = profiles.profiles.some((p) => p.id === state.returnTo) ? state.returnTo : profiles.profiles.find((p) => !p.demo)?.id
      if (target) await api.activateProfile(target)
      if (demo) await api.deleteProfile(demo.id)
      if (cleanup) {
        await wait(800) // let stopped runs finish saving
        const runs = await api.runs()
        await Promise.all(runs.filter((r) => !state.runIds.includes(r.id)).map((r) => api.deleteRun(r.id).catch(() => {})))
      }
    } finally {
      localStorage.removeItem(KEY)
      if (state.draft) localStorage.setItem(DRAFT_KEY, state.draft)
      else localStorage.removeItem(DRAFT_KEY)
      sessionStorage.clear()
      location.hash = ''
      location.reload()
    }
  }, [state, cleanup])

  if (!active) return null
  const s = STEPS[step]
  const last = step === STEPS.length - 1
  return (
    <div className="demo-card">
      <div className="row">
        <span className="pill accent">🎓 Teacher demo</span>
        <span className="tiny muted">{step + 1} / {STEPS.length}</span>
        <span className="spacer" />
        <button className="ghost sm" onClick={exit} disabled={exiting}>{exiting ? 'Leaving…' : 'Exit demo ✕'}</button>
      </div>
      <h3 style={{ margin: '8px 0 4px' }}>{s.title}</h3>
      <div className="small" style={{ lineHeight: 1.55 }}>{s.say}</div>
      <div className="progress" style={{ marginTop: 10, height: 4 }}><div style={{ width: `${Math.min(1, elapsed / s.seconds) * 100}%`, transition: 'width 0.25s linear' }} /></div>
      <div className="row" style={{ marginTop: 10 }}>
        <button className="sm" disabled={step === 0} onClick={() => setStep(step - 1)}>◀ Back</button>
        <button className="sm" onClick={() => setPlaying(!playing)}>{playing ? '⏸ Pause' : '▶ Play'}</button>
        <span className="spacer" />
        {last ? (
          <>
            <label className="row tiny muted"><input type="checkbox" checked={cleanup} onChange={(e) => setCleanup(e.target.checked)} /> delete the demo’s models</label>
            <button className="sm primary" onClick={exit} disabled={exiting}>Finish</button>
          </>
        ) : <button className="sm primary" onClick={() => setStep(step + 1)}>Next ▶</button>}
      </div>
    </div>
  )
}
