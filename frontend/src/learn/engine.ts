import {
  ACHIEVEMENTS, ALL_FEATURES, BASE_BLOCKS, BASE_DATASETS, DATASETS_AT_EXPLORER, GRANTED_SKILLS, LEVELS, QUESTS, SKILLS,
  levelFor, skillPointsEarned, type Feature, type Progress, type Stats,
} from './curriculum'

/** Facts about a finished training run, derived from its config + summary. */
export interface TrainInfo {
  dataset: string
  valAcc: number
  blockTypes: string[]
  epochs: number
  layers: number
  width?: string
  changed: string[] // hyperparameters that differ from the defaults
  optimizer: string
  lr: number
  built: boolean
  reproduced: boolean
  resumed: boolean
}

export type LearnEvent =
  | { type: 'train_done'; info: TrainInfo }
  | { type: 'flag'; name: string }
  | { type: 'lesson'; id: string }
  | { type: 'quiz'; id: string }

export interface Reward {
  kind: 'quest' | 'achievement' | 'discovery' | 'level'
  title: string
  xp: number
  icon?: string
  detail?: string
  level?: number
}

const add = (list: string[], v: string) => (list.includes(v) ? list : [...list, v])
const SKILL_BLOCKS = new Set(SKILLS.flatMap((s) => s.blocks))

function updateStats(s: Stats, e: LearnEvent, discover: (title: string, xp: number) => void): Stats {
  switch (e.type) {
    case 'flag':
      return s.flags[e.name] ? s : { ...s, flags: { ...s.flags, [e.name]: true } }
    case 'lesson':
      if (s.lessons.includes(e.id)) return s
      discover('Lesson read', 10)
      return { ...s, lessons: [...s.lessons, e.id] }
    case 'quiz':
      if (s.quizzes.includes(e.id)) return s
      discover('Quiz answered', 15)
      return { ...s, quizzes: [...s.quizzes, e.id] }
    case 'train_done': {
      const i = e.info
      discover('Training run finished', 5)
      const flags = { ...s.flags }
      for (const c of i.changed) flags[`hp:${c}`] = true
      flags[`opt:${i.optimizer}`] = true
      if (i.lr >= 0.1) flags.lrHigh = true
      if (i.lr <= 1e-5) flags.lrLow = true
      if (i.built) flags.trainedBuilt = true
      if (i.reproduced) flags.reproduced = true
      if (i.resumed) flags.resumed = true
      if (i.blockTypes.some((t) => t.startsWith('custom:'))) flags.customUsed = true
      if (i.blockTypes.some((t) => SKILL_BLOCKS.has(t) && t !== 'dropout' && t !== 'activation')) flags.usedSkillBlock = true
      const newBlocks = i.blockTypes.filter((t) => !s.blocksTrained.includes(t))
      newBlocks.filter((t) => !BASE_BLOCKS.includes(t)).forEach((t) => discover(`First time training with ${t}`, 15))
      if (!s.datasets.includes(i.dataset) && s.datasets.length) discover(`New dataset: ${i.dataset}`, 20)
      return {
        ...s,
        flags,
        runs: s.runs + 1,
        widths: i.width ? add(s.widths, i.width) : s.widths,
        datasets: add(s.datasets, i.dataset),
        best: { ...s.best, [i.dataset]: Math.max(s.best[i.dataset] ?? 0, i.valAcc) },
        blocksTrained: [...s.blocksTrained, ...newBlocks],
        maxLayers: Math.max(s.maxLayers, i.layers),
        maxEpochs: Math.max(s.maxEpochs, i.epochs),
      }
    }
  }
}

/** Apply one event. Returns the new progress and every reward it triggered (in order). */
export function applyEvent(p: Progress, e: LearnEvent): { next: Progress; rewards: Reward[] } {
  const rewards: Reward[] = []
  let xp = p.xp
  const guided = p.mode === 'guided'
  const stats = updateStats(p.stats, e, (title, gain) => {
    if (!guided) return
    xp += gain
    rewards.push({ kind: 'discovery', title, xp: gain })
  })
  let next: Progress = { ...p, stats, xp }
  if (!guided) return { next, rewards: [] }

  // Award until nothing changes: a level-up can make more quests eligible.
  for (let changed = true; changed; ) {
    changed = false
    const level = levelFor(next.xp).level
    for (const q of QUESTS) {
      if (q.level <= level && !next.quests.includes(q.id) && q.done(next.stats, next)) {
        next = { ...next, quests: [...next.quests, q.id], xp: next.xp + q.xp }
        rewards.push({ kind: 'quest', title: q.title, xp: q.xp, icon: '✅' })
        changed = true
      }
    }
    for (const a of ACHIEVEMENTS) {
      if (!next.achievements[a.id] && a.done(next.stats, next)) {
        next = { ...next, achievements: { ...next.achievements, [a.id]: Date.now() }, xp: next.xp + a.xp }
        rewards.push({ kind: 'achievement', title: a.title, xp: a.xp, icon: a.icon, detail: a.desc })
        changed = true
      }
    }
    const newLevel = levelFor(next.xp).level
    for (let l = level + 1; l <= newLevel; l++) {
      const L = LEVELS.find((x) => x.level === l)!
      rewards.push({ kind: 'level', title: L.title, xp: 0, level: l, detail: L.news.join(' ') })
    }
  }
  return { next, rewards }
}

// ---------------------------------------------------------------------------
// Derived views
// ---------------------------------------------------------------------------
export interface Access {
  unlocked: boolean
  level: number
  has: (f: Feature) => boolean
  blocks: Set<string> | null // null = every block (including custom)
  datasets: Set<string> | null
  skills: Set<string>
  skillPoints: number
}

export function access(p: Progress): Access {
  if (p.mode !== 'guided') {
    return {
      unlocked: true, level: 99, has: () => true, blocks: null, datasets: null,
      skills: new Set(SKILLS.map((s) => s.id)), skillPoints: 0,
    }
  }
  const level = levelFor(p.xp).level
  const features = new Set<Feature>(LEVELS.filter((l) => l.level <= level).flatMap((l) => l.unlocks))
  const skills = new Set([...GRANTED_SKILLS(level), ...p.skills])
  const blocks = new Set([...BASE_BLOCKS, ...SKILLS.filter((s) => skills.has(s.id)).flatMap((s) => s.blocks)])
  const datasets = new Set(features.has('datasets') ? DATASETS_AT_EXPLORER : BASE_DATASETS)
  SKILLS.filter((s) => skills.has(s.id)).forEach((s) => s.datasets?.forEach((d) => datasets.add(d)))
  return {
    unlocked: false,
    level,
    has: (f) => features.has(f),
    blocks: features.has('custom.page') ? new Set([...blocks, '__custom__']) : blocks,
    datasets,
    skills,
    skillPoints: skillPointsEarned(level) - p.skills.length,
  }
}

export function blockAllowed(a: Access, type: string) {
  if (!a.blocks) return true
  if (type.startsWith('custom:')) return a.blocks.has('__custom__')
  return a.blocks.has(type)
}

/** Which level (or skill) unlocks a feature/block, for "locked" hints. */
export function featureLevel(f: Feature) {
  return LEVELS.find((l) => l.unlocks.includes(f))?.level ?? 1
}

export function skillForBlock(type: string) {
  return SKILLS.find((s) => s.blocks.includes(type))
}

export function skillForDataset(id: string) {
  return SKILLS.find((s) => s.datasets?.includes(id))
}

export const everything = ALL_FEATURES
