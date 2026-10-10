import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { ACHIEVEMENTS, SKILLS, newProgress, type Progress } from './curriculum'
import { access, applyEvent, type Access, type LearnEvent, type Reward } from './engine'

interface Learn {
  ready: boolean
  progress: Progress
  access: Access
  track: (e: LearnEvent) => void
  flag: (name: string) => void
  chooseMode: (mode: 'guided' | 'unlocked') => void
  unlockSkill: (id: string) => string | null
  setAvatar: (icon: string) => void
  reset: () => void
  /** Lesson shown in the modal reader (null = closed). */
  lesson: string | null
  openLesson: (id: string | null) => void
  /** data-tour id of the element currently highlighted by the tutor. */
  spot: { id: string; text?: string } | null
  setSpot: (s: { id: string; text?: string } | null) => void
  rewards: (Reward & { key: number })[]
  dismissReward: (key: number) => void
  levelUp: Reward | null
  closeLevelUp: () => void
  update: (fn: (p: Progress) => Progress) => void
  overviewOpen: boolean
  setOverviewOpen: (b: boolean) => void
  /** Which part of the Guidebook page is showing (shared so other pages can deep-link to it). */
  guideView: GuideView
  setGuideView: (v: GuideView) => void
}

export type GuideView = 'lessons' | 'skills' | 'achievements'

const Ctx = createContext<Learn>(null as unknown as Learn)
export const useLearn = () => useContext(Ctx)

let rewardKey = 0

export function LearnProvider({ children }: { children: ReactNode }) {
  const [progress, setProgress] = useState<Progress>(newProgress)
  const [ready, setReady] = useState(false)
  const [lesson, openLesson] = useState<string | null>(null)
  const [spot, setSpot] = useState<{ id: string; text?: string } | null>(null)
  const [rewards, setRewards] = useState<(Reward & { key: number })[]>([])
  const [levelUp, setLevelUp] = useState<Reward | null>(null)
  const [overviewOpen, setOverviewOpen] = useState(false)
  const [guideView, setGuideViewState] = useState<GuideView>(() => (sessionStorage.getItem('lego-ai/guide-view') as GuideView) || 'lessons')
  const setGuideView = useCallback((v: GuideView) => {
    sessionStorage.setItem('lego-ai/guide-view', v)
    setGuideViewState(v)
  }, [])
  const saveTimer = useRef<ReturnType<typeof setTimeout>>(undefined)

  useEffect(() => {
    fetch('/api/progress')
      .then((r) => r.json())
      .then((d) => {
        if (d.progress) setProgress({ ...newProgress(), ...d.progress, stats: { ...newProgress().stats, ...d.progress.stats } })
        setReady(true)
      })
      .catch(() => setReady(true))
  }, [])

  // Persist (debounced) whenever progress changes.
  useEffect(() => {
    if (!ready) return
    clearTimeout(saveTimer.current)
    saveTimer.current = setTimeout(() => {
      fetch('/api/progress', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(progress) })
    }, 400)
  }, [progress, ready])

  const pushRewards = useCallback((rs: Reward[]) => {
    const visible = rs.filter((r) => r.kind !== 'level')
    if (visible.length) {
      const withKeys = visible.map((r) => ({ ...r, key: ++rewardKey }))
      setRewards((cur) => [...cur, ...withKeys].slice(-8))
      // Achievements stay until you close them; smaller notices fade by themselves.
      withKeys
        .filter((r) => r.kind !== 'achievement')
        .forEach((r) => setTimeout(() => setRewards((cur) => cur.filter((x) => x.key !== r.key)), r.kind === 'quest' ? 9000 : 4500))
    }
    const lvl = rs.filter((r) => r.kind === 'level').pop()
    if (lvl) setLevelUp(lvl)
  }, [])

  // Events are applied against the latest state; rewards are collected outside the updater
  // so StrictMode's double-invocation can't show them twice.
  const latest = useRef(progress)
  latest.current = progress
  const track = useCallback((e: LearnEvent) => {
    const { next, rewards: rs } = applyEvent(latest.current, e)
    latest.current = next
    setProgress(next)
    pushRewards(rs)
  }, [pushRewards])

  const flag = useCallback((name: string) => {
    if (!latest.current.stats.flags[name]) track({ type: 'flag', name })
  }, [track])

  const chooseMode = useCallback((mode: 'guided' | 'unlocked') => {
    const base = { ...latest.current, mode }
    if (mode === 'unlocked') {
      base.achievements = Object.fromEntries(ACHIEVEMENTS.map((a) => [a.id, Date.now()]))
      base.tourDone = false
    }
    latest.current = base
    setProgress(base)
  }, [])

  const unlockSkill = useCallback((id: string): string | null => {
    const p = latest.current
    const a = access(p)
    const node = SKILLS.find((s) => s.id === id)
    if (!node) return 'Unknown skill'
    if (a.skills.has(id)) return 'Already unlocked'
    if (!a.has('skilltree')) return 'The skill tree opens at level 6'
    const missing = node.requires.filter((r) => !a.skills.has(r))
    if (missing.length) return `Unlock ${missing.map((m) => SKILLS.find((s) => s.id === m)?.title).join(' and ')} first`
    if (a.skillPoints <= 0) return 'No skill points left: level up to earn more'
    const { next, rewards: rs } = applyEvent({ ...p, skills: [...p.skills, id] }, { type: 'flag', name: `skill:${id}` })
    latest.current = next
    setProgress(next)
    pushRewards(rs)
    return null
  }, [pushRewards])

  const update = useCallback((fn: (p: Progress) => Progress) => {
    const next = fn(latest.current)
    latest.current = next
    setProgress(next)
  }, [])

  const setAvatar = useCallback((icon: string) => update((p) => ({ ...p, avatar: icon })), [update])

  const reset = useCallback(() => {
    fetch('/api/progress', { method: 'DELETE' })
    const p = newProgress()
    latest.current = p
    setProgress(p)
    setSpot(null)
  }, [])

  const acc = useMemo(() => access(progress), [progress])

  const value: Learn = {
    ready, progress, access: acc, track, flag, chooseMode, unlockSkill, setAvatar, reset, lesson, openLesson, spot, setSpot,
    rewards, dismissReward: (k) => setRewards((r) => r.filter((x) => x.key !== k)), levelUp, closeLevelUp: () => setLevelUp(null),
    update, overviewOpen, setOverviewOpen, guideView, setGuideView,
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
