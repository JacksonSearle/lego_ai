import { lazy, Suspense, useEffect } from 'react'
import type { Feature } from './learn/curriculum'
import { featureLevel } from './learn/engine'
import { Guidebook, LessonModal } from './learn/Guidebook'
import { LearnProvider, useLearn } from './learn/LearnContext'
import { DemoPlayer } from './learn/Demo'
import { LevelUpModal, Onboarding, OverviewTour, ProfileBar, Rewards, Spotlight } from './learn/Overlay'
import { Builder } from './pages/Builder'
import { Datasets } from './pages/Datasets'
import { Explore } from './pages/Explore'
import { Lab } from './pages/Lab'
import { Leaderboard } from './pages/Leaderboard'
import { Scaling } from './pages/Scaling'
import { Train } from './pages/Train'
import { StoreProvider, useStore } from './store'

// The code editor is heavy; only load it when the Custom Blocks tab is opened.
const CustomBlocks = lazy(() => import('./pages/CustomBlocks').then((m) => ({ default: m.CustomBlocks })))

const TABS: { id: string; label: string; feature?: Feature }[] = [
  { id: 'lab', label: '🧪 Lab', feature: 'lab' },
  { id: 'build', label: 'Build', feature: 'builder' },
  { id: 'train', label: 'Train', feature: 'train.page' },
  { id: 'leaderboard', label: 'Leaderboard', feature: 'leaderboard.page' },
  { id: 'explore', label: 'Explore' },
  { id: 'scaling', label: 'Scaling', feature: 'scaling.page' },
  { id: 'datasets', label: 'Datasets', feature: 'datasets.custom' },
  { id: 'custom', label: 'Custom Blocks', feature: 'custom.page' },
  { id: 'guide', label: '📖 Guidebook' },
]

function Shell() {
  const { tab, setTab, session, toasts, draft, setDraft, inspect, toast, offline } = useStore()
  const { access, progress, flag, update, setSpot, ready } = useLearn()

  // The Lab is the level 1 to 4 playground. After that it leaves the tab bar (it stays reachable from
  // the profile menu, and stays put while you are still on it so it never vanishes under you).
  const visible = TABS.filter((t) => t.id !== 'lab' || tab === 'lab' || (!access.unlocked && access.level < 5))
  const allowed = (id: string) => {
    const t = TABS.find((x) => x.id === id)
    return !!t && visible.includes(t) && (!t.feature || access.has(t.feature))
  }

  // Keep the user on a page they can use.
  useEffect(() => {
    if (!ready || !progress.mode) return
    if (!allowed(tab)) setTab(access.has('builder') ? 'build' : 'lab')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, ready, progress.mode, access.level, access.unlocked])

  useEffect(() => {
    if (ready && progress.mode) flag(`visit:${tab}`)
  }, [tab, ready, progress.mode, flag])

  // The broken-model challenge is complete once the challenge model becomes valid.
  useEffect(() => {
    // The shape check is debounced: only trust a result that describes the current blocks.
    const current = !!inspect && inspect.blocks.map((b) => b.id).join() === draft.blocks.map((b) => b.id).join()
    if (progress.challenge === 'active' && draft.origin === 'challenge' && current && inspect?.ok && draft.blocks.length > 0) {
      update((p) => ({ ...p, challenge: 'done' }))
      setDraft((d) => ({ ...d, name: 'Repaired model' }), { undoable: false })
      flag('fixedShapes')
      setSpot(null)
      toast('🧩 Fixed! Every block fits and the output is 10 numbers, one per digit. Now train it!', 'ok')
    }
  }, [progress.challenge, draft.origin, draft.blocks, inspect, update, flag, setSpot, toast, setDraft])

  const page =
    {
      lab: <Lab />,
      build: <Builder />,
      train: <Train />,
      leaderboard: <Leaderboard />,
      explore: <Explore />,
      scaling: <Scaling />,
      datasets: <Datasets />,
      custom: <CustomBlocks />,
      guide: <Guidebook />,
    }[tab] ?? <Lab />

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-logo">
            {['#ef4444', '#facc15', '#3b82f6'].map((c) => <span key={c} style={{ background: c }} />)}
          </span>
          Lego AI
        </div>
        <nav className="tabs">
          {visible.map((t) => {
            const locked = !!t.feature && !access.has(t.feature)
            return (
              <button
                key={t.id}
                className={`tab ${tab === t.id ? 'active' : ''} ${locked ? 'locked' : ''}`}
                onClick={() => !locked && setTab(t.id)}
                title={locked ? `Unlocks at level ${featureLevel(t.feature!)}` : undefined}
              >
                {locked && '🔒 '}{t.label}
                {t.id === 'train' && session?.status === 'running' && <span className="dot" title="Training in progress" />}
              </button>
            )
          })}
        </nav>
        <span className="spacer" />
        <ProfileBar />
      </header>
      {offline && (
        <div className="offline-banner">
          ⚠️ Can’t reach the Lego AI server, so shape checks and training are paused. Start it with <code>./run.sh</code> in the project folder: this page reconnects automatically.
        </div>
      )}
      <main className="main">
        <Suspense fallback={<div className="page"><span className="spinner" /></div>}>{ready && progress.mode ? page : <div className="page"><span className="spinner" /></div>}</Suspense>
      </main>
      <div className="toasts">
        {toasts.map((t) => <div key={t.id} className={`toast ${t.kind}`}>{t.msg}</div>)}
      </div>
      <DemoPlayer />
      <Rewards />
      <Spotlight />
      <OverviewTour />
      <LessonModal />
      <LevelUpModal />
      <Onboarding />
    </div>
  )
}

export default function App() {
  return (
    <LearnProvider>
      <StoreProvider>
        <Shell />
      </StoreProvider>
    </LearnProvider>
  )
}
