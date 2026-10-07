import { lazy, Suspense } from 'react'
import { Builder } from './pages/Builder'
import { History } from './pages/History'
import { Leaderboard } from './pages/Leaderboard'
import { Train } from './pages/Train'
import { StoreProvider, useStore } from './store'

// The code editor is heavy; only load it when the Custom Blocks tab is opened.
const CustomBlocks = lazy(() => import('./pages/CustomBlocks').then((m) => ({ default: m.CustomBlocks })))

const TABS = [
  { id: 'build', label: 'Build' },
  { id: 'train', label: 'Train' },
  { id: 'leaderboard', label: 'Leaderboard' },
  { id: 'custom', label: 'Custom Blocks' },
  { id: 'history', label: 'History of AI' },
]

function Shell() {
  const { tab, setTab, session, toasts, meta } = useStore()
  const page = { build: <Builder />, train: <Train />, leaderboard: <Leaderboard />, custom: <CustomBlocks />, history: <History /> }[tab] ?? <Builder />
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
          {TABS.map((t) => (
            <button key={t.id} className={`tab ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
              {t.id === 'train' && session?.status === 'running' && <span className="dot" title="Training in progress" />}
            </button>
          ))}
        </nav>
        <span className="spacer" />
        {meta && (
          <span className="small faint mono" title={String(meta.environment.platform)}>
            torch {meta.environment.torch} · {meta.environment.available_devices.join(' / ')}
          </span>
        )}
      </header>
      <main className="main"><Suspense fallback={<div className="page"><span className="spinner" /></div>}>{page}</Suspense></main>
      <div className="toasts">
        {toasts.map((t) => <div key={t.id} className={`toast ${t.kind}`}>{t.msg}</div>)}
      </div>
    </div>
  )
}

export default function App() {
  return (
    <StoreProvider>
      <Shell />
    </StoreProvider>
  )
}
