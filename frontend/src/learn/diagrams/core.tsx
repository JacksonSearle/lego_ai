import { Component, useEffect, useRef, useState, type ReactNode } from 'react'

/** Seconds since mount, updated every animation frame while the diagram is on screen. */
export function useTime(): number {
  const [t, setT] = useState(0)
  const ref = useRef<number>(0)
  useEffect(() => {
    let raf = 0
    let last = performance.now()
    const tick = (now: number) => {
      // Advance by real elapsed time, capped so a background tab doesn't jump ahead. The first
      // frame's timestamp can predate `last`, so never step backwards.
      ref.current += Math.max(0, Math.min(0.1, (now - last) / 1000))
      last = now
      setT(ref.current)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])
  return t
}

/** Deterministic pseudo-random number in [0, 1) for index i. */
export const rnd = (i: number) => {
  const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453
  return x - Math.floor(x)
}

export const clamp = (x: number, a = 0, b = 1) => Math.max(a, Math.min(b, x))
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t
export const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2)

export const C = {
  bg: '#0b0f17',
  text: '#e6eaf2',
  muted: '#8f9ab0',
  faint: '#3a4560',
  line: '#34405c',
  panel: '#182032',
  accent: '#facc15',
  blue: '#60a5fa',
  green: '#4ade80',
  red: '#f87171',
  orange: '#fb923c',
  purple: '#a78bfa',
  pink: '#f472b6',
  teal: '#2dd4bf',
}

/** Color scale for a value in [0, 1] (dark → bright). */
export const heat = (v: number, hue = 210) => `hsl(${hue} 80% ${10 + clamp(v) * 55}%)`

export function Svg({ children, h = 220, w = 400 }: { children: ReactNode; h?: number; w?: number }) {
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="diagram-svg" role="img">
      {children}
    </svg>
  )
}

export function Label({ x, y, children, size = 11, color = C.muted, anchor = 'middle', weight = 500 }: {
  x: number; y: number; children: ReactNode; size?: number; color?: string; anchor?: 'start' | 'middle' | 'end'; weight?: number
}) {
  return (
    <text x={x} y={y} fontSize={size} fill={color} textAnchor={anchor} fontWeight={weight} fontFamily="Inter, system-ui, sans-serif">
      {children}
    </text>
  )
}

export function Choice<T extends string>({ options, value, onChange }: { options: { id: T; label: string }[]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="seg diagram-choice">
      {options.map((o) => (
        <button key={o.id} className={o.id === value ? 'on' : ''} onClick={() => onChange(o.id)}>{o.label}</button>
      ))}
    </div>
  )
}

export function Frame({ children, caption, controls }: { children: ReactNode; caption?: ReactNode; controls?: ReactNode }) {
  return (
    <figure className="diagram">
      {controls && <div className="diagram-controls">{controls}</div>}
      {children}
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  )
}

/** Phase helper: returns progress in [0,1] within a repeating cycle of `period` seconds, plus the cycle index. */
export const cycle = (t: number, period: number) => ({ p: (t % period) / period, n: Math.floor(t / period) })

/** Keeps a broken diagram from taking down the page around it. */
export class DiagramBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  render() {
    return this.state.failed ? <div className="diagram muted small" style={{ padding: 16 }}>This animation couldn’t be shown.</div> : this.props.children
  }
}
