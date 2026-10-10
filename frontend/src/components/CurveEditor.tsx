import { useEffect, useRef, useState, type PointerEvent } from 'react'

const PRESETS: { name: string; f: (x: number) => number }[] = [
  { name: 'ReLU', f: (x) => Math.max(0, x) },
  { name: 'Tanh', f: Math.tanh },
  { name: 'Sigmoid', f: (x) => 1 / (1 + Math.exp(-x)) },
  { name: 'Abs', f: Math.abs },
  { name: 'Sine', f: (x) => Math.sin(x * 1.5) },
  { name: 'Step', f: (x) => (x > 0 ? 1 : 0) },
  { name: 'Line', f: (x) => x },
]

const W = 300
const H = 190
const PAD = 14

/**
 * Draw a function by hand: drag across the plot (or drag single points) to set the height at
 * each of the fixed x positions. The result is the piecewise-linear curve through the points.
 */
export function CurveEditor({ xs, yLimit, value, onChange, disabled }: {
  xs: number[]
  yLimit: number
  value: number[]
  onChange: (ys: number[]) => void
  disabled?: boolean
}) {
  const [ys, setYs] = useState(value)
  const latest = useRef(value)
  latest.current = ys
  const drawing = useRef(false)
  const svg = useRef<SVGSVGElement>(null)
  // Follow outside changes (reset, undo) when not in the middle of a stroke.
  useEffect(() => {
    if (!drawing.current) setYs(value)
  }, [value])

  const x0 = xs[0]
  const x1 = xs[xs.length - 1]
  const X = (x: number) => PAD + ((x - x0) / (x1 - x0)) * (W - 2 * PAD)
  const Y = (y: number) => H / 2 - (y / yLimit) * (H / 2 - PAD)

  const paint = (e: PointerEvent) => {
    const r = svg.current!.getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * W
    const py = ((e.clientY - r.top) / r.height) * H
    const x = x0 + ((px - PAD) / (W - 2 * PAD)) * (x1 - x0)
    const i = Math.max(0, Math.min(xs.length - 1, Math.round((x - x0) / (xs[1] - xs[0]))))
    const y = Math.max(-yLimit, Math.min(yLimit, ((H / 2 - py) / (H / 2 - PAD)) * yLimit))
    setYs((cur) => cur.map((v, k) => (k === i ? Math.round(y * 100) / 100 : v)))
  }

  const finish = () => {
    if (!drawing.current) return
    drawing.current = false
    onChange(latest.current)
  }

  return (
    <div className="curve-editor">
      <svg
        ref={svg}
        viewBox={`0 0 ${W} ${H}`}
        style={{ touchAction: 'none', cursor: disabled ? 'not-allowed' : 'crosshair' }}
        onPointerDown={(e) => {
          if (disabled) return
          drawing.current = true
          ;(e.target as Element).setPointerCapture?.(e.pointerId)
          paint(e)
        }}
        onPointerMove={(e) => drawing.current && paint(e)}
        onPointerUp={finish}
        onPointerLeave={finish}
      >
        <rect x={0} y={0} width={W} height={H} rx={8} fill="#0a0e16" />
        {[-3, -2, -1, 1, 2, 3].map((v) => (
          <g key={v}>
            <line x1={X(v)} y1={PAD} x2={X(v)} y2={H - PAD} stroke="#141b2b" />
            {Math.abs(v) <= yLimit && <line x1={PAD} y1={Y(v)} x2={W - PAD} y2={Y(v)} stroke="#141b2b" />}
            <text x={X(v)} y={H / 2 + 11} fontSize={8} fill="#3a4560" textAnchor="middle">{v}</text>
          </g>
        ))}
        <line x1={PAD} y1={H / 2} x2={W - PAD} y2={H / 2} stroke="#34405c" />
        <line x1={X(0)} y1={PAD} x2={X(0)} y2={H - PAD} stroke="#34405c" />
        <polyline points={xs.map((x, i) => `${X(x)},${Y(ys[i])}`).join(' ')} fill="none" stroke="#facc15" strokeWidth={2.5} strokeLinejoin="round" />
        {xs.map((x, i) => <circle key={i} cx={X(x)} cy={Y(ys[i])} r={4} fill="#facc15" stroke="#0a0e16" strokeWidth={1.5} />)}
      </svg>
      <div className="row wrap" style={{ gap: 4, marginTop: 6 }}>
        {PRESETS.map((p) => (
          <button key={p.name} type="button" className="sm" disabled={disabled} onClick={() => {
            const next = xs.map((x) => Math.round(Math.max(-yLimit, Math.min(yLimit, p.f(x))) * 100) / 100)
            setYs(next)
            onChange(next)
          }}>{p.name}</button>
        ))}
      </div>
    </div>
  )
}
