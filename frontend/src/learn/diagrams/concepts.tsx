import { useEffect, useMemo, useState, type ReactElement } from 'react'
import { TensorImage } from '../../components/TensorImage'
import { C, Choice, Frame, Label, Svg, clamp, cycle, ease, heat, lerp, rnd, useTime } from './core'

// ---------------------------------------------------------------------------
// What is a neural network: signals flowing through layers of neurons
// ---------------------------------------------------------------------------
export function NeuralNet() {
  const t = useTime()
  const layers = [4, 6, 3]
  const xs = [60, 200, 340]
  const ys = (n: number) => Array.from({ length: n }, (_, i) => 110 + (i - (n - 1) / 2) * 32)
  const { p } = cycle(t, 2.6)
  const edges: ReactElement[] = []
  const pulses: ReactElement[] = []
  for (let l = 0; l < 2; l++) {
    const a = ys(layers[l])
    const b = ys(layers[l + 1])
    a.forEach((ya, i) =>
      b.forEach((yb, j) => {
        const w = rnd(l * 100 + i * 10 + j)
        edges.push(<line key={`e${l}${i}${j}`} x1={xs[l]} y1={ya} x2={xs[l + 1]} y2={yb} stroke={w > 0.5 ? C.blue : C.orange} strokeOpacity={0.15 + w * 0.35} strokeWidth={0.5 + w * 2} />)
        const local = clamp((p - l * 0.45) / 0.45)
        if (local > 0 && local < 1) pulses.push(<circle key={`p${l}${i}${j}`} cx={lerp(xs[l], xs[l + 1], local)} cy={lerp(ya, yb, local)} r={2.2} fill={C.accent} opacity={0.8} />)
      }),
    )
  }
  const lit = (l: number) => clamp((p - l * 0.45) * 4)
  return (
    <Frame caption="Each circle is a neuron. Each line is a weight — a number the network learns. Signals flow left to right: pixels in, an answer out.">
      <Svg>
        {edges}
        {pulses}
        {layers.map((n, l) =>
          ys(n).map((y, i) => (
            <circle key={`n${l}${i}`} cx={xs[l]} cy={y} r={11} fill={l === 2 && i === 1 ? heat(lit(l) * 0.9, 45) : heat(0.25 + lit(l) * 0.5)} stroke={C.line} />
          )),
        )}
        <Label x={60} y={208}>pixels in</Label>
        <Label x={200} y={208}>hidden layer</Label>
        <Label x={340} y={208}>answer out</Label>
        {['0', '1', '2'].map((d, i) => <Label key={d} x={362} y={ys(3)[i] + 4} anchor="start" color={i === 1 ? C.accent : C.muted}>{i === 1 ? `“7” ${Math.round(lit(2) * 92)}%` : ''}</Label>)}
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// MNIST: real digits, and what a digit looks like to a computer
// ---------------------------------------------------------------------------
interface Sample { image: number[]; shape: number[]; label: number }

export function useSamples(dataset: string, n: number) {
  const [s, setS] = useState<Sample[]>([])
  useEffect(() => {
    fetch(`/api/datasets/${dataset}/samples?n=${n}`).then((r) => r.json()).then((d) => setS(d.examples ?? []))
  }, [dataset, n])
  return s
}

export function MnistPeek() {
  const samples = useSamples('mnist', 16)
  const t = useTime()
  const { n, p } = cycle(t, 4)
  if (!samples.length) return <Frame><div className="muted small" style={{ padding: 30 }}>Loading digits…</div></Frame>
  const sel = samples[n % samples.length]
  const row = Math.floor(p * 28)
  const cell = 6.5
  return (
    <Frame caption={<>Left: real MNIST digits with their labels. Right: what the computer actually sees — a 28×28 grid of brightness numbers from 0 (black) to 255 (white). That is 28 × 28 = <b>784 numbers</b> per image.</>}>
      <div className="row" style={{ gap: 20, alignItems: 'flex-start', flexWrap: 'wrap', justifyContent: 'center' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 52px)', gap: 8 }}>
          {samples.map((s, i) => (
            <div key={i} style={{ textAlign: 'center', outline: s === sel ? `2px solid ${C.accent}` : 'none', borderRadius: 6, padding: 2 }}>
              <TensorImage data={s.image} shape={s.shape} size={46} />
              <div className="tiny mono muted">{s.label}</div>
            </div>
          ))}
        </div>
        <div>
          <svg width={28 * cell} height={28 * cell} style={{ display: 'block', borderRadius: 4 }}>
            {sel.image.map((v, i) => (
              <rect key={i} x={(i % 28) * cell} y={Math.floor(i / 28) * cell} width={cell - 0.4} height={cell - 0.4} fill={`rgb(${v},${v},${v})`} />
            ))}
            <rect x={0} y={row * cell} width={28 * cell} height={cell} fill="none" stroke={C.accent} strokeWidth={1.5} />
          </svg>
          <div className="mono tiny" style={{ width: 28 * cell, marginTop: 6, color: C.muted, wordBreak: 'break-all', lineHeight: 1.5 }}>
            row {row + 1}: {sel.image.slice(row * 28, row * 28 + 28).map((v, j) => <span key={j} style={{ color: v > 0 ? C.accent : C.faint }}>{v} </span>)}
          </div>
        </div>
      </div>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// The training loop: guess → check → blame → adjust
// ---------------------------------------------------------------------------
export function TrainingLoop() {
  const t = useTime()
  const { n, p } = cycle(t, 3.2)
  const loss = 2.3 * Math.exp(-n * 0.32) + 0.08
  const acc = 1 - 0.9 * Math.exp(-n * 0.38)
  const stations = [
    { a: -90, title: '1. Guess', sub: 'forward pass', c: C.blue },
    { a: 0, title: '2. Check', sub: 'loss = how wrong', c: C.red },
    { a: 90, title: '3. Blame', sub: 'backpropagation', c: C.purple },
    { a: 180, title: '4. Adjust', sub: 'optimizer step', c: C.green },
  ]
  const ang = (-90 + p * 360) * (Math.PI / 180)
  const R = 72
  return (
    <Frame caption="Training repeats this loop thousands of times, one small batch of examples at a time. Each lap, the weights are nudged so the next guess is a little less wrong.">
      <Svg h={230}>
        <circle cx={200} cy={115} r={R} fill="none" stroke={C.line} strokeWidth={2} strokeDasharray="4 5" />
        {stations.map((s, i) => {
          const a = (s.a * Math.PI) / 180
          const x = 200 + Math.cos(a) * R
          const y = 115 + Math.sin(a) * R
          const active = Math.floor(p * 4) === i
          const tx = 200 + Math.cos(a) * (R + 62)
          return (
            <g key={i}>
              <circle cx={x} cy={y} r={active ? 10 : 7} fill={s.c} opacity={active ? 1 : 0.55} />
              <Label x={tx} y={y - 3} color={active ? C.text : C.muted} weight={700} size={12}>{s.title}</Label>
              <Label x={tx} y={y + 11} size={10}>{s.sub}</Label>
            </g>
          )
        })}
        <circle cx={200 + Math.cos(ang) * R} cy={115 + Math.sin(ang) * R} r={5} fill={C.accent} />
        <Label x={200} y={104} size={11}>batch #{n + 1}</Label>
        <Label x={200} y={121} size={14} color={C.red} weight={700}>loss {loss.toFixed(2)}</Label>
        <Label x={200} y={137} size={12} color={C.green} weight={600}>acc {Math.round(acc * 100)}%</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Depth: more hidden layers
// ---------------------------------------------------------------------------
export function Depth() {
  const [layers, setLayers] = useState<'1' | '2' | '3' | '4'>('2')
  const t = useTime()
  const L = Number(layers)
  const cols = [5, ...Array(L).fill(6), 3]
  const xs = cols.map((_, i) => 40 + (i * 320) / (cols.length - 1))
  const ys = (n: number) => Array.from({ length: n }, (_, i) => 105 + (i - (n - 1) / 2) * 26)
  const { p } = cycle(t, 2.4)
  const front = p * (cols.length - 1)
  return (
    <Frame
      controls={<Choice options={[1, 2, 3, 4].map((v) => ({ id: String(v) as '1', label: `${v} hidden` }))} value={layers} onChange={setLayers} />}
      caption="Each extra hidden layer can build on the patterns found by the one below it: strokes → loops and corners → whole digits. More layers = a deeper network."
    >
      <Svg h={210}>
        {cols.slice(0, -1).map((n, l) =>
          ys(n).flatMap((ya, i) => ys(cols[l + 1]).map((yb, j) => (
            <line key={`${l}-${i}-${j}`} x1={xs[l]} y1={ya} x2={xs[l + 1]} y2={yb} stroke={front > l && front < l + 1 ? C.accent : C.line} strokeOpacity={front > l && front < l + 1 ? 0.35 : 0.3} strokeWidth={0.7} />
          ))),
        )}
        {cols.map((n, l) => ys(n).map((y, i) => <circle key={`${l}-${i}`} cx={xs[l]} cy={y} r={8} fill={front >= l ? heat(0.6, l === 0 ? 210 : l === cols.length - 1 ? 45 : 140) : C.panel} stroke={C.line} />))}
        {xs.map((x, i) => <Label key={i} x={x} y={200} size={10}>{i === 0 ? 'input' : i === xs.length - 1 ? 'output' : `layer ${i}`}</Label>)}
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Tensor shapes: blocks only fit when the shapes match
// ---------------------------------------------------------------------------
const SHAPE_STEPS = [
  { title: 'An image is a 28×28 grid', text: 'The input tensor has shape 1×28×28: one colour channel, 28 rows, 28 columns.' },
  { title: 'Linear expects a flat list', text: 'A Linear layer multiplies a list of numbers by a weight matrix. It wants shape 784, not a 28×28 grid — the bricks don’t fit.' },
  { title: 'Flatten unrolls the grid', text: 'Flatten lays the 28 rows end-to-end: 1×28×28 → 784. Same numbers, new shape.' },
  { title: 'Now it fits', text: 'Linear(784 → 128) sits on the 784 strip and outputs 128 numbers. Every block’s output shape must match the next block’s input shape.' },
]

export function Shapes() {
  const t = useTime()
  const [manual, setManual] = useState<number | null>(null)
  const auto = Math.floor(t / 3.2) % SHAPE_STEPS.length
  const step = manual ?? auto
  const local = manual === null ? (t % 3.2) / 3.2 : 1
  const flatten = step >= 2 ? ease(step === 2 ? clamp(local * 1.6) : 1) : 0
  const shake = step === 1 ? Math.sin(t * 40) * 3 * (1 - clamp(local * 2 - 1)) : 0
  const cells: ReactElement[] = []
  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 6; c++) {
      const i = r * 6 + c
      const gx = 140 + c * 20
      const gy = 120 + r * 14
      const sx = 20 + i * 10
      const sy = 190
      cells.push(<rect key={i} x={lerp(gx, sx, flatten)} y={lerp(gy, sy, flatten)} width={lerp(18, 9, flatten)} height={lerp(12, 12, flatten)} rx={2} fill={heat(rnd(i) * 0.8 + 0.1, 210)} />)
    }
  }
  const fits = step === 3
  const brickY = step === 0 ? 20 : step === 1 ? 58 : step === 2 ? 30 : 140
  return (
    <Frame
      controls={<div className="row">{SHAPE_STEPS.map((_s, i) => <button key={i} className={`sm ${step === i ? 'primary' : ''}`} onClick={() => setManual(i)}>{i + 1}</button>)}{manual !== null && <button className="sm ghost" onClick={() => setManual(null)}>▶ auto</button>}</div>}
      caption={<><b>{SHAPE_STEPS[step].title}.</b> {SHAPE_STEPS[step].text}</>}
    >
      <Svg h={215}>
        {cells}
        <Label x={flatten > 0.5 ? 200 : 200} y={flatten > 0.5 ? 212 : 113} color={C.text} weight={700} size={12}>
          {flatten > 0.5 ? 'shape: 784' : 'shape: 1 × 28 × 28'}
        </Label>
        <g transform={`translate(${shake}, ${step === 3 ? -38 : 0})`}>
          <rect x={110} y={brickY} width={180} height={34} rx={6} fill={step >= 1 ? '#3b82f6' : C.panel} stroke={fits ? C.green : step === 1 ? C.red : C.line} strokeWidth={2} opacity={step === 0 ? 0.25 : 1} />
          {[0, 1, 2, 3].map((i) => <rect key={i} x={124 + i * 42} y={brickY - 5} width={22} height={6} rx={2} fill="#3b82f6" opacity={step === 0 ? 0.25 : 1} />)}
          <Label x={200} y={brickY + 15} color="#fff" weight={700} size={12}>Linear  784 → 128</Label>
          <Label x={200} y={brickY + 28} color="#dbeafe" size={10}>expects input: 784</Label>
        </g>
        {step === 1 && <Label x={330} y={92} color={C.red} weight={700} size={22}>✗</Label>}
        {step === 3 && <Label x={318} y={130} color={C.green} weight={700} size={22}>✓</Label>}
        {step === 3 && <Label x={200} y={88} color={C.accent} weight={700}>output: 128</Label>}
      </Svg>
    </Frame>
  )
}

/** Matrix multiplication: the inner dimensions must agree. */
export function MatMul() {
  const [mode, setMode] = useState<'ok' | 'bad'>('ok')
  const t = useTime()
  const glow = 0.5 + 0.5 * Math.sin(t * 4)
  const k2 = mode === 'ok' ? 784 : 100
  const ok = mode === 'ok'
  const box = (x: number, w: number, h: number, fill: string) => <rect x={x} y={110 - h / 2} width={w} height={h} rx={4} fill={fill} opacity={0.85} />
  return (
    <Frame
      controls={<Choice options={[{ id: 'ok', label: 'Matching' }, { id: 'bad', label: 'Mismatched' }]} value={mode} onChange={setMode} />}
      caption="Inside a Linear layer, a batch of inputs (batch × 784) is multiplied by a weight matrix (784 × 128). The two inner numbers must be equal — each input number needs exactly one row of weights. If they differ, PyTorch raises a shape error."
    >
      <Svg h={200}>
        {box(30, 50, 110, '#3b82f6')}
        <Label x={55} y={175}>batch</Label>
        <Label x={55} y={48} color={ok ? C.green : C.red} weight={700} size={13}>784</Label>
        <circle cx={55} cy={44} r={17} fill="none" stroke={ok ? C.green : C.red} strokeOpacity={glow} strokeDasharray="4 3" />
        <Label x={102} y={116} size={20} color={C.text}>×</Label>
        {box(150, 100, ok ? 110 : 40, '#8b5cf6')}
        <Label x={132} y={115} color={ok ? C.green : C.red} weight={700} size={13}>{k2}</Label>
        <circle cx={132} cy={111} r={17} fill="none" stroke={ok ? C.green : C.red} strokeOpacity={glow} strokeDasharray="4 3" />
        <Label x={200} y={ok ? 48 : 82}>128</Label>
        <Label x={272} y={116} size={20} color={C.text}>=</Label>
        {ok ? box(295, 80, 110, '#22c55e') : <Label x={335} y={118} size={28} color={C.red} weight={800}>✗</Label>}
        {ok && <><Label x={335} y={48}>128</Label><Label x={335} y={175}>batch</Label></>}
        <Label x={200} y={196} color={ok ? C.green : C.red} weight={600}>{ok ? '(batch × 784) · (784 × 128) → (batch × 128)' : '784 ≠ 100 — “mat1 and mat2 shapes cannot be multiplied”'}</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Overfitting: train vs validation loss
// ---------------------------------------------------------------------------
export function Overfitting() {
  const t = useTime()
  const reveal = clamp((t % 7) / 5.5)
  const N = 60
  const X = (e: number) => 40 + (e / 30) * 330
  const Y = (l: number) => 180 - (l / 2.2) * 150
  const train = (e: number) => 2 * Math.exp(-e / 5) + 0.05
  const val = (e: number) => 2 * Math.exp(-e / 5) + 0.17 + 0.0022 * Math.max(0, e - 9) ** 2
  const pts = (f: (e: number) => number) => Array.from({ length: Math.floor(N * reveal) + 1 }, (_, i) => `${X((i / N) * 30)},${Y(f((i / N) * 30))}`).join(' ')
  return (
    <Frame caption="Training loss (blue) keeps falling as the model memorises the training examples. Validation loss (orange) — measured on examples it never trains on — falls, then rises: the model is overfitting. The best model is at the bottom of the orange curve.">
      <Svg h={210}>
        <line x1={40} y1={180} x2={375} y2={180} stroke={C.line} />
        <line x1={40} y1={20} x2={40} y2={180} stroke={C.line} />
        {reveal > 0.5 && <rect x={X(14)} y={20} width={X(30) - X(14)} height={160} fill={C.red} opacity={0.07} />}
        {reveal > 0.5 && <Label x={X(22)} y={34} color={C.red} size={11}>overfitting zone</Label>}
        <polyline points={pts(train)} fill="none" stroke={C.blue} strokeWidth={2.2} />
        <polyline points={pts(val)} fill="none" stroke={C.orange} strokeWidth={2.2} />
        {reveal > 0.4 && <><circle cx={X(10.5)} cy={Y(val(10.5))} r={5} fill={C.green} /><Label x={X(10.5)} y={Y(val(10.5)) + 18} color={C.green} weight={600}>sweet spot</Label></>}
        <Label x={205} y={200}>epochs →</Label>
        <Label x={18} y={100} anchor="middle">loss</Label>
        <Label x={360} y={Y(train(30)) - 6} color={C.blue} anchor="end">train</Label>
        <Label x={360} y={Y(val(28)) - 8} color={C.orange} anchor="end">validation</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Epochs: passes over the data
// ---------------------------------------------------------------------------
export function Epochs() {
  const t = useTime()
  const { n, p } = cycle(t, 2.6)
  const epoch = (n % 6) + 1
  const acc = 1 - 0.55 * Math.exp(-(epoch - 1 + p) * 0.7)
  return (
    <Frame caption="One epoch = the model has seen every training example once. More epochs give it more practice — with diminishing returns, and eventually overfitting.">
      <Svg h={170}>
        {Array.from({ length: 12 }, (_, i) => {
          const seen = i / 12 < p
          return (
            <g key={i}>
              <rect x={22 + i * 30} y={30} width={24} height={34} rx={4} fill={seen ? '#1e3a66' : C.panel} stroke={Math.floor(p * 12) === i ? C.accent : C.line} strokeWidth={1.5} />
              <Label x={34 + i * 30} y={52} color={seen ? C.text : C.faint} size={13} weight={700}>{Math.floor(rnd(i) * 10)}</Label>
            </g>
          )
        })}
        <Label x={200} y={88} color={C.text} weight={700} size={13}>Epoch {epoch} · example {Math.floor(p * 12) + 1} of 12</Label>
        <rect x={60} y={110} width={280} height={14} rx={7} fill={C.panel} />
        <rect x={60} y={110} width={280 * acc} height={14} rx={7} fill={C.green} />
        <Label x={200} y={148} color={C.green} weight={600}>validation accuracy {Math.round(acc * 100)}%</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Learning rate: a ball rolling into a valley
// ---------------------------------------------------------------------------
const LRS = { tiny: 0.04, good: 0.3, big: 0.92, huge: 1.08 }
type LrKey = keyof typeof LRS

export function LearningRate() {
  const [k, setK] = useState<LrKey>('good')
  const t = useTime()
  const step = Math.floor((t * 2.2) % 18)
  const lr = LRS[k]
  const xs = [-2.6]
  for (let i = 0; i < step; i++) xs.push(xs[i] - lr * 2 * xs[i])
  const X = (x: number) => 200 + clamp(x, -3.3, 3.3) * 52
  const Y = (x: number) => 185 - Math.min(x * x, 11) * 14.5
  const curve = Array.from({ length: 61 }, (_, i) => { const x = -3.2 + (i / 60) * 6.4; return `${X(x)},${Y(x)}` }).join(' ')
  const cur = xs[xs.length - 1]
  const notes: Record<LrKey, string> = {
    tiny: 'Tiny steps: always going the right way, but it would take forever to reach the bottom.',
    good: 'Good: big confident steps that settle at the bottom quickly.',
    big: 'Too big: it overshoots and bounces from side to side, only slowly settling.',
    huge: 'Way too big: every step overshoots further — the loss explodes (you may see “NaN”).',
  }
  return (
    <Frame
      controls={<Choice options={[{ id: 'tiny', label: 'Too small' }, { id: 'good', label: 'Just right' }, { id: 'big', label: 'Too big' }, { id: 'huge', label: 'Way too big' }]} value={k} onChange={setK} />}
      caption={<>The valley is the loss: lower is better. Each step moves the weights downhill by <i>learning rate × slope</i>. {notes[k]}</>}
    >
      <Svg h={205}>
        <polyline points={curve} fill="none" stroke={C.line} strokeWidth={2.5} />
        {xs.slice(1).map((x, i) => <line key={i} x1={X(xs[i])} y1={Y(xs[i]) - 7} x2={X(x)} y2={Y(x) - 7} stroke={C.accent} strokeOpacity={0.5} strokeDasharray="3 3" />)}
        {xs.map((x, i) => <circle key={i} cx={X(x)} cy={Y(x) - 7} r={2.5} fill={C.accent} opacity={0.4} />)}
        <circle cx={X(cur)} cy={Y(cur) - 8} r={8} fill={Math.abs(cur) > 3 ? C.red : C.accent} />
        <Label x={200} y={200} size={10}>weight value →</Label>
        <Label x={360} y={30} anchor="end" color={C.text} weight={600}>loss = {(Math.min(cur * cur, 99)).toFixed(2)}</Label>
        <Label x={360} y={46} anchor="end">step {step} · lr {lr}</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Batch size: noisy vs smooth gradient steps
// ---------------------------------------------------------------------------
function path(noise: number, seed: number) {
  const pts: [number, number][] = [[50, 175]]
  for (let i = 1; i < 26; i++) {
    const [x, y] = pts[i - 1]
    const dx = (300 - x) * 0.16 + (rnd(seed + i) - 0.5) * noise
    const dy = (95 - y) * 0.16 + (rnd(seed + i * 7) - 0.5) * noise
    pts.push([x + dx, y + dy])
  }
  return pts
}

export function BatchNoise() {
  const t = useTime()
  const k = Math.floor((t * 5) % 34)
  const small = useMemo(() => path(70, 3), [])
  const large = useMemo(() => path(10, 9), [])
  const draw = (pts: [number, number][], color: string) => (
    <>
      <polyline points={pts.slice(0, Math.min(k, pts.length) + 1).map((p) => p.join(',')).join(' ')} fill="none" stroke={color} strokeWidth={2} />
      <circle cx={pts[Math.min(k, pts.length - 1)][0]} cy={pts[Math.min(k, pts.length - 1)][1]} r={5} fill={color} />
    </>
  )
  return (
    <Frame caption="Each step estimates “downhill” from one batch of examples. A small batch (orange) gives a noisy estimate — a wobbly path, but each step is cheap. A large batch (blue) gives a smoother path but each step costs more memory and time.">
      <Svg h={210}>
        {[1, 2, 3, 4, 5].map((r) => <ellipse key={r} cx={300} cy={95} rx={r * 50} ry={r * 26} fill="none" stroke={C.line} strokeOpacity={0.6} />)}
        <circle cx={300} cy={95} r={4} fill={C.green} />
        {draw(small, C.orange)}
        {draw(large, C.blue)}
        <Label x={20} y={20} anchor="start" color={C.orange}>● batch size 4</Label>
        <Label x={20} y={36} anchor="start" color={C.blue}>● batch size 256</Label>
        <Label x={300} y={83} color={C.green}>minimum</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Optimizers on a long narrow valley
// ---------------------------------------------------------------------------
function optPath(kind: 'sgd' | 'momentum' | 'adam') {
  const a = 0.05, b = 2
  let x = -9, y = 2.2, vx = 0, vy = 0, mx = 0, my = 0, sx = 0, sy = 0
  const pts: [number, number][] = [[x, y]]
  for (let i = 1; i <= 40; i++) {
    const gx = a * x, gy = b * y
    if (kind === 'sgd') { x -= 0.85 * gx; y -= 0.85 * gy }
    else if (kind === 'momentum') { vx = 0.85 * vx + gx; vy = 0.85 * vy + gy; x -= 0.12 * vx; y -= 0.12 * vy }
    else {
      mx = 0.9 * mx + 0.1 * gx; my = 0.9 * my + 0.1 * gy
      sx = 0.999 * sx + 0.001 * gx * gx; sy = 0.999 * sy + 0.001 * gy * gy
      const mh = 1 - 0.9 ** i, sh = 1 - 0.999 ** i
      x -= (0.45 * (mx / mh)) / (Math.sqrt(sx / sh) + 1e-8)
      y -= (0.45 * (my / mh)) / (Math.sqrt(sy / sh) + 1e-8)
    }
    pts.push([x, y])
  }
  return pts
}

export function Optimizers() {
  const t = useTime()
  const k = Math.floor((t * 6) % 52)
  const paths = useMemo(() => ({ sgd: optPath('sgd'), momentum: optPath('momentum'), adam: optPath('adam') }), [])
  const P = ([x, y]: [number, number]) => `${200 + x * 19},${105 + y * 30}`
  const cols = { sgd: C.orange, momentum: C.purple, adam: C.green }
  return (
    <Frame caption="A long, narrow valley is common in real losses. Plain SGD (orange) zig-zags across it. Momentum (purple) builds up speed along the valley. Adam (green) adapts the step size for each weight separately and heads more directly to the minimum.">
      <Svg h={210}>
        {[0.3, 1, 2, 3.5, 5].map((c) => <ellipse key={c} cx={200} cy={105} rx={Math.sqrt((2 * c) / 0.05) * 19} ry={Math.sqrt((2 * c) / 2) * 30} fill="none" stroke={C.line} strokeOpacity={0.6} />)}
        {(Object.keys(paths) as (keyof typeof paths)[]).map((key) => {
          const pts = paths[key].slice(0, Math.min(k, 40) + 1)
          return (
            <g key={key}>
              <polyline points={pts.map(P).join(' ')} fill="none" stroke={cols[key]} strokeWidth={1.8} />
              <circle cx={Number(P(pts[pts.length - 1]).split(',')[0])} cy={Number(P(pts[pts.length - 1]).split(',')[1])} r={4.5} fill={cols[key]} />
            </g>
          )
        })}
        <Label x={14} y={18} anchor="start" color={C.orange}>● SGD</Label>
        <Label x={70} y={18} anchor="start" color={C.purple}>● SGD + momentum</Label>
        <Label x={190} y={18} anchor="start" color={C.green}>● Adam</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Learning-rate schedules
// ---------------------------------------------------------------------------
type Sched = 'none' | 'cosine' | 'onecycle' | 'step' | 'exponential'
const schedFn: Record<Sched, (p: number) => number> = {
  none: () => 1,
  cosine: (p) => 0.5 * (1 + Math.cos(Math.PI * p)),
  onecycle: (p) => (p < 0.3 ? 0.04 + (0.96 * p) / 0.3 : 0.001 + 0.999 * 0.5 * (1 + Math.cos((Math.PI * (p - 0.3)) / 0.7))),
  step: (p) => 0.5 ** Math.floor(p * 5),
  exponential: (p) => 0.5 ** (p * 5),
}

export function Schedules() {
  const [s, setS] = useState<Sched>('cosine')
  const t = useTime()
  const p = (t % 5) / 5
  const X = (q: number) => 40 + q * 330
  const Y = (v: number) => 175 - v * 140
  return (
    <Frame
      controls={<Choice options={(Object.keys(schedFn) as Sched[]).map((k) => ({ id: k, label: k }))} value={s} onChange={setS} />}
      caption="A schedule changes the learning rate during training: start with big steps to make fast progress, finish with small steps to settle precisely into a good spot."
    >
      <Svg h={205}>
        <line x1={40} y1={175} x2={372} y2={175} stroke={C.line} />
        <line x1={40} y1={25} x2={40} y2={175} stroke={C.line} />
        {(Object.keys(schedFn) as Sched[]).filter((k) => k !== s).map((k) => (
          <polyline key={k} points={Array.from({ length: 101 }, (_, i) => `${X(i / 100)},${Y(schedFn[k](i / 100))}`).join(' ')} fill="none" stroke={C.line} strokeWidth={1} />
        ))}
        <polyline points={Array.from({ length: 101 }, (_, i) => `${X(i / 100)},${Y(schedFn[s](i / 100))}`).join(' ')} fill="none" stroke={C.purple} strokeWidth={2.6} />
        <circle cx={X(p)} cy={Y(schedFn[s](p))} r={6} fill={C.accent} />
        <Label x={205} y={196}>training progress →</Label>
        <Label x={46} y={20} anchor="start">learning rate</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Regularisation
// ---------------------------------------------------------------------------
export function Regularization() {
  const [mode, setMode] = useState<'wd' | 'ls'>('wd')
  const t = useTime()
  const { p } = cycle(t, 4)
  const shrink = 1 - 0.6 * ease(clamp(p * 1.4))
  return (
    <Frame
      controls={<Choice options={[{ id: 'wd', label: 'Weight decay' }, { id: 'ls', label: 'Label smoothing' }]} value={mode} onChange={setMode} />}
      caption={mode === 'wd'
        ? 'Weight decay gently pulls every weight toward zero at each step. Big weights must “earn their keep” by really reducing the loss, which keeps the model simpler and less likely to memorise.'
        : 'Label smoothing replaces the hard target “100% it’s a 3” with “91% a 3, a little bit everything else”. The model stops trying to be infinitely confident.'}
    >
      <Svg h={190}>
        {mode === 'wd' ? (
          <>
            <line x1={30} y1={95} x2={370} y2={95} stroke={C.line} />
            {Array.from({ length: 12 }, (_, i) => {
              const w = (rnd(i + 3) - 0.5) * 150 * shrink
              return <rect key={i} x={40 + i * 28} y={w > 0 ? 95 - w : 95} width={18} height={Math.abs(w)} rx={3} fill={w > 0 ? C.blue : C.orange} />
            })}
            <Label x={200} y={182}>weights (shrinking toward 0)</Label>
          </>
        ) : (
          <>
            {Array.from({ length: 10 }, (_, i) => {
              const hard = i === 3 ? 1 : 0
              const soft = i === 3 ? 0.91 : 0.01
              const v = lerp(hard, soft, ease(clamp(p * 1.5)))
              return (
                <g key={i}>
                  <rect x={40 + i * 33} y={160 - Math.max(2, v * 130)} width={24} height={Math.max(2, v * 130)} rx={3} fill={i === 3 ? C.green : C.purple} />
                  <Label x={52 + i * 33} y={176}>{i}</Label>
                </g>
              )
            })}
            <Label x={200} y={20} color={C.text}>target for an image of a “3”</Label>
          </>
        )}
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Optuna: searching hyperparameter space
// ---------------------------------------------------------------------------
export function OptunaSearch() {
  const t = useTime()
  const N = 24
  const k = Math.floor((t * 1.6) % (N + 6))
  const score = (x: number, y: number) => Math.exp(-(((x - 0.62) / 0.22) ** 2) - (((y - 0.3) / 0.28) ** 2))
  const trials = useMemo(() => {
    const out: { x: number; y: number; s: number }[] = []
    for (let i = 0; i < N; i++) {
      let x: number, y: number
      if (i < 6) { x = rnd(i * 3 + 1); y = rnd(i * 5 + 2) } else {
        const b = out.reduce((a, c) => (c.s > a.s ? c : a))
        const spread = 0.35 * Math.exp(-(i - 6) / 8)
        x = clamp(b.x + (rnd(i * 11) - 0.5) * spread * 2); y = clamp(b.y + (rnd(i * 13) - 0.5) * spread * 2)
      }
      out.push({ x, y, s: score(x, y) })
    }
    return out
  }, [])
  const shown = trials.slice(0, Math.min(k, N))
  const bestT = shown.length ? shown.reduce((a, c) => (c.s > a.s ? c : a)) : null
  const median = shown.length ? [...shown].sort((a, b) => a.s - b.s)[Math.floor(shown.length / 2)].s : 0
  const X = (x: number) => 40 + x * 330
  const Y = (y: number) => 180 - y * 160
  return (
    <Frame caption="Each dot is one trial: a full (short) training run with one combination of settings. Optuna starts by sampling at random, then focuses near the best results so far. Trials that are clearly worse than average are stopped early (✕ = pruned). The background shows the true score, which Optuna can’t see — it has to discover it.">
      <Svg h={205}>
        {Array.from({ length: 22 * 10 }, (_, i) => {
          const cx = (i % 22) / 21, cy = Math.floor(i / 22) / 9
          return <rect key={i} x={X(cx) - 8} y={Y(cy) - 9} width={16} height={18} fill={heat(score(cx, cy) * 0.7, 140)} opacity={0.5} />
        })}
        {shown.map((tr, i) => tr.s < median && i > 5
          ? <text key={i} x={X(tr.x)} y={Y(tr.y) + 4} fontSize={11} textAnchor="middle" fill={C.red}>✕</text>
          : <circle key={i} cx={X(tr.x)} cy={Y(tr.y)} r={4.5} fill={C.blue} stroke="#000" strokeWidth={0.5} />)}
        {bestT && <text x={X(bestT.x)} y={Y(bestT.y) + 5} fontSize={16} textAnchor="middle" fill={C.accent}>★</text>}
        <Label x={205} y={200}>learning rate (log scale) →</Label>
        <Label x={14} y={100} anchor="middle">dropout</Label>
        <Label x={372} y={16} anchor="end" color={C.text}>trial {Math.min(k, N)} / {N}</Label>
      </Svg>
    </Frame>
  )
}
