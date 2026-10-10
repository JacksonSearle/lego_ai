import { useMemo, useState, type ReactElement } from 'react'
import { C, Choice, Frame, Label, Svg, clamp, cycle, ease, heat, lerp, rnd, useTime } from './core'

const nodesY = (n: number, cy = 105, gap = 30) => Array.from({ length: n }, (_, i) => cy + (i - (n - 1) / 2) * gap)

// ---------------------------------------------------------------------------
// Linear: y = Wx + b
// ---------------------------------------------------------------------------
export function LinearDiagram() {
  const t = useTime()
  const { n, p } = cycle(t, 2.4)
  const xin = [0, 1, 2, 3].map((i) => +(rnd(n * 7 + i) * 2 - 1).toFixed(1))
  const W = [0, 1, 2].map((j) => [0, 1, 2, 3].map((i) => +(rnd(j * 4 + i + 50) * 2 - 1).toFixed(1)))
  const y = W.map((row) => row.reduce((a, w, i) => a + w * xin[i], 0))
  const yi = nodesY(4, 105, 38)
  const yo = nodesY(3, 105, 45)
  const focus = Math.floor(p * 3)
  return (
    <Frame caption={<>Every output is a weighted sum of <i>every</i> input, plus a bias: <code>y = W·x + b</code>. The highlighted output adds up input × weight along its four lines. The weights are what training learns.</>}>
      <Svg h={210}>
        {yi.map((a, i) => yo.map((b, j) => (
          <g key={`${i}${j}`}>
            <line x1={90} y1={a} x2={290} y2={b} stroke={W[j][i] > 0 ? C.blue : C.orange} strokeWidth={0.6 + Math.abs(W[j][i]) * 2.5} strokeOpacity={j === focus ? 0.9 : 0.15} />
            {j === focus && <Label x={lerp(90, 290, 0.32)} y={lerp(a, b, 0.32) - 3} size={9} color={W[j][i] > 0 ? C.blue : C.orange}>{W[j][i]}</Label>}
          </g>
        )))}
        {yi.map((a, i) => <g key={i}><circle cx={90} cy={a} r={14} fill={heat(clamp((xin[i] + 1) / 2) * 0.7 + 0.1)} stroke={C.line} /><Label x={90} y={a + 4} color="#fff" size={10} weight={700}>{xin[i]}</Label></g>)}
        {yo.map((b, j) => <g key={j}><circle cx={290} cy={b} r={16} fill={j === focus ? C.accent : heat(0.4, 140)} stroke={C.line} /><Label x={290} y={b + 4} color={j === focus ? '#000' : '#fff'} size={10} weight={700}>{y[j].toFixed(2)}</Label></g>)}
        <Label x={90} y={200}>inputs x (4)</Label>
        <Label x={290} y={200}>outputs y (3)</Label>
        <Label x={350} y={110} size={10} anchor="start">4×3 = 12 weights</Label>
        <Label x={350} y={124} size={10} anchor="start">+ 3 biases</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Flatten: grid → strip
// ---------------------------------------------------------------------------
export function FlattenDiagram() {
  const t = useTime()
  const { p } = cycle(t, 4)
  const k = ease(clamp(p < 0.5 ? p * 2.4 - 0.1 : (1 - p) * 2.4 - 0.1))
  return (
    <Frame caption="Flatten lays the rows of a grid end-to-end into one long list. Nothing is learned and no numbers change, only the shape: 4×4 → 16 here, 1×28×28 → 784 for MNIST.">
      <Svg h={170}>
        {Array.from({ length: 16 }, (_, i) => {
          const r = Math.floor(i / 4), c = i % 4
          return <g key={i}>
            <rect x={lerp(150 + c * 26, 16 + i * 23, k)} y={lerp(20 + r * 26, 110, k)} width={22} height={22} rx={3} fill={heat(0.15 + (r / 4) * 0.6, 200 + r * 30)} />
            <Label x={lerp(161 + c * 26, 27 + i * 23, k)} y={lerp(35 + r * 26, 125, k)} size={9} color="#fff">{i + 1}</Label>
          </g>
        })}
        <Label x={200} y={160} color={C.text} weight={600}>{k > 0.5 ? 'shape 16' : 'shape 4 × 4'}</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Output head: scores → probabilities
// ---------------------------------------------------------------------------
export function OutputHeadDiagram() {
  const t = useTime()
  const { n, p } = cycle(t, 3.2)
  const winner = [7, 2, 4, 1, 9][n % 5]
  const logits = Array.from({ length: 10 }, (_, i) => (i === winner ? 3.2 : (rnd(i + n * 10) - 0.4) * 2.4))
  const ex = logits.map((l) => Math.exp(l))
  const sum = ex.reduce((a, b) => a + b, 0)
  const probs = ex.map((e) => e / sum)
  const k = ease(clamp(p * 2 - 0.4))
  return (
    <Frame caption="The Output Head is a final Linear layer with one output per class (10 for digits). Its raw scores (“logits”) are turned into probabilities with softmax: make them positive, then divide by the total so they add up to 100%. The highest bar is the model’s answer.">
      <Svg h={200}>
        {logits.map((l, i) => {
          const v = lerp(l / 4, probs[i], k)
          const h = Math.abs(v) * 120
          return (
            <g key={i}>
              <rect x={40 + i * 33} y={v >= 0 ? 150 - h : 150} width={24} height={Math.max(1, h)} rx={3} fill={i === winner ? C.accent : v >= 0 ? C.blue : C.orange} />
              <Label x={52 + i * 33} y={168} color={i === winner && k > 0.5 ? C.accent : C.muted} weight={i === winner ? 700 : 500}>{i}</Label>
              {k > 0.8 && probs[i] > 0.04 && <Label x={52 + i * 33} y={145 - h} size={9}>{Math.round(probs[i] * 100)}%</Label>}
            </g>
          )
        })}
        <line x1={34} y1={150} x2={370} y2={150} stroke={C.line} />
        <Label x={200} y={192} color={C.text} weight={600}>{k < 0.5 ? 'raw scores (logits)' : `softmax → probabilities: it’s a ${winner}!`}</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------
const ACT: Record<string, (x: number) => number> = {
  relu: (x) => Math.max(0, x),
  gelu: (x) => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x ** 3))),
  sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  tanh: Math.tanh,
  silu: (x) => x / (1 + Math.exp(-x)),
  leaky_relu: (x) => (x > 0 ? x : 0.1 * x),
  elu: (x) => (x > 0 ? x : Math.exp(x) - 1),
  mish: (x) => x * Math.tanh(Math.log(1 + Math.exp(x))),
  softplus: (x) => Math.log(1 + Math.exp(x)),
  identity: (x) => x,
}

export function ActivationDiagram() {
  const [fn, setFn] = useState('relu')
  const t = useTime()
  const x = Math.sin(t * 0.9) * 3
  const f = ACT[fn]
  // The plot shows x and y from -3.4 to 3.4. Curves are drawn unclamped and clipped at the
  // frame, so a straight line (relu, identity) stays straight all the way to the edge.
  const R = 3.4
  const X = (v: number) => 200 + v * 50
  const Y = (v: number) => 100 - v * 27
  const notes: Record<string, string> = {
    relu: 'ReLU: negatives become 0, positives pass through. Simple, fast, the default since 2012.',
    gelu: 'GELU: a smooth ReLU used in transformers (GPT, BERT).',
    sigmoid: 'Sigmoid squashes everything into 0…1. Used in 1980s networks; its flat ends slow learning.',
    tanh: 'Tanh squashes into −1…1. Used inside RNNs and LSTMs.',
    silu: 'SiLU (also called Swish): x × sigmoid(x). Smooth, dips slightly below zero, then rises like ReLU.',
    leaky_relu: 'Leaky ReLU lets a little bit of negative signal through so neurons never fully “die”. (Drawn here with a slope of 0.1 so you can see it; the block uses 0.01.)',
    elu: 'ELU: like ReLU for positives, but negatives curve smoothly down toward −1 instead of being cut to 0.',
    mish: 'Mish: a smooth, self-regularizing cousin of SiLU, popular in some vision models.',
    softplus: 'Softplus: a smooth version of ReLU that is always positive, log(1 + eˣ).',
    identity: 'No activation (identity): stacking such layers is no better than one layer: they collapse into a single linear map.',
  }
  return (
    <Frame
      controls={<Choice options={Object.keys(ACT).map((k) => ({ id: k, label: k }))} value={fn} onChange={setFn} />}
      caption={<>An activation function bends each number after a Linear layer. Without this bend, any stack of layers is equivalent to one straight-line (linear) model. {notes[fn]}</>}
    >
      <Svg h={200}>
        <defs><clipPath id="act-frame"><rect x={X(-R)} y={Y(R)} width={X(R) - X(-R)} height={Y(-R) - Y(R)} /></clipPath></defs>
        <line x1={X(-R)} y1={100} x2={X(R)} y2={100} stroke={C.line} />
        <line x1={200} y1={Y(R)} x2={200} y2={Y(-R)} stroke={C.line} />
        {[-3, -2, -1, 1, 2, 3].map((v) => (
          <g key={v}>
            <line x1={X(v)} y1={97} x2={X(v)} y2={103} stroke={C.line} />
            <Label x={X(v)} y={115} size={9} color={C.faint}>{v}</Label>
            <line x1={197} y1={Y(v)} x2={203} y2={Y(v)} stroke={C.line} />
            <Label x={192} y={Y(v) + 3} size={9} color={C.faint} anchor="end">{v}</Label>
          </g>
        ))}
        <polyline clipPath="url(#act-frame)" points={Array.from({ length: 201 }, (_, i) => { const v = -R + (i / 200) * 2 * R; return `${X(v)},${Y(f(v))}` }).join(' ')} fill="none" stroke={C.accent} strokeWidth={2.5} />
        <line x1={X(x)} y1={100} x2={X(x)} y2={Y(f(x))} stroke={C.blue} strokeDasharray="3 3" />
        <circle cx={X(x)} cy={100} r={4} fill={C.blue} />
        <circle cx={X(x)} cy={Y(f(x))} r={6} fill={C.accent} />
        <Label x={34} y={22} anchor="start" color={C.text} weight={600}>in {x.toFixed(2)} → out {f(x).toFixed(2)}</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Dropout
// ---------------------------------------------------------------------------
export function DropoutDiagram() {
  const t = useTime()
  const tick = Math.floor(t / 0.9)
  const layers = [5, 7, 7, 3]
  const xs = [50, 150, 250, 350]
  const off = (l: number, i: number) => l > 0 && l < 3 && rnd(tick * 31 + l * 7 + i) < 0.4
  return (
    <Frame caption="During training, Dropout switches off a random 40% of neurons at every step (a different random set each time). The network can’t rely on any single neuron, so it learns more robust, spread-out patterns. When you test the model, nothing is dropped.">
      <Svg h={210}>
        {layers.slice(0, -1).map((n, l) => nodesY(n, 100, 26).map((a, i) => nodesY(layers[l + 1], 100, 26).map((b, j) => (
          <line key={`${l}${i}${j}`} x1={xs[l]} y1={a} x2={xs[l + 1]} y2={b} stroke={C.blue} strokeOpacity={off(l, i) || off(l + 1, j) ? 0.03 : 0.25} />
        ))))}
        {layers.map((n, l) => nodesY(n, 100, 26).map((y, i) => (
          <g key={`${l}${i}`}>
            <circle cx={xs[l]} cy={y} r={9} fill={off(l, i) ? C.panel : heat(0.55, 210)} stroke={off(l, i) ? C.red : C.line} strokeDasharray={off(l, i) ? '2 2' : undefined} />
            {off(l, i) && <Label x={xs[l]} y={y + 4} size={10} color={C.red}>✕</Label>}
          </g>
        )))}
        <Label x={200} y={202}>training step #{tick + 1}: dropped neurons change every step</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// MLP block: expand → activate → contract (+ optional residual)
// ---------------------------------------------------------------------------
export function MlpDiagram() {
  const t = useTime()
  const { p } = cycle(t, 2.4)
  const boxes = [
    { x: 30, w: 60, h: 70, label: 'input', sub: 'D', c: C.faint },
    { x: 120, w: 60, h: 140, label: 'Linear', sub: 'D → hidden', c: '#3b82f6' },
    { x: 200, w: 50, h: 140, label: 'act', sub: 'ReLU', c: '#eab308' },
    { x: 270, w: 60, h: 70, label: 'Linear', sub: 'hidden → D', c: '#3b82f6' },
  ]
  return (
    <Frame caption="An MLP block is two Linear layers with an activation between them. The first widens the signal (e.g. 128 → 512) so it has room to compute; the second narrows it back. With “residual” on, the input is added to the output (the curved arrow), the trick used inside every transformer.">
      <Svg h={210}>
        {boxes.map((b, i) => (
          <g key={i}>
            <rect x={b.x} y={105 - b.h / 2} width={b.w} height={b.h} rx={6} fill={b.c} opacity={clamp(p * 5 - i) * 0.6 + 0.35} />
            <Label x={b.x + b.w / 2} y={105} color="#fff" weight={700} size={11}>{b.label}</Label>
            <Label x={b.x + b.w / 2} y={120} color="#e5e7eb" size={9}>{b.sub}</Label>
          </g>
        ))}
        <path d="M 60 70 C 60 10, 300 10, 330 60" fill="none" stroke={C.green} strokeWidth={2} strokeDasharray="5 4" />
        <Label x={190} y={22} color={C.green} size={10}>residual: + input</Label>
        <circle cx={355} cy={105} r={12} fill="none" stroke={C.green} strokeWidth={2} />
        <Label x={355} y={110} color={C.green} size={14} weight={700}>+</Label>
        <circle cx={lerp(60, 340, p)} cy={105} r={5} fill={C.accent} />
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Normalization: batch norm vs layer norm
// ---------------------------------------------------------------------------
export function NormDiagram({ initial = 'batch' }: { initial?: 'batch' | 'layer' }) {
  const [mode, setMode] = useState<'batch' | 'layer'>(initial)
  const t = useTime()
  const { p, n } = cycle(t, 3)
  const k = ease(clamp(p * 1.8 - 0.3))
  const rows = 4, cols = 6
  const sel = n % (mode === 'batch' ? cols : rows)
  const vals = (r: number, c: number) => 2 + c * 0.9 + (rnd(r * 9 + c) - 0.5) * 3
  return (
    <Frame
      controls={<Choice options={[{ id: 'batch', label: 'Batch Norm' }, { id: 'layer', label: 'Layer Norm' }]} value={mode} onChange={setMode} />}
      caption={mode === 'batch'
        ? 'Batch Norm looks at one feature (a column) across all examples in the batch, and rescales it to average 0 and spread 1. It then applies a learned scale and shift. Great for CNNs.'
        : 'Layer Norm looks at one example (a row) across all its features, and rescales those to average 0 and spread 1. It doesn’t depend on the batch, which is why transformers and RNNs use it.'}
    >
      <Svg h={200}>
        <Label x={70} y={18} size={10}>{mode === 'batch' ? 'features →' : 'features →'}</Label>
        <Label x={18} y={90} size={10}>batch ↓</Label>
        {Array.from({ length: rows * cols }, (_, i) => {
          const r = Math.floor(i / cols), c = i % cols
          const inSel = mode === 'batch' ? c === sel : r === sel
          const v = vals(r, c)
          const normed = inSel ? lerp(v, (v - 4.5) / 2.2, k) : v
          return (
            <g key={i}>
              <rect x={40 + c * 30} y={28 + r * 34} width={26} height={28} rx={3} fill={heat(clamp((normed + 2) / 9), inSel ? 45 : 210)} stroke={inSel ? C.accent : 'none'} />
              <Label x={53 + c * 30} y={46 + r * 34} size={9} color="#fff">{normed.toFixed(1)}</Label>
            </g>
          )
        })}
        {/* histogram of the selected slice before → after */}
        {Array.from({ length: mode === 'batch' ? rows : cols }, (_, i) => {
          const v = mode === 'batch' ? vals(i, sel) : vals(sel, i)
          const x = lerp(250 + v * 12, 300 + ((v - 4.5) / 2.2) * 30, k)
          return <circle key={i} cx={x} cy={120 - i * 12} r={6} fill={C.accent} opacity={0.85} />
        })}
        <line x1={240} y1={140} x2={380} y2={140} stroke={C.line} />
        <line x1={300} y1={134} x2={300} y2={146} stroke={C.green} />
        <Label x={300} y={160} color={C.green} size={10}>0</Label>
        <Label x={310} y={185} size={10}>{k < 0.5 ? 'values before' : 'centered & scaled'}</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Conv2D: a kernel sliding over the image
// ---------------------------------------------------------------------------
const IMG7 = [
  [0, 0, 0, 0, 0, 0, 0],
  [0, 0, 1, 1, 1, 0, 0],
  [0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 1, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0],
]
const KERNEL = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]

export function ConvDiagram() {
  const t = useTime()
  const pos = Math.floor((t * 3) % 25)
  const r0 = Math.floor(pos / 5), c0 = pos % 5
  const out = useMemo(() => Array.from({ length: 25 }, (_, i) => {
    const r = Math.floor(i / 5), c = i % 5
    let s = 0
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) s += IMG7[r + a][c + b] * KERNEL[a][b]
    return s
  }), [])
  const cs = 22
  return (
    <Frame caption="A convolution slides a small grid of weights (the kernel or filter, here 3×3) across the image. At each position it multiplies the 9 pixels under it by the 9 weights and adds them up, one output number. This kernel detects horizontal edges (dark above, bright below). A Conv2D layer learns many such filters at once; each produces its own output map (channel).">
      <Svg h={215}>
        <Label x={95} y={14}>input 7×7</Label>
        {IMG7.flatMap((row, r) => row.map((v, c) => <rect key={`${r}${c}`} x={18 + c * cs} y={22 + r * cs} width={cs - 2} height={cs - 2} rx={2} fill={v ? '#e5e7eb' : '#1e293b'} />))}
        <rect x={18 + c0 * cs - 2} y={22 + r0 * cs - 2} width={cs * 3 + 2} height={cs * 3 + 2} rx={4} fill="none" stroke={C.accent} strokeWidth={2.5} />
        <Label x={210} y={14}>kernel 3×3</Label>
        {KERNEL.flatMap((row, r) => row.map((v, c) => (
          <g key={`k${r}${c}`}><rect x={180 + c * 22} y={40 + r * 22} width={20} height={20} rx={2} fill={v > 0 ? C.blue : v < 0 ? C.orange : C.panel} /><Label x={190 + c * 22} y={54 + r * 22} size={10} color="#fff">{v}</Label></g>
        )))}
        <Label x={212} y={130} size={18} color={C.text}>→</Label>
        <Label x={318} y={14}>output 5×5</Label>
        {out.map((v, i) => {
          const r = Math.floor(i / 5), c = i % 5
          const shown = i <= pos
          return <g key={`o${i}`}>
            <rect x={262 + c * cs} y={22 + r * cs} width={cs - 2} height={cs - 2} rx={2} fill={!shown ? C.panel : v > 0 ? heat(0.2 + v * 0.2, 210) : v < 0 ? heat(0.2 - v * 0.2, 25) : '#1e293b'} stroke={i === pos ? C.accent : 'none'} strokeWidth={2} />
            {shown && <Label x={273 + c * cs} y={37 + r * cs} size={9} color="#fff">{v}</Label>}
          </g>
        })}
        <Label x={200} y={205} color={C.text}>sum of 9 products = {out[pos]}</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Pooling
// ---------------------------------------------------------------------------
const POOL_IN = [[1, 3, 2, 0], [5, 6, 1, 2], [0, 2, 8, 4], [1, 1, 3, 7]]

export function PoolDiagram() {
  const [mode, setMode] = useState<'max' | 'avg' | 'global'>('max')
  const t = useTime()
  const w = Math.floor((t * 1.2) % 5)
  const outs = [0, 1, 2, 3].map((q) => {
    const r = Math.floor(q / 2) * 2, c = (q % 2) * 2
    const v = [POOL_IN[r][c], POOL_IN[r][c + 1], POOL_IN[r + 1][c], POOL_IN[r + 1][c + 1]]
    return mode === 'max' ? Math.max(...v) : v.reduce((a, b) => a + b) / 4
  })
  const all = POOL_IN.flat()
  const g = all.reduce((a, b) => a + b) / 16
  const cs = 34
  return (
    <Frame
      controls={<Choice options={[{ id: 'max', label: 'Max pool' }, { id: 'avg', label: 'Average pool' }, { id: 'global', label: 'Global average' }]} value={mode} onChange={setMode} />}
      caption={mode === 'global'
        ? 'Global Average Pooling averages a whole feature map into a single number, one per channel. It turns C×H×W into just C, ready for the output head, with no weights at all.'
        : `Pooling shrinks an image by summarizing each 2×2 window with its ${mode === 'max' ? 'largest value (“was this feature present anywhere here?”)' : 'average'}. The grid halves in width and height, making later layers cheaper and a little less sensitive to exact positions.`}
    >
      <Svg h={190}>
        {POOL_IN.flatMap((row, r) => row.map((v, c) => {
          const q = Math.floor(r / 2) * 2 + Math.floor(c / 2)
          const active = mode === 'global' ? w < 4 : q === w
          return <g key={`${r}${c}`}>
            <rect x={40 + c * cs} y={20 + r * cs} width={cs - 3} height={cs - 3} rx={3} fill={heat(v / 9)} stroke={active ? C.accent : 'none'} strokeWidth={2} />
            <Label x={40 + c * cs + 15} y={20 + r * cs + 21} color="#fff" weight={600}>{v}</Label>
          </g>
        }))}
        <Label x={210} y={95} size={20} color={C.text}>→</Label>
        {mode === 'global' ? (
          <g><rect x={270} y={70} width={50} height={40} rx={4} fill={heat(g / 9, 45)} /><Label x={295} y={95} color="#fff" weight={700}>{g.toFixed(2)}</Label></g>
        ) : outs.map((v, q) => (
          <g key={q}>
            <rect x={255 + (q % 2) * 44} y={50 + Math.floor(q / 2) * 44} width={40} height={40} rx={3} fill={q <= w ? heat(v / 9, 45) : C.panel} stroke={q === w ? C.accent : 'none'} strokeWidth={2} />
            {q <= w && <Label x={275 + (q % 2) * 44} y={75 + Math.floor(q / 2) * 44} color="#fff" weight={700}>{Number.isInteger(v) ? v : v.toFixed(2)}</Label>}
          </g>
        ))}
        <Label x={108} y={180}>4 × 4</Label>
        <Label x={298} y={180}>{mode === 'global' ? '1 × 1' : '2 × 2'}</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Residual block
// ---------------------------------------------------------------------------
export function ResidualDiagram() {
  const t = useTime()
  const { p } = cycle(t, 2.2)
  return (
    <Frame caption="A residual block computes a correction F(x) with two convolutions, then adds the original input back: output = F(x) + x. If a block has nothing useful to add, it can just learn F(x) ≈ 0 and pass x straight through. These shortcut paths let gradients flow through 100+ layers.">
      <Svg h={200}>
        <rect x={20} y={85} width={50} height={36} rx={6} fill={C.faint} /><Label x={45} y={107} color="#fff" weight={700}>x</Label>
        <rect x={110} y={120} width={70} height={40} rx={6} fill="#ea580c" /><Label x={145} y={138} color="#fff" weight={700} size={11}>Conv+BN</Label><Label x={145} y={151} color="#fde68a" size={9}>+ ReLU</Label>
        <rect x={210} y={120} width={70} height={40} rx={6} fill="#ea580c" /><Label x={245} y={143} color="#fff" weight={700} size={11}>Conv+BN</Label>
        <path d="M70 103 L90 103 L90 140 L110 140" fill="none" stroke={C.line} strokeWidth={2} />
        <path d="M180 140 L210 140" stroke={C.line} strokeWidth={2} />
        <path d="M280 140 L320 140 L320 116" fill="none" stroke={C.line} strokeWidth={2} />
        <path d="M90 103 L90 50 L320 50 L320 90" fill="none" stroke={C.green} strokeWidth={2.5} strokeDasharray="6 4" />
        <Label x={205} y={42} color={C.green}>shortcut: x passes straight through</Label>
        <circle cx={320} cy={103} r={13} fill={C.panel} stroke={C.accent} strokeWidth={2} /><Label x={320} y={108} color={C.accent} size={16} weight={800}>+</Label>
        <path d="M333 103 L370 103" stroke={C.line} strokeWidth={2} /><Label x={375} y={92} size={10} anchor="end">F(x)+x</Label>
        <circle cx={p < 0.5 ? lerp(90, 320, p * 2) : 320} cy={p < 0.5 ? 50 : lerp(50, 92, (p - 0.5) * 2)} r={5} fill={C.green} />
        <circle cx={lerp(90, 320, p)} cy={140} r={5} fill={C.accent} />
        <Label x={195} y={185}>main path F(x): learns what to change</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Depthwise-separable conv
// ---------------------------------------------------------------------------
export function DepthwiseDiagram() {
  const t = useTime()
  const { p } = cycle(t, 3)
  const ch = [C.red, C.green, C.blue]
  return (
    <Frame caption="A normal conv mixes all channels and all positions at once (expensive). Depthwise-separable conv splits that in two: (1) each channel gets its own small spatial filter, (2) a 1×1 conv mixes the channels. Similar power, often 8–9× fewer weights: that’s how MobileNet runs on phones.">
      <Svg h={200}>
        {ch.map((c, i) => (
          <g key={i}>
            <rect x={30} y={30 + i * 50} width={44} height={38} rx={4} fill={c} opacity={0.75} />
            <rect x={110} y={38 + i * 50} width={22} height={22} rx={3} fill="none" stroke={c} strokeWidth={2} />
            <Label x={121} y={30 + i * 50} size={9}>3×3</Label>
            <line x1={74} y1={49 + i * 50} x2={110} y2={49 + i * 50} stroke={c} strokeOpacity={p < 0.5 ? 1 : 0.3} />
            <rect x={160} y={30 + i * 50} width={44} height={38} rx={4} fill={c} opacity={p > 0.25 ? 0.75 : 0.15} />
            <line x1={132} y1={49 + i * 50} x2={160} y2={49 + i * 50} stroke={c} />
          </g>
        ))}
        {[0, 1, 2, 3].map((j) => (
          <g key={j}>
            {ch.map((c, i) => <line key={i} x1={204} y1={49 + i * 50} x2={300} y2={32 + j * 42} stroke={c} strokeOpacity={p > 0.5 ? 0.6 : 0.08} />)}
            <rect x={300} y={18 + j * 42} width={50} height={30} rx={4} fill={p > 0.7 ? C.purple : C.panel} />
          </g>
        ))}
        <Label x={120} y={190}>① per-channel filters</Label>
        <Label x={290} y={190}>② 1×1 channel mixing</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Patch embedding (ViT)
// ---------------------------------------------------------------------------
export function PatchDiagram() {
  const t = useTime()
  const { p } = cycle(t, 4)
  const k = ease(clamp(p * 2 - 0.2))
  return (
    <Frame caption="Patch Embedding cuts the image into squares (patches), flattens each, and projects it to a vector, a “visual word”. The image becomes a sequence of tokens that a transformer can read. MNIST with 4×4 patches → 7×7 = 49 tokens.">
      <Svg h={200}>
        {Array.from({ length: 16 }, (_, i) => {
          const r = Math.floor(i / 4), c = i % 4
          const x0 = 30 + c * 36, y0 = 30 + r * 36
          const x1 = 20 + i * 23, y1 = 150
          return (
            <g key={i}>
              <rect x={lerp(x0, x1, k)} y={lerp(y0, y1, k)} width={lerp(34, 20, k)} height={lerp(34, 20, k)} rx={3} fill={heat(0.25 + rnd(i) * 0.5, 300 + r * 15)} stroke={C.bg} />
              {k > 0.9 && <rect x={x1 + 2} y={112} width={16} height={30} rx={3} fill={heat(0.3 + rnd(i + 5) * 0.4, 330)} opacity={0.9} />}
            </g>
          )
        })}
        {k < 0.2 && <Label x={102} y={190}>image</Label>}
        {k > 0.9 && <Label x={200} y={104} color={C.text}>one embedding vector per patch</Label>}
        {k > 0.9 && <Label x={200} y={190}>sequence of 16 patch tokens</Label>}
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Image → sequence (rows)
// ---------------------------------------------------------------------------
export function ImageRowsDiagram() {
  const t = useTime()
  const { p } = cycle(t, 4)
  return (
    <Frame caption="Image → Sequence reads the image one row at a time: row 1 is step 1, row 2 is step 2, and so on. A 28×28 digit becomes 28 steps of 28 numbers, which an RNN or transformer can process like a sentence.">
      <Svg h={200}>
        {Array.from({ length: 7 }, (_, r) => {
          const k = ease(clamp(p * 9 - r * 0.9))
          return (
            <g key={r}>
              <rect x={lerp(40, 190, k)} y={lerp(30 + r * 20, 30 + r * 20, k)} width={110} height={16} rx={3} fill={heat(0.3 + r * 0.08, 330)} opacity={0.9} />
              {k > 0.95 && <Label x={318} y={42 + r * 20} anchor="start" size={10}>step {r + 1}</Label>}
            </g>
          )
        })}
        <Label x={95} y={185}>image rows</Label>
        <Label x={245} y={185}>sequence (time steps)</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// MLP-Mixer: token-mixing then channel-mixing
// ---------------------------------------------------------------------------
export function MixerDiagram() {
  const t = useTime()
  const { p, n } = cycle(t, 2)
  const token = n % 2 === 0
  const idx = Math.floor(p * 6)
  return (
    <Frame caption="MLP-Mixer alternates two plain MLPs. Token-mixing (columns): for each feature, mix information across all patches. Channel-mixing (rows): for each patch, mix its features. No attention needed.">
      <Svg h={200}>
        <Label x={130} y={18}>{token ? 'token-mixing: across patches ↓' : 'channel-mixing: across features →'}</Label>
        {Array.from({ length: 6 * 6 }, (_, i) => {
          const r = Math.floor(i / 6), c = i % 6
          const hl = token ? c === idx : r === idx
          return <rect key={i} x={40 + c * 32} y={30 + r * 26} width={28} height={22} rx={3} fill={hl ? (token ? C.purple : C.teal) : heat(0.2 + rnd(i) * 0.3, 260)} />
        })}
        <Label x={20} y={110} size={10}>patches</Label>
        <Label x={136} y={195} size={10}>features</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Token embedding
// ---------------------------------------------------------------------------
const WORDS = [['the', 4], ['cat', 87], ['sat', 12], ['on', 9], ['mat', 301]] as const

export function EmbeddingDiagram() {
  const t = useTime()
  const i = Math.floor(t / 1.4) % WORDS.length
  const rows = [4, 9, 12, 87, 301]
  return (
    <Frame caption="Computers can’t multiply words. A tokenizer turns each word into an id number, and the Embedding layer looks up that row in a big learned table, a list of numbers (a vector) for each word. Words used in similar ways end up with similar vectors.">
      <Svg h={200}>
        {WORDS.map(([w, id], k) => (
          <g key={w}>
            <rect x={20} y={20 + k * 34} width={56} height={26} rx={5} fill={k === i ? C.pink : C.panel} />
            <Label x={48} y={38 + k * 34} color="#fff" weight={600}>{w}</Label>
            <Label x={100} y={38 + k * 34} color={k === i ? C.accent : C.muted} weight={k === i ? 700 : 500}>#{id}</Label>
          </g>
        ))}
        <Label x={210} y={14} size={10}>embedding table (one row per word)</Label>
        {rows.map((id, r) => (
          <g key={id}>
            <Label x={148} y={38 + r * 34} anchor="end" size={10}>{id}</Label>
            {Array.from({ length: 6 }, (_, c) => <rect key={c} x={156 + c * 22} y={22 + r * 34} width={20} height={24} rx={2} fill={heat(rnd(id * 7 + c) * 0.8, 320)} stroke={id === WORDS[i][1] ? C.accent : 'none'} strokeWidth={2} />)}
          </g>
        ))}
        <Label x={350} y={95} color={C.text} weight={600}>“{WORDS[i][0]}”</Label>
        <Label x={350} y={112} size={10}>→ vector of</Label>
        <Label x={350} y={126} size={10}>6 numbers</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Sequence pooling
// ---------------------------------------------------------------------------
export function SeqPoolDiagram() {
  const t = useTime()
  const { p } = cycle(t, 3)
  const k = ease(clamp(p * 2 - 0.3))
  return (
    <Frame caption="Sequence Pool squashes a sequence of vectors (one per word) into a single vector by averaging (or taking the max, first or last). The result describes the whole sentence and can go into the output head. Averaged word embeddings is the classic “bag of words” model.">
      <Svg h={190}>
        {Array.from({ length: 5 }, (_, s) => Array.from({ length: 5 }, (_, d) => (
          <rect key={`${s}${d}`} x={lerp(40 + s * 50, 280, k)} y={30 + d * 24} width={36} height={20} rx={3} fill={heat(0.2 + rnd(s * 5 + d) * 0.6, 190)} opacity={lerp(1, 0.35, k)} />
        )))}
        {k > 0.95 && Array.from({ length: 5 }, (_, d) => <rect key={d} x={280} y={30 + d * 24} width={36} height={20} rx={3} fill={heat(0.5, 45)} />)}
        <Label x={140} y={170}>{k < 0.5 ? '5 words × 5 features' : ''}</Label>
        <Label x={298} y={170} color={k > 0.9 ? C.accent : C.muted}>{k > 0.9 ? 'one vector' : ''}</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// RNN: hidden state traveling through time
// ---------------------------------------------------------------------------
export function RnnDiagram() {
  const t = useTime()
  const { p } = cycle(t, 5)
  const words = ['I', 'love', 'deep', 'learning', '!']
  const step = Math.floor(p * 5)
  return (
    <Frame caption="An RNN reads a sequence one step at a time, carrying a hidden state (its memory, the yellow ball) from each step to the next. The same cell with the same weights is reused at every step: new memory = tanh(W·input + U·old memory). In a vanilla RNN the memory fades quickly over long sequences.">
      <Svg h={200}>
        {words.map((w, i) => (
          <g key={w}>
            <rect x={20 + i * 76} y={80} width={56} height={42} rx={8} fill={i === step ? '#22c55e' : i < step ? '#166534' : C.panel} stroke={C.line} />
            <Label x={48 + i * 76} y={106} color="#fff" weight={700} size={11}>RNN</Label>
            <Label x={48 + i * 76} y={165} color={i <= step ? C.text : C.faint} weight={600}>{w}</Label>
            <line x1={48 + i * 76} y1={150} x2={48 + i * 76} y2={124} stroke={C.line} markerEnd="" />
            {i < 4 && <line x1={76 + i * 76} y1={101} x2={96 + i * 76} y2={101} stroke={C.line} strokeWidth={2} />}
            {i <= step && <rect x={36 + i * 76} y={40} width={24} height={18} rx={3} fill={heat(0.3 + i * 0.1, 140)} />}
          </g>
        ))}
        <circle cx={48 + (p * 5) * 76 - 0} cy={101} r={7} fill={C.accent} opacity={0.9} />
        <Label x={200} y={30} size={10}>output at each step</Label>
        <Label x={200} y={190} size={10}>input words, one per time step</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// LSTM / GRU: gated memory
// ---------------------------------------------------------------------------
export function LstmDiagram({ initial = 'lstm' }: { initial?: 'lstm' | 'gru' }) {
  const [mode, setMode] = useState<'lstm' | 'gru'>(initial)
  const t = useTime()
  const g = (o: number) => 0.5 + 0.5 * Math.sin(t * 1.6 + o)
  const gates = mode === 'lstm'
    ? [{ name: 'forget', sym: '×', x: 110, v: g(0), c: C.red, note: 'what to erase' }, { name: 'input', sym: '+', x: 200, v: g(2), c: C.green, note: 'what to write' }, { name: 'output', sym: '→', x: 290, v: g(4), c: C.blue, note: 'what to reveal' }]
    : [{ name: 'reset', sym: '×', x: 140, v: g(0), c: C.red, note: 'how much past to use' }, { name: 'update', sym: '⇄', x: 260, v: g(3), c: C.green, note: 'keep old vs write new' }]
  const belt = (t * 60) % 40
  return (
    <Frame
      controls={<Choice options={[{ id: 'lstm', label: 'LSTM (1997)' }, { id: 'gru', label: 'GRU (2014)' }]} value={mode} onChange={setMode} />}
      caption={mode === 'lstm'
        ? 'The LSTM adds a “conveyor belt” (the cell state, top) that carries memory along with only small, controlled changes. Three gates, little valves between 0 (closed) and 1 (open), decide what to forget, what to write and what to output. Because the belt isn’t squashed at every step, memory and gradients survive hundreds of steps.'
        : 'The GRU is a slimmer LSTM: no separate belt, just two gates. The update gate blends old memory with the new candidate; the reset gate decides how much of the past to use when making that candidate. Often just as good, and faster.'}
    >
      <Svg h={190}>
        <rect x={30} y={30} width={340} height={22} rx={11} fill="#14532d" />
        {Array.from({ length: 10 }, (_, i) => <rect key={i} x={30 + ((i * 40 + belt) % 340)} y={36} width={14} height={10} rx={3} fill={C.green} opacity={0.7} />)}
        <Label x={200} y={22} size={10}>{mode === 'lstm' ? 'cell state: long-term memory' : 'hidden state: memory'}</Label>
        {gates.map((gt) => (
          <g key={gt.name}>
            <line x1={gt.x} y1={52} x2={gt.x} y2={100} stroke={gt.c} strokeWidth={2} strokeOpacity={0.3 + gt.v * 0.7} />
            <circle cx={gt.x} cy={41} r={11} fill={C.panel} stroke={gt.c} strokeWidth={2} />
            <Label x={gt.x} y={46} color={gt.c} weight={800} size={13}>{gt.sym}</Label>
            <rect x={gt.x - 34} y={100} width={68} height={36} rx={6} fill={gt.c} opacity={0.2 + gt.v * 0.6} />
            <Label x={gt.x} y={116} color="#fff" weight={700} size={11}>{gt.name}</Label>
            <Label x={gt.x} y={129} color="#fff" size={10}>{gt.v.toFixed(2)}</Label>
            <Label x={gt.x} y={152} size={9}>{gt.note}</Label>
          </g>
        ))}
        <Label x={200} y={180} size={10}>gates read the current input + previous hidden state</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Self-attention
// ---------------------------------------------------------------------------
const SENT = ['The', 'cat', 'sat', 'because', 'it', 'was', 'tired']
const ATT: number[][] = SENT.map((_, q) => SENT.map((_, k) => {
  if (q === 4) return [0.05, 0.62, 0.05, 0.03, 0.1, 0.05, 0.1][k] // "it" → "cat"
  if (q === 6) return [0.02, 0.45, 0.08, 0.05, 0.2, 0.12, 0.08][k]
  return k === q ? 0.4 : 0.6 * rnd(q * 13 + k) / 3
}))

export function AttentionDiagram({ causalDefault = false }: { causalDefault?: boolean }) {
  const [causal, setCausal] = useState(causalDefault)
  const t = useTime()
  const q = Math.floor(t / 1.8) % SENT.length
  const xs = SENT.map((_, i) => 32 + i * 56)
  const raw = ATT[q].map((w, k) => (causal && k > q ? 0 : w))
  const sum = raw.reduce((a, b) => a + b, 0) || 1
  const w = raw.map((v) => v / sum)
  return (
    <Frame
      controls={<Choice options={[{ id: 'full', label: 'Full attention' }, { id: 'causal', label: 'Causal (no peeking)' }]} value={causal ? 'causal' : 'full'} onChange={(v) => setCausal(v === 'causal')} />}
      caption={<>Self-attention lets every word gather information from every other word. The highlighted word asks a question (its <i>query</i>); each word offers a <i>key</i>; the better they match, the thicker the line and the more of that word’s <i>value</i> is mixed in. Here “it” attends mostly to “cat”. That’s how the model works out what “it” means. {causal && 'With a causal mask, words can only look backwards, needed when the model is predicting the next word.'}</>}
    >
      <Svg h={190}>
        {xs.map((x, k) => w[k] > 0.005 && (
          <path key={k} d={`M ${xs[q]} 120 Q ${(xs[q] + x) / 2} ${120 - 40 - Math.abs(xs[q] - x) * 0.25} ${x} 120`} fill="none" stroke={C.purple} strokeWidth={1 + w[k] * 14} strokeOpacity={0.35 + w[k]} />
        ))}
        {SENT.map((s, k) => (
          <g key={k}>
            <rect x={xs[k] - 26} y={122} width={52} height={26} rx={6} fill={k === q ? C.accent : causal && k > q ? '#111827' : C.panel} stroke={C.line} />
            <Label x={xs[k]} y={139} color={k === q ? '#000' : causal && k > q ? C.faint : C.text} weight={600}>{s}</Label>
            <Label x={xs[k]} y={168} size={9} color={C.purple}>{Math.round(w[k] * 100)}%</Label>
          </g>
        ))}
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Positional encoding
// ---------------------------------------------------------------------------
export function PositionalDiagram() {
  const t = useTime()
  const pos = Math.floor(t / 1.2) % 8
  const freqs = [1, 0.5, 0.25, 0.125]
  const X = (i: number) => 40 + i * 42
  return (
    <Frame caption="Attention by itself ignores word order (“dog bites man” = “man bites dog”). Positional encoding adds a unique pattern to each position (here, sine waves of different speeds). Read down a column and you get a position’s “fingerprint”, added to that token’s embedding.">
      <Svg h={200}>
        {freqs.map((f, r) => (
          <g key={r}>
            <polyline points={Array.from({ length: 141 }, (_, i) => { const x = (i / 140) * 7.5; return `${40 + x * 42},${40 + r * 40 - Math.sin(x * f * Math.PI) * 14}` }).join(' ')} fill="none" stroke={[C.pink, C.purple, C.blue, C.teal][r]} strokeWidth={1.8} />
            <circle cx={X(pos)} cy={40 + r * 40 - Math.sin(pos * f * Math.PI) * 14} r={5} fill={C.accent} />
          </g>
        ))}
        <line x1={X(pos)} y1={18} x2={X(pos)} y2={180} stroke={C.accent} strokeDasharray="3 3" />
        {Array.from({ length: 8 }, (_, i) => <Label key={i} x={X(i)} y={194} color={i === pos ? C.accent : C.muted} weight={i === pos ? 700 : 500}>{i}</Label>)}
        <Label x={392} y={194} anchor="end" size={9}>position</Label>
      </Svg>
    </Frame>
  )
}

// ---------------------------------------------------------------------------
// Transformer block
// ---------------------------------------------------------------------------
export function TransformerDiagram() {
  const t = useTime()
  const { p } = cycle(t, 3)
  const stages: { y: number; label: string; c: string; h?: number }[] = [
    { y: 170, label: 'tokens in', c: C.faint },
    { y: 138, label: 'LayerNorm', c: '#0f766e' },
    { y: 104, label: 'Multi-head self-attention', c: '#7c3aed', h: 26 },
    { y: 70, label: 'LayerNorm', c: '#0f766e' },
    { y: 40, label: 'Feed-forward MLP', c: '#1d4ed8', h: 24 },
  ]
  const dotY = lerp(180, 20, p)
  return (
    <Frame caption="A transformer block is two sub-layers, each with a shortcut that adds its input back (green): first self-attention (tokens share information), then a small MLP applied to each token separately (tokens think on their own). Stack many blocks and you have GPT. Data flows upward, like your Builder.">
      <Svg h={205}>
        {stages.map((s, i) => (
          <g key={i}>
            <rect x={110} y={s.y - (s.h ?? 20) / 2} width={180} height={s.h ?? 20} rx={5} fill={s.c} opacity={dotY < s.y + 10 ? 0.95 : 0.45} />
            <Label x={200} y={s.y + 4} color="#fff" weight={650} size={11}>{s.label}</Label>
          </g>
        ))}
        {[0, 1, 2, 3].map((h) => <rect key={h} x={300 + h * 15} y={96} width={11} height={16} rx={2} fill="#a78bfa" opacity={0.5 + 0.5 * Math.sin(t * 3 + h)} />)}
        <Label x={330} y={128} size={9}>4 heads</Label>
        <path d="M 100 165 L 80 165 L 80 88 L 104 88" fill="none" stroke={C.green} strokeWidth={2} />
        <path d="M 100 85 L 70 85 L 70 22 L 104 22" fill="none" stroke={C.green} strokeWidth={2} />
        <Label x={56} y={130} color={C.green} size={10}>+</Label>
        <Label x={56} y={56} color={C.green} size={10}>+</Label>
        <circle cx={200} cy={dotY} r={5} fill={C.accent} />
      </Svg>
    </Frame>
  )
}

export const BLOCK_DIAGRAMS: Record<string, () => ReactElement> = {
  linear: LinearDiagram,
  flatten: FlattenDiagram,
  classifier: OutputHeadDiagram,
  activation: ActivationDiagram,
  dropout: DropoutDiagram,
  mlp: MlpDiagram,
  batchnorm: () => <NormDiagram initial="batch" />,
  layernorm: () => <NormDiagram initial="layer" />,
  conv2d: ConvDiagram,
  pooling: PoolDiagram,
  resblock: ResidualDiagram,
  depthwise_conv: DepthwiseDiagram,
  patch_embed: PatchDiagram,
  image_to_sequence: ImageRowsDiagram,
  mixer: MixerDiagram,
  embedding: EmbeddingDiagram,
  seq_pool: SeqPoolDiagram,
  rnn: RnnDiagram,
  lstm: () => <LstmDiagram initial="lstm" />,
  gru: () => <LstmDiagram initial="gru" />,
  self_attention: () => <AttentionDiagram />,
  positional_encoding: PositionalDiagram,
  transformer: TransformerDiagram,
}
