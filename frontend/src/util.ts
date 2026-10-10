import type { BlockInst, BlockSpec, DatasetInfo } from './types'

export const fmtShape = (s: number[] | null | undefined) => (s && s.length ? s.join('×') : '-')

export function fmtNum(n: number | null | undefined) {
  if (n == null) return '-'
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 1 : 2)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}k`
  return String(n)
}

export const fmtPct = (x: number | null | undefined, d = 2) => (x == null || x < 0 ? '-' : `${(x * 100).toFixed(d)}%`)
export const fmtLoss = (x: number | null | undefined) => (x == null ? '-' : x.toFixed(4))

export function fmtDur(s: number | null | undefined) {
  if (s == null) return '-'
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)}s`
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
  return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`
}

export function fmtAgo(t: number) {
  const s = Date.now() / 1000 - t
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return new Date(t * 1000).toLocaleDateString()
}

/** 'light' when the color is bright enough to need dark text. */
export function tone(hex: string): 'light' | 'dark' {
  const h = hex.replace('#', '')
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255)
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
  return lum > 0.6 ? 'light' : 'dark'
}

const HIDDEN_IN_SUMMARY = new Set(['padding', 'causal', 'norm_first', 'ff_mult', 'return_sequences', 'bias'])
const BOOL_TAGS: Record<string, string> = { batch_norm: 'BN', bidirectional: 'bidir', residual: 'residual' }

/** Short one-line description of a block's settings, e.g. "256 · relu". */
export function paramSummary(spec: BlockSpec | undefined, params: Record<string, unknown>) {
  if (!spec) return ''
  const parts: string[] = []
  for (const p of spec.params) {
    const v = params[p.name] ?? p.default
    if (HIDDEN_IN_SUMMARY.has(p.name)) continue
    if (p.type === 'curve') {
      parts.push('hand-drawn')
      continue
    }
    if (p.type === 'bool') {
      if (v && BOOL_TAGS[p.name]) parts.push(BOOL_TAGS[p.name])
    } else if (v === 0 && (p.name === 'out_features' || p.name === 'out_channels')) parts.push('same')
    else if (p.name === 'stride') {
      if ((v as number) > 1) parts.push(`stride ${v}`)
    } else if (p.type === 'activation') parts.push(v === 'none' || v === 'identity' ? 'linear' : String(v))
    else if (p.name === 'kernel_size') parts.push(`${v}×${v}`)
    else if (p.name === 'heads') parts.push(`${v} heads`)
    else if (p.name === 'dropout' || p.name === 'p') {
      if (v) parts.push(`drop ${v}`)
    } else if (p.name === 'layers') {
      if ((v as number) > 1) parts.push(`${v} layers`)
    } else parts.push(String(v))
  }
  return parts.slice(0, 4).join(' · ')
}

export function describeInput(ds: DatasetInfo | undefined) {
  if (!ds) return ''
  if (ds.input_kind === 'image') return `${ds.input_shape[0] === 1 ? 'grayscale' : 'color'} image`
  if (ds.input_kind === 'tokens') return `${ds.seq_len} token ids (vocab ${ds.vocab_size})`
  return `${ds.input_shape[0]} features`
}

export function describeOutput(ds: DatasetInfo | undefined) {
  if (!ds) return ''
  if (ds.task === 'lm') return `next-token logits at each of ${ds.seq_len} steps`
  return `${ds.num_classes} class scores`
}

export function cloneBlock(b: BlockInst, id: string): BlockInst {
  return { ...b, id, params: { ...b.params } }
}

export function defaultParams(spec: BlockSpec): Record<string, unknown> {
  return Object.fromEntries(spec.params.map((p) => [p.name, p.default]))
}

export const MEDALS = ['🥇', '🥈', '🥉']

/**
 * Fit y = a · x^(-b) by least squares in log-log space (the shape most loss curves follow once
 * the first rapid drop is over). Returns null when there is too little usable data.
 */
export function fitPowerLaw(pts: { x: number; y: number }[], minPoints = 6): { a: number; b: number } | null {
  const use = pts.filter((p) => p.x > 0 && p.y > 0)
  if (use.length < minPoints) return null
  const lx = use.map((p) => Math.log(p.x))
  const ly = use.map((p) => Math.log(p.y))
  const n = use.length
  const mx = lx.reduce((s, v) => s + v, 0) / n
  const my = ly.reduce((s, v) => s + v, 0) / n
  let sxx = 0
  let sxy = 0
  for (let i = 0; i < n; i++) {
    sxx += (lx[i] - mx) ** 2
    sxy += (lx[i] - mx) * (ly[i] - my)
  }
  if (sxx === 0) return null
  const slope = sxy / sxx
  return { a: Math.exp(my - slope * mx), b: -slope }
}

/** Where a fitted power law reaches `target` (x), or null if the trend never gets there. */
export function powerLawReach(fit: { a: number; b: number } | null, target: number): number | null {
  if (!fit || fit.b <= 0.01 || target <= 0) return null
  const x = (fit.a / target) ** (1 / fit.b)
  return Number.isFinite(x) ? x : null
}
