import type { BlockInst, BlockSpec, DatasetInfo } from '../types'
import { fmtShape } from '../util'
import { Brick } from './Brick'

/** Read-only vertical stack (input at the bottom, output on top). */
export function ArchView({ blocks, specs, dataset, width = 280 }: {
  blocks: BlockInst[]
  specs: Record<string, BlockSpec>
  dataset?: DatasetInfo
  width?: number
}) {
  return (
    <div style={{ width, display: 'flex', flexDirection: 'column', gap: 12, paddingRight: 44 }}>
      {dataset && (
        <div className="io-node" style={{ padding: '6px 10px' }}>
          <div className="io-label">OUTPUT</div>
          <div className="io-shape" style={{ fontSize: 13 }}>
            {dataset.task === 'lm' ? fmtShape([dataset.seq_len!, dataset.vocab_size!]) : `${dataset.num_classes} classes`}
          </div>
        </div>
      )}
      {[...blocks].reverse().map((b) => (
        <Brick key={b.id} spec={specs[b.type]} inst={b} compact />
      ))}
      {dataset && (
        <div className="io-node" style={{ padding: '6px 10px' }}>
          <div className="io-label">INPUT · {dataset.name}</div>
          <div className="io-shape" style={{ fontSize: 13 }}>{fmtShape(dataset.input_shape)}</div>
        </div>
      )}
    </div>
  )
}

/** Tiny coloured preview of an architecture. */
export function MiniStack({ blocks, specs, width = 120 }: { blocks: BlockInst[]; specs: Record<string, BlockSpec>; width?: number }) {
  return (
    <div className="mini-stack" style={{ width }}>
      {blocks.map((b) => (
        <div key={b.id} className="mini-brick" style={{ background: specs[b.type]?.color ?? '#475569' }} title={specs[b.type]?.name}>
          {b.repeat > 1 && <span className="x">×{b.repeat}</span>}
        </div>
      ))}
    </div>
  )
}

/** Inline row of colour chips, one per block. */
export function Chips({ types, specs }: { types: string[]; specs: Record<string, BlockSpec> }) {
  return (
    <div className="chips">
      {types.map((t, i) => (
        <span key={i} className="chip" style={{ background: specs[t]?.color ?? '#475569' }} title={specs[t]?.name ?? t} />
      ))}
    </div>
  )
}
