import type { CSSProperties, DragEvent, MouseEvent } from 'react'
import type { BlockInst, BlockReport, BlockSpec } from '../types'
import { fmtNum, paramSummary, tone } from '../util'

interface Props {
  spec: BlockSpec | undefined
  inst: BlockInst
  report?: BlockReport
  selected?: boolean
  dragging?: boolean
  onClick?: (e: MouseEvent) => void
  onDragStart?: (e: DragEvent) => void
  onDragEnd?: (e: DragEvent) => void
  compact?: boolean
}

/** One Lego brick in the stack. Repeated blocks show stacked shadows and a ×N badge to the side. */
export function Brick({ spec, inst, report, selected, dragging, onClick, onDragStart, onDragEnd, compact }: Props) {
  const color = spec?.color ?? '#475569'
  const cls = [
    'brick',
    tone(color),
    selected && 'selected',
    report?.error && 'error',
    dragging && 'dragging',
    inst.repeat > 1 && 'stacked',
  ].filter(Boolean).join(' ')
  const style = { '--c': color, minHeight: compact ? 38 : undefined, padding: compact ? '6px 12px' : undefined } as CSSProperties
  return (
    <div
      className={cls}
      style={style}
      onClick={onClick}
      draggable={!!onDragStart}
      onDragStart={onDragStart}
      onDragEnd={onDragEnd}
      title={spec?.description}
    >
      <div className="brick-title">
        {spec?.name ?? inst.type}
        {spec?.source === 'custom' && <span className="src-tag">PY</span>}
      </div>
      <div className="brick-sub">{spec ? paramSummary(spec, inst.params) || ' ' : 'unknown block (deleted custom block?)'}</div>
      {!compact && report && !report.error && report.params > 0 && <div className="brick-params">{fmtNum(report.params)} params</div>}
      {report?.error && <div className="brick-err">{report.error}</div>}
      {inst.repeat > 1 && <div className="repeat-badge">×{inst.repeat}</div>}
    </div>
  )
}
