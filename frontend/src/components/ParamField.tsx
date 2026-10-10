import { useState } from 'react'
import { CurveEditor } from './CurveEditor'
import { NumInput } from './NumInput'

interface FieldSpec {
  name: string
  label: string
  type: string
  default: unknown
  help?: string
  min?: number | null
  max?: number | null
  step?: number | null
  choices?: unknown[] | null
  nullable?: boolean
  xs?: number[]
  y_limit?: number
}

interface Props {
  spec: FieldSpec
  value: unknown
  onChange: (v: unknown) => void
  disabled?: boolean
  showDefault?: boolean
  /** Hide the explanation behind a small “?” (hover shows it, click pins it) to save space. */
  brief?: boolean
}

/** A labeled form control generated from a parameter spec (shared by block params and hyperparameters). */
export function ParamField({ spec, value, onChange, disabled, showDefault = true, brief }: Props) {
  const [pinned, setPinned] = useState(false)
  const v = value === undefined ? spec.default : value
  const isDefault = JSON.stringify(v) === JSON.stringify(spec.default)
  const label = (
    <span className="lbl row" title={brief ? spec.help : undefined}>
      {spec.label}
      {brief && spec.help && (
        <button type="button" className={`ghost sm help-toggle ${pinned ? 'on' : ''}`} onClick={(e) => { e.preventDefault(); setPinned(!pinned) }} title={spec.help}>?</button>
      )}
      <span className="spacer" />
      {showDefault && !isDefault && (
        <button type="button" className="ghost sm" style={{ padding: '0 4px', fontSize: 11 }} onClick={() => onChange(spec.default)} title="Reset to default">
          reset
        </button>
      )}
    </span>
  )
  const help = spec.help && (!brief || pinned) ? (
    <span className="help">
      {spec.help}
      {showDefault && spec.default !== null && spec.default !== undefined && spec.type !== 'bool' && (
        <span className="faint"> Default: {String(spec.default)}.</span>
      )}
    </span>
  ) : null

  if (spec.type === 'bool') {
    return (
      <label className="field">
        <span className="row">
          <label className="switch">
            <input type="checkbox" checked={!!v} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
            <span className="track" />
          </label>
          <span className="lbl" style={{ fontSize: 12, fontWeight: 600 }} title={brief ? spec.help : undefined}>{spec.label}</span>
          {brief && spec.help && (
            <button type="button" className={`ghost sm help-toggle ${pinned ? 'on' : ''}`} onClick={(e) => { e.preventDefault(); setPinned(!pinned) }} title={spec.help}>?</button>
          )}
        </span>
        {help}
      </label>
    )
  }

  if (spec.type === 'choice' || spec.type === 'activation') {
    return (
      <label className="field">
        {label}
        <select value={String(v)} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
          {(spec.choices ?? []).map((c) => (
            <option key={String(c)} value={String(c)}>{String(c)}</option>
          ))}
        </select>
        {help}
      </label>
    )
  }

  if (spec.type === 'curve') {
    return (
      <div className="field">
        {label}
        <CurveEditor xs={spec.xs ?? []} yLimit={spec.y_limit ?? 4} value={(v as number[]) ?? []} onChange={onChange} disabled={disabled} />
        {spec.help && <span className="help">{spec.help}</span>}
      </div>
    )
  }

  if (spec.type === 'int' || spec.type === 'float') {
    return (
      <label className="field">
        {label}
        <NumInput
          value={v as number | null}
          int={spec.type === 'int'}
          min={spec.min}
          max={spec.max}
          placeholder={spec.nullable ? 'off' : undefined}
          disabled={disabled}
          emptyValue={spec.nullable ? null : (spec.default as number)}
          onChange={onChange}
        />
        {help}
      </label>
    )
  }

  return (
    <label className="field">
      {label}
      <input value={String(v ?? '')} disabled={disabled} onChange={(e) => onChange(e.target.value)} />
      {help}
    </label>
  )
}
