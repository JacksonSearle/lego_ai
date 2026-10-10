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
}

interface Props {
  spec: FieldSpec
  value: unknown
  onChange: (v: unknown) => void
  disabled?: boolean
  showDefault?: boolean
}

/** A labeled form control generated from a parameter spec (shared by block params and hyperparameters). */
export function ParamField({ spec, value, onChange, disabled, showDefault = true }: Props) {
  const v = value === undefined ? spec.default : value
  const isDefault = JSON.stringify(v) === JSON.stringify(spec.default)
  const label = (
    <span className="lbl row">
      {spec.label}
      <span className="spacer" />
      {showDefault && !isDefault && (
        <button type="button" className="ghost sm" style={{ padding: '0 4px', fontSize: 11 }} onClick={() => onChange(spec.default)} title="Reset to default">
          reset
        </button>
      )}
    </span>
  )
  const help = spec.help ? (
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
          <span className="lbl" style={{ fontSize: 12, fontWeight: 600 }}>{spec.label}</span>
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
