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

/** A labelled form control generated from a parameter spec (shared by block params and hyperparameters). */
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
    const step = spec.step ?? (spec.type === 'int' ? 1 : 'any')
    return (
      <label className="field">
        {label}
        <input
          type="number"
          value={v === null || v === undefined ? '' : String(v)}
          placeholder={spec.nullable ? 'off' : undefined}
          min={spec.min ?? undefined}
          max={spec.max ?? undefined}
          step={step}
          disabled={disabled}
          onChange={(e) => {
            const raw = e.target.value
            if (raw === '') return onChange(spec.nullable ? null : spec.default)
            const n = spec.type === 'int' ? parseInt(raw, 10) : parseFloat(raw)
            if (!Number.isNaN(n)) onChange(n)
          }}
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
