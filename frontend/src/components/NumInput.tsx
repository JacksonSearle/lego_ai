import { useEffect, useState, type CSSProperties } from 'react'

interface Props {
  value: number | null | undefined
  /** Called with a valid number while typing, and with `emptyValue` when the box is left empty. */
  onChange: (v: number | null) => void
  /** What an empty box means once you leave it: null (unset) or a fallback number. */
  emptyValue?: number | null
  int?: boolean
  min?: number | null
  max?: number | null
  placeholder?: string
  disabled?: boolean
  style?: CSSProperties
  title?: string
}

const show = (v: number | null | undefined) => (v === null || v === undefined || Number.isNaN(v) ? '' : String(v))

/**
 * A number box you can freely clear and retype. The text is kept locally, so an empty or
 * half-typed value ("1e-") is never replaced under your fingers; it is only resolved when you
 * leave the field. It is a text input on purpose: no accidental changes from the scroll wheel.
 */
export function NumInput({ value, onChange, emptyValue = null, int, min, max, placeholder, disabled, style, title }: Props) {
  const [text, setText] = useState(show(value))
  const parse = (t: string) => (int ? parseInt(t, 10) : parseFloat(t))

  // Follow outside changes (reset, undo, template), but leave equivalent text alone ("0." vs 0).
  useEffect(() => {
    setText((t) => {
      const cur = t.trim() === '' ? null : parse(t)
      const same = cur === (value ?? null) || (cur !== null && Number.isNaN(cur) && value == null)
      return same ? t : show(value)
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value])

  const clampTo = (n: number) => Math.max(min ?? -Infinity, Math.min(max ?? Infinity, n))

  const commit = () => {
    const n = text.trim() === '' ? NaN : parse(text)
    if (Number.isNaN(n)) {
      setText(show(emptyValue))
      if ((value ?? null) !== emptyValue) onChange(emptyValue)
      return
    }
    const c = clampTo(n)
    setText(show(c))
    if (c !== value) onChange(c)
  }

  return (
    <input
      type="text"
      inputMode={int ? 'numeric' : 'decimal'}
      className="num-input"
      value={text}
      placeholder={placeholder}
      disabled={disabled}
      style={style}
      title={title}
      onChange={(e) => {
        const t = e.target.value
        if (!/^-?[0-9]*\.?[0-9]*(e-?[0-9]*)?$/i.test(t.trim())) return
        setText(t)
        const n = t.trim() === '' ? NaN : parse(t)
        // Apply complete, in-range numbers straight away so the rest of the page stays live.
        if (!Number.isNaN(n) && n === clampTo(n) && n !== value && !/[e.-]$/i.test(t.trim())) onChange(n)
      }}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') commit()
        if (int && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
          e.preventDefault()
          const base = Number.isNaN(parse(text)) ? (emptyValue ?? min ?? 0) : parse(text)
          const n = clampTo(base + (e.key === 'ArrowUp' ? 1 : -1))
          setText(show(n))
          onChange(n)
        }
      }}
    />
  )
}
