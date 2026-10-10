import { useEffect, useRef, useState, type DragEvent, type PointerEvent } from 'react'

const PAD = 252 // on-screen size of the drawing area

/** Flattened C×H×W uint8 pixels, as the model expects them. */
type Pixels = number[]

interface Props {
  /** The model's input shape: [channels, height, width]. */
  shape: number[]
  onPixels: (px: Pixels) => void
  /** Drawing only makes sense for grayscale datasets (digits, clothing silhouettes). */
  allowDraw: boolean
}

/**
 * Give a model your own image: draw it, or drop/upload a picture. Either way it is reduced to
 * exactly what the model was trained on (for MNIST: 28×28 grayscale, white ink on black, with
 * the drawing centered like the dataset's digits).
 */
export function ImageInput({ shape, onPixels, allowDraw }: Props) {
  const [c, h, w] = shape
  const pad = useRef<HTMLCanvasElement>(null)
  const file = useRef<HTMLInputElement>(null)
  const drawing = useRef(false)
  const last = useRef<[number, number] | null>(null)
  const [mode, setMode] = useState<'draw' | 'photo'>(allowDraw ? 'draw' : 'photo')
  const [photo, setPhoto] = useState<HTMLImageElement | null>(null)
  const [invert, setInvert] = useState(false)
  const [over, setOver] = useState(false)
  const [empty, setEmpty] = useState(true)

  const clear = () => {
    const ctx = pad.current?.getContext('2d')
    if (!ctx) return
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, PAD, PAD)
    setEmpty(true)
  }
  useEffect(clear, [mode])

  /** Reduce the drawing to the model's size, centred and scaled like the training digits. */
  const sendDrawing = () => {
    const src = pad.current!
    const data = src.getContext('2d')!.getImageData(0, 0, PAD, PAD).data
    let x0 = PAD, y0 = PAD, x1 = -1, y1 = -1
    for (let y = 0; y < PAD; y++) {
      for (let x = 0; x < PAD; x++) {
        if (data[(y * PAD + x) * 4] > 40) {
          if (x < x0) x0 = x
          if (x > x1) x1 = x
          if (y < y0) y0 = y
          if (y > y1) y1 = y
        }
      }
    }
    if (x1 < 0) return
    const out = document.createElement('canvas')
    out.width = w
    out.height = h
    const ctx = out.getContext('2d')!
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, w, h)
    // Fit the ink into the central ~70% of the frame, keeping its proportions.
    const bw = x1 - x0 + 1, bh = y1 - y0 + 1
    const box = Math.min(w, h) * 0.72
    const s = box / Math.max(bw, bh)
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(src, x0, y0, bw, bh, (w - bw * s) / 2, (h - bh * s) / 2, bw * s, bh * s)
    const px = ctx.getImageData(0, 0, w, h).data
    onPixels(Array.from({ length: h * w }, (_, i) => px[i * 4]))
  }

  /** Centre-crop a photo to the model's shape. Grayscale models get brightness, colour models RGB planes. */
  const sendPhoto = (img: HTMLImageElement, inv: boolean) => {
    const out = document.createElement('canvas')
    out.width = w
    out.height = h
    const ctx = out.getContext('2d')!
    const s = Math.max(w / img.width, h / img.height)
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(img, (w - img.width * s) / 2, (h - img.height * s) / 2, img.width * s, img.height * s)
    const px = ctx.getImageData(0, 0, w, h).data
    const n = h * w
    if (c === 1) {
      onPixels(Array.from({ length: n }, (_, i) => {
        const v = Math.round(0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2])
        return inv ? 255 - v : v
      }))
    } else {
      onPixels(Array.from({ length: 3 * n }, (_, k) => px[(k % n) * 4 + Math.floor(k / n)]))
    }
  }

  const loadFile = (f: File | undefined) => {
    if (!f || !f.type.startsWith('image/')) return
    const img = new Image()
    img.onload = () => {
      // Dark ink on light paper is the opposite of the training data: flip it by default.
      let inv = false
      if (c === 1) {
        const t = document.createElement('canvas')
        t.width = t.height = 16
        const tc = t.getContext('2d')!
        tc.drawImage(img, 0, 0, 16, 16)
        const d = tc.getImageData(0, 0, 16, 16).data
        let sum = 0
        for (let i = 0; i < 256; i++) sum += d[i * 4] + d[i * 4 + 1] + d[i * 4 + 2]
        inv = sum / (256 * 3) > 127
      }
      setInvert(inv)
      setPhoto(img)
      setMode('photo')
      sendPhoto(img, inv)
    }
    img.src = URL.createObjectURL(f)
  }

  const pos = (e: PointerEvent): [number, number] => {
    const r = pad.current!.getBoundingClientRect()
    return [((e.clientX - r.left) / r.width) * PAD, ((e.clientY - r.top) / r.height) * PAD]
  }
  const stroke = (e: PointerEvent) => {
    const ctx = pad.current!.getContext('2d')!
    const p = pos(e)
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 18
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.beginPath()
    ctx.moveTo(...(last.current ?? p))
    ctx.lineTo(...p)
    ctx.stroke()
    last.current = p
    setEmpty(false)
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setOver(false)
    loadFile(e.dataTransfer.files[0])
  }

  return (
    <div className="image-input">
      <div className="row" style={{ marginBottom: 8 }}>
        {allowDraw && (
          <div className="seg">
            <button className={mode === 'draw' ? 'on' : ''} onClick={() => setMode('draw')}>✏️ Draw</button>
            <button className={mode === 'photo' ? 'on' : ''} onClick={() => setMode('photo')}>🖼️ Photo</button>
          </div>
        )}
        <span className="spacer" />
        {mode === 'draw' && <button className="sm" onClick={clear} disabled={empty}>Clear</button>}
        {mode === 'photo' && <button className="sm" onClick={() => file.current?.click()}>Choose a file…</button>}
        <input ref={file} type="file" accept="image/*" hidden onChange={(e) => { loadFile(e.target.files?.[0]); e.target.value = '' }} />
      </div>
      {mode === 'draw' ? (
        <canvas
          ref={pad}
          width={PAD}
          height={PAD}
          className="draw-pad"
          style={{ touchAction: 'none' }}
          onPointerDown={(e) => { drawing.current = true; last.current = null; (e.target as Element).setPointerCapture?.(e.pointerId); stroke(e) }}
          onPointerMove={(e) => drawing.current && stroke(e)}
          onPointerUp={() => { if (drawing.current) { drawing.current = false; sendDrawing() } }}
        />
      ) : (
        <div
          className={`drop-zone ${over ? 'hot' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setOver(true) }}
          onDragLeave={() => setOver(false)}
          onDrop={onDrop}
          onClick={() => file.current?.click()}
        >
          {photo ? <img src={photo.src} alt="Your upload" /> : <span>Drop a picture here<br /><span className="tiny muted">or click to choose one</span></span>}
        </div>
      )}
      {mode === 'photo' && c === 1 && photo && (
        <label className="row small muted" style={{ marginTop: 6 }} title="The training images are light shapes on a dark background. Photos of dark ink on paper need flipping.">
          <input type="checkbox" checked={invert} onChange={(e) => { setInvert(e.target.checked); sendPhoto(photo, e.target.checked) }} /> invert (dark ink on light paper)
        </label>
      )}
      <div className="tiny faint" style={{ marginTop: 6 }}>
        {mode === 'draw' ? `Draw with the mouse. It is shrunk to ${w}×${h} and centered, like the training images.` : `Your picture is cropped to a square and shrunk to ${w}×${h}${c === 1 ? ' in grayscale' : ''}: that is all the model sees.`}
      </div>
    </div>
  )
}
