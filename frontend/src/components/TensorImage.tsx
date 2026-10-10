import { useEffect, useRef } from 'react'

/** Draws a raw uint8 image (flattened C×H×W, as returned by the API) onto a canvas. */
export function TensorImage({ data, shape, size = 56, className }: { data: number[]; shape: number[]; size?: number; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const cv = ref.current
    if (!cv) return
    const [c, h, w] = shape.length === 3 ? shape : [1, shape[0], shape[1]]
    cv.width = w
    cv.height = h
    const ctx = cv.getContext('2d')!
    const img = ctx.createImageData(w, h)
    for (let i = 0; i < h * w; i++) {
      const r = data[i]
      const g = c === 3 ? data[h * w + i] : r
      const b = c === 3 ? data[2 * h * w + i] : r
      img.data.set([r, g, b, 255], i * 4)
    }
    ctx.putImageData(img, 0, 0)
  }, [data, shape])
  return <canvas ref={ref} className={className} style={{ width: size, height: size, imageRendering: 'pixelated', borderRadius: 4, background: '#000' }} />
}
