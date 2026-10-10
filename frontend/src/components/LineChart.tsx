import { useEffect, useRef } from 'react'
import {
  Chart, CategoryScale, Filler, Legend, LinearScale, LineController, LineElement, LogarithmicScale, PointElement, Tooltip,
  type ChartDataset,
} from 'chart.js'

Chart.register(LineController, LineElement, PointElement, LinearScale, LogarithmicScale, CategoryScale, Legend, Tooltip, Filler)
Chart.defaults.color = '#8f9ab0'
Chart.defaults.borderColor = '#263049'
Chart.defaults.font.family = 'Inter, ui-sans-serif, system-ui, sans-serif'

// Which series the user has switched off in a chart legend. Remembered by series name, so the
// choice survives live updates, new runs and page reloads.
const HIDDEN_KEY = 'lego-ai/hidden-series'
const hidden: Set<string> = (() => {
  try {
    return new Set<string>(JSON.parse(localStorage.getItem(HIDDEN_KEY) ?? '[]'))
  } catch {
    return new Set<string>()
  }
})()

export interface Series {
  label: string
  color: string
  points: { x: number; y: number }[]
  dashed?: boolean
  showPoints?: boolean
  width?: number
}

interface Props {
  series: Series[]
  xLabel?: string
  yLabel?: string
  yMin?: number
  yMax?: number
  logY?: boolean
  percent?: boolean
  height?: number
}

/** Chart.js line chart that updates in place (no re-create) as points stream in. */
export function LineChart({ series, xLabel = 'epoch', yLabel, yMin, yMax, logY, percent, height = 230 }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const chart = useRef<Chart<'line', { x: number; y: number }[]> | null>(null)

  useEffect(() => {
    if (!canvas.current) return
    chart.current = new Chart(canvas.current, {
      type: 'line',
      data: { datasets: [] },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        parsing: false,
        normalized: true,
        interaction: { mode: 'nearest', intersect: false },
        plugins: {
          legend: {
            position: 'top',
            align: 'end',
            labels: { boxWidth: 10, boxHeight: 10, usePointStyle: true, font: { size: 11 } },
            onClick: (_e, item, legend) => {
              const label = item.text
              if (hidden.has(label)) hidden.delete(label)
              else hidden.add(label)
              localStorage.setItem(HIDDEN_KEY, JSON.stringify([...hidden]))
              const i = item.datasetIndex ?? 0
              legend.chart.setDatasetVisibility(i, !hidden.has(label))
              legend.chart.update('none')
            },
          },
          tooltip: {
            callbacks: {
              label: (c) => `${c.dataset.label}: ${percent ? `${((c.parsed.y ?? 0) * 100).toFixed(2)}%` : (c.parsed.y ?? 0).toFixed(4)}`,
              title: (items) => `${xLabel} ${items[0]?.parsed.x?.toFixed(2)}`,
            },
          },
        },
        scales: {
          x: { type: 'linear', title: { display: true, text: xLabel, font: { size: 11 } }, grid: { color: '#1b2336' } },
          y: {
            type: logY ? 'logarithmic' : 'linear',
            min: yMin,
            max: yMax,
            title: { display: !!yLabel, text: yLabel, font: { size: 11 } },
            grid: { color: '#1b2336' },
            ticks: { callback: (v) => (percent ? `${Math.round(Number(v) * 100)}%` : Number(v).toPrecision(2)) },
          },
        },
      },
    })
    return () => {
      chart.current?.destroy()
      chart.current = null
    }
  }, [xLabel, yLabel, yMin, yMax, logY, percent])

  useEffect(() => {
    const c = chart.current
    if (!c) return
    c.data.datasets = series.map(
      (s): ChartDataset<'line', { x: number; y: number }[]> => ({
        label: s.label,
        data: s.points,
        borderColor: s.color,
        backgroundColor: s.color,
        borderWidth: s.width ?? 1.8,
        borderDash: s.dashed ? [5, 4] : undefined,
        pointRadius: s.showPoints ? 3.5 : 0,
        pointHoverRadius: 5,
        tension: 0.25,
        hidden: hidden.has(s.label),
      }),
    )
    c.update('none')
  }, [series])

  return (
    <div className="chart-canvas" style={{ height }}>
      <canvas ref={canvas} />
    </div>
  )
}
