import { useMemo } from 'react'
import type { EpochPoint, StepPoint } from '../types'
import { LineChart } from './LineChart'

const C = { train: '#60a5fa', val: '#fb923c', lr: '#a78bfa' }

/** Loss + accuracy (+ optional learning-rate) charts for one run. */
export function LiveCharts({ steps, epochs, showLr = true, height = 230, task, stacked }: {
  steps: StepPoint[]; epochs: EpochPoint[]; showLr?: boolean; height?: number; task?: string; stacked?: boolean
}) {
  const loss = useMemo(() => [
    { label: 'train loss', color: C.train, points: steps.map((s) => ({ x: s.epoch, y: s.loss })), width: 1.5 },
    { label: 'val loss', color: C.val, points: epochs.map((e) => ({ x: e.epoch, y: e.val_loss })), showPoints: true, width: 2.2 },
  ], [steps, epochs])
  const acc = useMemo(() => [
    { label: 'train acc', color: C.train, points: steps.map((s) => ({ x: s.epoch, y: s.acc })), width: 1.5 },
    { label: 'val acc', color: C.val, points: epochs.map((e) => ({ x: e.epoch, y: e.val_acc })), showPoints: true, width: 2.2 },
  ], [steps, epochs])
  const lr = useMemo(() => [{ label: 'learning rate', color: C.lr, points: steps.map((s) => ({ x: s.epoch, y: s.lr })) }], [steps])
  return (
    <div className="charts" style={stacked ? { gridTemplateColumns: '1fr' } : undefined}>
      <div className="chart-box">
        <h4>Loss</h4>
        <LineChart series={loss} height={height} />
      </div>
      <div className="chart-box">
        <h4>{task === 'lm' ? 'Next-token accuracy' : 'Accuracy'}</h4>
        <LineChart series={acc} percent height={height} />
      </div>
      {showLr && (
        <div className="chart-box" style={{ gridColumn: '1 / -1' }}>
          <h4>Learning rate</h4>
          <LineChart series={lr} height={120} />
        </div>
      )}
    </div>
  )
}
