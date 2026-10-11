import type { DatasetInfo } from '../types'
import { fmtPct } from '../util'

export const DIFFICULTY = ['', 'Easy', 'Medium', 'Hard']
const CLS = ['', 'ok', 'warn', 'bad']

export const difficultyTag = (d: DatasetInfo | undefined) => (d ? (d.custom ? 'Yours' : DIFFICULTY[d.difficulty] ?? '') : '')

/** How hard a dataset is, what to expect, and reference accuracies to aim for. */
export function Difficulty({ dataset, compact }: { dataset: DatasetInfo | undefined; compact?: boolean }) {
  if (!dataset) return null
  return (
    <div className="difficulty">
      <div className="row wrap" style={{ gap: 6 }}>
        {dataset.custom ? <span className="pill blue" title="A dataset you imported">your data</span> : (
          <span className={`pill ${CLS[dataset.difficulty] ?? ''}`} title="How hard it is to get good results on this dataset">
            {'●'.repeat(dataset.difficulty)}{'○'.repeat(3 - dataset.difficulty)} {DIFFICULTY[dataset.difficulty]}
          </span>
        )}
        {dataset.targets && (
          <span className="tiny muted" title="Validation accuracy: a trivial baseline, a good result and a great one">
            baseline {fmtPct(dataset.targets.baseline, 0)} · good <b>{fmtPct(dataset.targets.good, 0)}</b> · great <b>{fmtPct(dataset.targets.great, dataset.targets.great > 0.99 ? 1 : 0)}</b>
          </span>
        )}
      </div>
      {!compact && <div className="small muted" style={{ marginTop: 5 }}>{dataset.difficulty_note}</div>}
    </div>
  )
}
