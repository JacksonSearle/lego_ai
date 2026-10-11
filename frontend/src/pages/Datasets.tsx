import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { Difficulty } from '../components/Difficulty'
import { TensorImage } from '../components/TensorImage'
import { GuideButton } from '../learn/GuideButton'
import { useLearn } from '../learn/LearnContext'
import { useStore } from '../store'
import type { DatasetInfo, DatasetPreview } from '../types'
import { fmtNum } from '../util'

const KIND_ICON: Record<string, string> = { image: '🖼️', tokens: '📝', vector: '🔢' }
const kindLabel = (d: DatasetInfo) => (d.task === 'lm' ? 'text to continue' : d.input_kind === 'image' ? 'images' : d.input_kind === 'tokens' ? 'text' : 'table of numbers')

/** Read a file as base64 (without the "data:...;base64," prefix). */
const readB64 = (f: File) => new Promise<string>((resolve, reject) => {
  const r = new FileReader()
  r.onload = () => resolve(String(r.result).split(',')[1] ?? '')
  r.onerror = () => reject(new Error('Could not read the file'))
  r.readAsDataURL(f)
})

/** The narrated steps of the worked example. */
const COACH = [
  { title: 'Step 1 of 4 · Meet the data', text: 'This is the Iris table, a classic first dataset: 150 flowers, four measurements each (in centimeters), and the species a botanist assigned. Every row is one example. Look at the sample rows below.' },
  { title: 'Step 2 of 4 · Which column is the answer?', text: 'A model learns to predict one column from the others. Here the answer is “species”: it has only 3 different values, which makes it a category. The app guessed it for you (the selected circle). The answer column is also called the label.' },
  { title: 'Step 3 of 4 · Which columns are the inputs?', text: 'The four measurements are ticked as inputs. On import each one is rescaled to be centered on zero with a similar spread, because networks learn poorly when one input is in the thousands and another in fractions. Untick a column to see if the model can cope without it.' },
  { title: 'Step 4 of 4 · Hold some back', text: 'Part of the data (15% here) is set aside as a test set the model never trains on. Scoring on rows it has not seen is the only way to know it learned the pattern and did not just memorize the rows. Press Import.' },
]

export function Datasets() {
  const { datasets, refreshDatasets, setDraft, setTab, toast, draft } = useStore()
  const { flag, access } = useLearn()
  const [pv, setPv] = useState<DatasetPreview | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [coach, setCoach] = useState<number | null>(null)
  const [done, setDone] = useState<DatasetInfo | null>(null)
  const [repo, setRepo] = useState('')
  // What the user chose in the preview.
  const [name, setName] = useState('')
  const [label, setLabel] = useState<string | null>(null)
  const [inputs, setInputs] = useState<string[]>([])
  const [size, setSize] = useState(32)
  const [gray, setGray] = useState(false)
  const [maxLen, setMaxLen] = useState(64)
  const [testFrac, setTestFrac] = useState(15)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => { refreshDatasets() }, [refreshDatasets])

  const open = (p: DatasetPreview, suggested: string) => {
    setPv(p)
    setDone(null)
    setErr(null)
    setName(suggested)
    setLabel(p.suggest.label)
    setInputs(p.suggest.inputs)
    const img = p.rows[0]?.find((c) => c && typeof c === 'object') as { size: number[] } | undefined
    if (img) {
      const s = Math.min(...img.size)
      setSize(s <= 20 ? 16 : s <= 30 ? 28 : s <= 40 ? 32 : s <= 56 ? 48 : 64)
    }
  }
  const run = async (what: string, fn: () => Promise<void>) => {
    setBusy(what)
    setErr(null)
    try { await fn() } catch (e) { setErr((e as Error).message) } finally { setBusy(null) }
  }
  const startExample = () => run('Opening the example…', async () => { open(await api.uploadDataset({ example: true }), 'Iris flowers'); setCoach(0) })
  const upload = (f: File) => run(`Reading ${f.name}…`, async () => {
    if (f.size > 150e6) throw new Error('That file is over 150 MB. Try a smaller sample of it.')
    setCoach(null)
    open(await api.uploadDataset({ filename: f.name, content: await readB64(f) }), f.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' '))
  })
  const fromHub = () => run('Downloading from Hugging Face…', async () => { setCoach(null); open(await api.hubDataset(repo.trim()), repo.trim().split('/').pop() ?? 'dataset') })

  const types = pv ? Object.fromEntries(pv.columns.map((c) => [c.name, c.type])) : {}
  const inKinds = new Set(inputs.map((c) => types[c]))
  const problem = !pv || pv.kind === 'text' ? null
    : !label ? 'Choose the answer column.'
    : inputs.length === 0 ? 'Tick at least one input column.'
    : inKinds.size > 1 || ((inKinds.has('text') || inKinds.has('image')) && inputs.length > 1) ? 'Inputs must be of one kind: several number columns, or one text column, or one image column.'
    : null
  const doImport = () => run('Importing…', async () => {
    const info = await api.importDataset({ token: pv!.token, name, label, inputs, image_size: size, grayscale: gray, max_len: maxLen, test_fraction: testFrac / 100 })
    await refreshDatasets()
    flag('importDataset')
    if (coach != null) flag('datasetExample')
    setDone(info)
    setPv(null)
    setCoach(null)
  })
  const use = (d: DatasetInfo) => {
    setDraft((x) => ({ ...x, dataset: d.id, seq_len: null, train_subset: null, blocks: x.dataset === d.id ? x.blocks : [], name: x.dataset === d.id ? x.name : 'My model', init_from: undefined }))
    setTab('build')
    toast(`Now building for “${d.name}”. Pick a template to start.`, 'ok')
  }
  const remove = async (d: DatasetInfo) => {
    if (!confirm(`Delete the dataset “${d.name}”? Models already trained on it stay on the leaderboard but can no longer be tested or retrained.`)) return
    try {
      await api.deleteDataset(d.id)
      await refreshDatasets()
      if (draft.dataset === d.id) setDraft((x) => ({ ...x, dataset: 'mnist', blocks: [] }))
      toast('Dataset deleted', 'ok')
    } catch (e) { toast((e as Error).message, 'error') }
  }

  const mine = datasets.filter((d) => d.custom)
  const builtin = datasets.filter((d) => !d.custom)
  const step = coach != null ? COACH[coach] : null

  return (
    <div className="page">
      <div className="row wrap" style={{ marginBottom: 14, gap: 12 }}>
        <h2>Datasets</h2>
        <GuideButton lesson="app.datasets" />
        <span className="small muted">Bring your own data: a spreadsheet, a text file, folders of pictures, or a dataset from Hugging Face.</span>
      </div>
      <div className="datasets-layout">
        <div className="col" style={{ gap: 14 }}>
          <div className="panel">
            <div className="panel-head"><h3 className="grow">Your datasets</h3><span className="small muted">{mine.length}</span></div>
            <div className="panel-body col" style={{ gap: 8 }}>
              {mine.length === 0 && <div className="small muted">Nothing imported yet. Start with the worked example on the right.</div>}
              {mine.map((d) => (
                <div key={d.id} className="ds-card">
                  <div className="row"><b className="grow">{KIND_ICON[d.input_kind]} {d.name}</b><button className="sm primary" onClick={() => use(d)}>Build a model</button><button className="sm danger" onClick={() => remove(d)}>Delete</button></div>
                  <div className="small muted">{d.description}</div>
                  <div className="tiny faint">{fmtNum(d.train_size)} training · {fmtNum(d.test_size)} test · {d.task === 'lm' ? `${d.vocab_size} characters` : `${d.num_classes} classes`} · from {d.source}</div>
                </div>
              ))}
            </div>
          </div>
          <div className="panel">
            <div className="panel-head"><h3>Built in</h3></div>
            <div className="panel-body col" style={{ gap: 8 }}>
              {builtin.map((d) => {
                const locked = !!access.datasets && !access.datasets.has(d.id)
                return (
                  <div key={d.id} className="ds-card" style={locked ? { opacity: 0.55 } : undefined}>
                    <div className="row"><b className="grow">{locked ? '🔒' : KIND_ICON[d.input_kind]} {d.name}</b><span className="tiny muted">{kindLabel(d)}</span>{!locked && <button className="sm" onClick={() => use(d)}>Build a model</button>}</div>
                    <div className="small muted">{d.description}</div>
                    <Difficulty dataset={d} compact />
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <h3 className="grow">{pv ? `Import: ${pv.source}` : done ? 'Imported' : 'Import a dataset'}</h3>
            {(pv || done) && <button className="sm ghost" onClick={() => { setPv(null); setDone(null); setCoach(null); setErr(null) }}>Start over</button>}
          </div>
          <div className="panel-body col" style={{ gap: 14 }}>
            {busy && <div className="small muted"><span className="spinner" style={{ width: 10, height: 10 }} /> {busy}</div>}
            {err && <div className="error-box">{err}</div>}

            {!pv && !done && <>
              <div className="source-grid">
                <div className="template-card" onClick={startExample}>
                  <h3>🌸 Worked example</h3>
                  <div className="small muted">Import a small real table step by step, with each choice explained. Do this first: it takes a minute.</div>
                  <span className="pill accent" style={{ alignSelf: 'flex-start' }}>start here</span>
                </div>
                <div className="template-card" onClick={() => fileRef.current?.click()}>
                  <h3>📄 A file from your computer</h3>
                  <div className="small muted"><b>.csv</b>: a table with a header row. <b>.txt</b>: text for a model to learn to continue. <b>.zip</b>: pictures sorted into one folder per class.</div>
                </div>
              </div>
              <input ref={fileRef} type="file" accept=".csv,.tsv,.txt,.md,.zip" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) upload(f); e.target.value = '' }} />
              <div className="col" style={{ gap: 6 }}>
                <b className="small">🤗 From Hugging Face</b>
                <div className="row">
                  <input className="grow" placeholder="owner/name, for example scikit-learn/iris or ylecun/mnist" value={repo} onChange={(e) => setRepo(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && repo.trim() && fromHub()} />
                  <button disabled={!repo.trim() || !!busy} onClick={fromHub}>Fetch</button>
                </div>
                <div className="tiny faint">Thousands of public datasets live at huggingface.co/datasets. The first 20,000 rows are downloaded. Datasets with one text or image column and a label column work best.</div>
              </div>
              <div className="note-box">
                <b>What makes a dataset this app can learn from?</b>
                <ul style={{ margin: '6px 0 0', paddingLeft: 18, lineHeight: 1.6 }}>
                  <li>Many examples of the same kind: hundreds at least, thousands are better.</li>
                  <li>For each example, the <b>inputs</b> (numbers, a piece of text, or a picture) and the <b>answer</b> you want predicted, which must be a category such as “spam / not spam”.</li>
                  <li>Or just a long text, with no answers at all: the model then learns to predict the next character.</li>
                </ul>
              </div>
            </>}

            {step && (
              <div className="coach-box">
                <b>{step.title}</b>
                <div style={{ marginTop: 4 }}>{step.text}</div>
                <div className="row" style={{ marginTop: 8 }}>
                  {coach! > 0 && <button className="sm" onClick={() => setCoach(coach! - 1)}>← Back</button>}
                  {coach! < COACH.length - 1 && <button className="sm primary" onClick={() => setCoach(coach! + 1)}>Next →</button>}
                  <span className="spacer" />
                  <button className="sm ghost" onClick={() => setCoach(null)}>Hide the guide</button>
                </div>
              </div>
            )}

            {pv?.kind === 'text' && <>
              <div className="small">{fmtNum(pv.n_chars)} characters using {pv.vocab} different ones. The model will read a stretch of this text and learn to predict the next character, like Tiny Shakespeare.</div>
              <pre className="json" style={{ maxHeight: 180, whiteSpace: 'pre-wrap' }}>{pv.sample}…</pre>
            </>}

            {pv?.kind === 'table' && <>
              <div className="small muted">{fmtNum(pv.n_rows)} rows{pv.hub?.truncated ? ' (the first part of a larger dataset)' : ''}{pv.has_split ? ' · comes with its own test split' : ''}. Choose the answer column and the inputs:</div>
              <table className={`tbl small ${coach === 1 ? 'hl-answer' : coach === 2 ? 'hl-input' : ''}`}>
                <thead><tr><th>Column</th><th>Holds</th><th className="right">Different values</th><th title="The column the model learns to predict">Answer</th><th title="What the model gets to look at">Input</th></tr></thead>
                <tbody>
                  {pv.columns.map((c) => (
                    <tr key={c.name}>
                      <td><b>{c.name}</b></td>
                      <td className="muted">{c.type === 'number' ? 'numbers' : c.type === 'image' ? 'pictures' : c.avg_words >= 3 ? `text (about ${Math.round(c.avg_words)} words)` : 'short text'}{c.missing > 0 && ` · ${c.missing} missing`}</td>
                      <td className="num">{c.type === 'image' ? '-' : c.unique >= 5000 ? '5,000+' : c.unique}</td>
                      <td className="col-answer"><input type="radio" name="answer" disabled={c.type === 'image' || c.unique < 2 || c.unique > 200} checked={label === c.name} onChange={() => { setLabel(c.name); setInputs((i) => i.filter((x) => x !== c.name)) }} /></td>
                      <td className="col-input"><input type="checkbox" disabled={label === c.name} checked={inputs.includes(c.name)} onChange={(e) => setInputs((i) => (e.target.checked ? [...i, c.name] : i.filter((x) => x !== c.name)))} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ overflowX: 'auto' }}>
                <table className="tbl small">
                  <thead><tr>{pv.columns.map((c) => <th key={c.name} className={c.name === label ? 'ok' : ''}>{c.name}{c.name === label && ' (answer)'}</th>)}</tr></thead>
                  <tbody>
                    {pv.rows.map((r, i) => (
                      <tr key={i}>{r.map((v, j) => <td key={j} className={typeof v === 'string' && v.length > 40 ? 'small' : 'mono'}>{v && typeof v === 'object' ? <TensorImage data={v.image} shape={v.shape} size={40} /> : (v as string)}</td>)}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {problem && <div className="small warn">{problem}</div>}
            </>}

            {pv && (
              <div className={`hp-grid ${coach === 3 ? 'hl-box' : ''}`}>
                <label className="field"><span className="lbl">Name</span><input value={name} maxLength={50} onChange={(e) => setName(e.target.value)} /></label>
                {pv.kind === 'table' && !pv.has_split && (
                  <label className="field"><span className="lbl">Held back for testing</span>
                    <select value={testFrac} onChange={(e) => setTestFrac(Number(e.target.value))}>{[10, 15, 20, 30].map((p) => <option key={p} value={p}>{p}% of the rows</option>)}</select>
                  </label>
                )}
                {inKinds.has('image') && <>
                  <label className="field"><span className="lbl">Picture size</span>
                    <select value={size} onChange={(e) => setSize(Number(e.target.value))}>{[16, 28, 32, 48, 64].map((s) => <option key={s} value={s}>{s}×{s} pixels</option>)}</select>
                    <span className="help">Every picture is cropped to a square and shrunk to this size. Smaller trains much faster.</span>
                  </label>
                  <label className="field"><span className="lbl">Color</span>
                    <select value={gray ? 'g' : 'c'} onChange={(e) => setGray(e.target.value === 'g')}><option value="c">Color</option><option value="g">Grayscale</option></select>
                  </label>
                </>}
                {inKinds.has('text') && (
                  <label className="field"><span className="lbl">Longest text kept</span>
                    <select value={maxLen} onChange={(e) => setMaxLen(Number(e.target.value))}>{[16, 32, 64, 128, 256].map((s) => <option key={s} value={s}>{s} words</option>)}</select>
                    <span className="help">Longer texts are cut off here. The 20,000 most common words get their own number; the rest count as “unknown”.</span>
                  </label>
                )}
              </div>
            )}
            {pv && <div className="row"><span className="spacer" /><button className="primary big" disabled={!!busy || !!problem || !name.trim()} onClick={doImport}>Import “{name || '…'}”</button></div>}

            {done && (
              <div className="col" style={{ gap: 12 }}>
                <div className="note-box">
                  ✓ <b>{done.name}</b> is ready: {fmtNum(done.train_size)} training examples, {fmtNum(done.test_size)} held back for testing, {done.task === 'lm' ? `${done.vocab_size} different characters` : `${done.num_classes} classes (${done.class_names.slice(0, 6).join(', ')}${done.class_names.length > 6 ? ', …' : ''})`}.
                  <div className="small" style={{ marginTop: 4 }}>{done.difficulty_note}</div>
                </div>
                <div className="row wrap">
                  <button className="primary" onClick={() => use(done)}>Build a model for it →</button>
                  <button onClick={() => setDone(null)}>Import another</button>
                </div>
                <div className="coach-box">
                  <b>Now try one of your own</b>
                  <ul style={{ margin: '6px 0 0', paddingLeft: 18, lineHeight: 1.6 }}>
                    <li><b>A spreadsheet:</b> in Excel, Numbers or Google Sheets choose “Save as” or “Download” and pick CSV. Keep one header row. One column must be the category to predict.</li>
                    <li><b>Your own writing, a book, song lyrics:</b> save it as a .txt file. More text is better: a novel is a good size.</li>
                    <li><b>Photos:</b> make one folder per class (for example <code>cats</code>, <code>dogs</code>), put at least a few dozen pictures in each, and zip the folders together.</li>
                    <li><b>Something from Hugging Face:</b> search huggingface.co/datasets and paste the name.</li>
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
