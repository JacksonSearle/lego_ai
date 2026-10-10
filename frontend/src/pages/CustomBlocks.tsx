import { useEffect, useRef, useState } from 'react'
import { basicSetup, EditorView } from 'codemirror'
import { keymap } from '@codemirror/view'
import { indentWithTab } from '@codemirror/commands'
import { python } from '@codemirror/lang-python'
import { oneDark } from '@codemirror/theme-one-dark'
import { api } from '../api'
import { Brick } from '../components/Brick'
import { GuideButton } from '../learn/GuideButton'
import { useLearn } from '../learn/LearnContext'
import { useStore } from '../store'
import type { BlockSpec } from '../types'
import { defaultParams } from '../util'

const NEW_FILE = `import torch
import torch.nn as nn

from lego.sdk import Param, lego_block, make_activation


@lego_block(
    name="My Block",
    color="#22d3ee",
    description="What this block does.",
    params=[
        Param("width", "int", 128, "Width", "Number of output features.", min=1),
        Param("activation", "activation", "gelu", "Activation", "Non-linearity to apply."),
    ],
)
class MyBlock(nn.Module):
    def __init__(self, in_shape, width=128, activation="gelu"):
        super().__init__()
        self.fc = nn.Linear(in_shape[-1], width)
        self.act = make_activation(activation)

    def forward(self, x):
        return self.act(self.fc(x))
`

function Editor({ value, onChange }: { value: string; onChange: (s: string) => void }) {
  const host = useRef<HTMLDivElement>(null)
  const cb = useRef(onChange)
  cb.current = onChange
  useEffect(() => {
    const view = new EditorView({
      doc: value,
      parent: host.current!,
      extensions: [
        basicSetup,
        keymap.of([indentWithTab]),
        python(),
        oneDark,
        EditorView.updateListener.of((u) => u.docChanged && cb.current(u.state.doc.toString())),
        EditorView.theme({ '&': { height: '100%' }, '.cm-scroller': { fontFamily: 'var(--mono)' } }),
      ],
    })
    return () => view.destroy()
    // The editor is keyed by file; it owns its document after mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  return <div className="editor-host" ref={host} />
}

export function CustomBlocks() {
  const { refreshBlocks, toast, setTab } = useStore()
  const { flag, openLesson } = useLearn()
  const [files, setFiles] = useState<{ filename: string; source: string; blocks: string[] }[]>([])
  const [current, setCurrent] = useState<string | null>(null)
  const [filename, setFilename] = useState('')
  const [source, setSource] = useState('')
  const [dirty, setDirty] = useState(false)
  const [result, setResult] = useState<{ ok: boolean; error?: string; blocks?: BlockSpec[] } | null>(null)
  const [editorKey, setEditorKey] = useState(0)

  const load = async (select?: string) => {
    const r = await api.customBlocks()
    setFiles(r.files)
    const pick = r.files.find((f) => f.filename === (select ?? current)) ?? r.files[0]
    if (pick && (select || !current)) open(pick)
    return r.files
  }
  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const open = (f: { filename: string; source: string }) => {
    if (dirty && !confirm('Discard unsaved changes?')) return
    setCurrent(f.filename)
    setFilename(f.filename)
    setSource(f.source)
    setDirty(false)
    setResult(null)
    setEditorKey((k) => k + 1)
  }

  const newFile = () => {
    if (dirty && !confirm('Discard unsaved changes?')) return
    let n = 1
    while (files.some((f) => f.filename === `my_blocks${n > 1 ? n : ''}.py`)) n++
    setCurrent(null)
    setFilename(`my_blocks${n > 1 ? n : ''}.py`)
    setSource(NEW_FILE)
    flag('customNew')
    setDirty(true)
    setResult(null)
    setEditorKey((k) => k + 1)
  }

  const validate = async () => setResult(await api.validateCustom(filename, source))

  const save = async () => {
    try {
      const r = await api.saveCustom(filename, source)
      setResult(r)
      setDirty(false)
      setCurrent(filename.endsWith('.py') ? filename : `${filename}.py`)
      await refreshBlocks()
      await load(filename.endsWith('.py') ? filename : `${filename}.py`)
      flag('customSaved')
      toast(`Saved — ${r.blocks?.length} block(s) now in the builder palette`, 'ok')
    } catch (e) {
      setResult({ ok: false, error: (e as Error).message })
    }
  }

  const remove = async () => {
    if (!current || !confirm(`Delete ${current}? Its blocks will disappear from the palette.`)) return
    await api.deleteCustom(current)
    await refreshBlocks()
    setCurrent(null)
    setDirty(false)
    const fs = await load()
    if (!fs.length) { setFilename(''); setSource('') }
  }

  return (
    <div className="custom-layout">
      <aside className="file-list">
        <div className="row" style={{ marginBottom: 10 }}>
          <h4 className="grow">Files</h4>
          <button className="sm" onClick={newFile} data-tour="custom-new">+ New</button>
        </div>
        {files.map((f) => (
          <div key={f.filename} className={`file-item ${f.filename === current ? 'active' : ''}`} onClick={() => open(f)}>
            <span className="grow">{f.filename}</span>
            <span className="faint tiny">{f.blocks.length}</span>
          </div>
        ))}
        {current === null && filename && <div className="file-item active"><span className="grow">{filename}</span><span className="faint tiny">new</span></div>}
      </aside>

      <section className="editor-wrap">
        <div className="canvas-toolbar" style={{ position: 'static' }}>
          <input className="mono" value={filename} onChange={(e) => { setFilename(e.target.value); setDirty(true) }} style={{ width: 220 }} disabled={current !== null} title={current ? 'Saved files keep their name' : 'File name'} />
          {dirty && <span className="pill warn">unsaved</span>}
          <span className="spacer" />
          <button onClick={validate} disabled={!source}>Check</button>
          <button className="primary" onClick={save} disabled={!source || !filename} data-tour="custom-save">Save & load</button>
          <GuideButton lesson="app.custom" label="step-by-step" />
          {current && <button className="danger" onClick={remove}>Delete</button>}
        </div>
        {filename ? (
          <Editor key={editorKey} value={source} onChange={(s) => { setSource(s); setDirty(true) }} />
        ) : (
          <div className="empty" style={{ margin: 30 }}>No custom block files. <button onClick={newFile}>Create one</button></div>
        )}
      </section>

      <aside className="custom-side">
        {result && (result.ok ? (
          <div className="col">
            <h4>Blocks in this file</h4>
            {result.blocks?.map((b) => (
              <div key={b.type} style={{ paddingRight: 30 }}>
                <Brick spec={b} inst={{ id: b.type, type: b.type, params: defaultParams(b), repeat: 1 }} compact />
                <div className="tiny faint mono" style={{ marginTop: 4 }}>type: {b.type} · {b.params.length} params</div>
              </div>
            ))}
            {!dirty && <button onClick={() => setTab('build')}>Use them in the builder →</button>}
          </div>
        ) : (
          <div className="error-box">{result.error}</div>
        ))}

        <button onClick={() => openLesson('app.custom')}>📖 Walk me through it, line by line</button>
        <div>
          <h4>How custom blocks work</h4>
          <ol className="small muted" style={{ paddingLeft: 18, lineHeight: 1.65, margin: '8px 0 0' }}>
            <li>Write an <code>nn.Module</code> and decorate it with <code>@lego_block(...)</code>.</li>
            <li><code>__init__</code> receives <code>in_shape</code> first — the input shape <i>without</i> the batch dimension, e.g. <code>(64,)</code>, <code>(T, D)</code> or <code>(C, H, W)</code>.</li>
            <li>Each <code>Param</code> becomes a field in the block's side panel; its value is passed as a keyword argument.</li>
            <li>Add a <code>ctx=None</code> argument to see dataset facts: <code>ctx.num_classes</code>, <code>ctx.vocab_size</code>, <code>ctx.task</code>.</li>
            <li>Output shapes are inferred automatically by running a tiny dummy batch.</li>
            <li>Raise <code>ValueError("…")</code> for unsupported input shapes — the message is shown on the block.</li>
          </ol>
        </div>
        <div>
          <h4>Param types</h4>
          <pre className="json small" style={{ marginTop: 8 }}>{`Param(name, type, default, label, help,
      min=, max=, step=, choices=)

type: "int" | "float" | "bool"
    | "choice"      (needs choices=[...])
    | "activation"  (any registered activation)`}</pre>
        </div>
        <div className="note-box">
          Runs that use a custom block store its source code in their config, so they can always be rebuilt — even after you edit or delete the file.
        </div>
      </aside>
    </div>
  )
}
