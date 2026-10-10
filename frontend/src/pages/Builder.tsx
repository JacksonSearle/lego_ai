import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent } from 'react'
import { api, downloadJSON } from '../api'
import { MiniStack } from '../components/ArchView'
import { Brick } from '../components/Brick'
import { ParamField } from '../components/ParamField'
import { GuideButton } from '../learn/GuideButton'
import { blockAllowed, skillForDataset } from '../learn/engine'
import { useLearn } from '../learn/LearnContext'
import { lessonForBlock } from '../learn/lessons'
import { QuickTrain } from './QuickTrain'
import { newId, useStore } from '../store'
import type { BlockInst, BlockSpec, Template } from '../types'
import { cloneBlock, defaultParams, describeInput, describeOutput, fmtNum, fmtShape } from '../util'

type DragData = { kind: 'new'; type: string } | { kind: 'move'; id: string }
const MIME = 'application/x-lego'

export function Builder() {
  const { blocks: specs, meta, draft, setDraft, inspect, dataset, datasets, undo, redo, canUndo, canRedo, setTab, toast, loadConfig } = useStore()
  const { access, flag, openLesson } = useLearn()
  const [taskFilter, setTaskFilter] = useState<'all' | 'image' | 'text'>('all')
  const [selected, setSelected] = useState<string | null>(null)
  const [drag, setDrag] = useState<DragData | null>(null)
  const [insertAt, setInsertAt] = useState<number | null>(null)
  const [overTrash, setOverTrash] = useState(false)
  const [showTemplates, setShowTemplates] = useState(false)
  const [templates, setTemplates] = useState<Template[]>([])
  const stackRef = useRef<HTMLDivElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const blocks = draft.blocks
  const reports = useMemo(() => Object.fromEntries((inspect?.blocks ?? []).map((r) => [r.id, r])), [inspect])
  const sel = blocks.find((b) => b.id === selected) ?? null

  useEffect(() => {
    api.templates(draft.dataset).then(setTemplates)
  }, [draft.dataset])

  // Any change made here counts as the learner's own design.
  const setBlocks = useCallback((fn: (b: BlockInst[]) => BlockInst[]) => setDraft((d) => ({ ...d, edited: true, origin: d.origin ?? 'scratch', blocks: fn(d.blocks) })), [setDraft])

  const insert = useCallback(
    (type: string, at: number) => {
      const spec = specs[type]
      if (!spec) return
      const b: BlockInst = { id: newId(), type, params: defaultParams(spec), repeat: 1 }
      setBlocks((bs) => [...bs.slice(0, at), b, ...bs.slice(at)])
      setSelected(b.id)
    },
    [specs, setBlocks],
  )

  /** Where a click-to-add lands: above the selected block, else just below the output head, else on top. */
  const defaultInsertIndex = () => {
    if (sel) return blocks.indexOf(sel) + 1
    const head = blocks.findIndex((b) => b.type === 'classifier')
    return head >= 0 ? head : blocks.length
  }

  const remove = useCallback(
    (id: string) => {
      setBlocks((bs) => bs.filter((b) => b.id !== id))
      flag('removeBlock')
      setSelected((s) => (s === id ? null : s))
    },
    [setBlocks, flag],
  )

  const duplicate = useCallback(
    (id: string) => {
      const i = blocks.findIndex((b) => b.id === id)
      if (i < 0) return
      const copy = cloneBlock(blocks[i], newId())
      setBlocks((bs) => [...bs.slice(0, i + 1), copy, ...bs.slice(i + 1)])
      setSelected(copy.id)
    },
    [blocks, setBlocks],
  )

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      if (t?.closest?.('input, textarea, select, .cm-editor')) return
      const mod = e.metaKey || e.ctrlKey
      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        if (e.shiftKey) redo()
        else undo()
      } else if (mod && e.key.toLowerCase() === 'y') {
        e.preventDefault()
        redo()
      } else if (mod && e.key.toLowerCase() === 'd' && selected) {
        e.preventDefault()
        duplicate(selected)
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && selected) {
        e.preventDefault()
        remove(selected)
      } else if (e.key === 'Escape') setSelected(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selected, undo, redo, duplicate, remove])

  // ---- drag & drop ----
  const startDrag = (e: DragEvent, data: DragData) => {
    e.dataTransfer.setData(MIME, JSON.stringify(data))
    e.dataTransfer.effectAllowed = data.kind === 'new' ? 'copy' : 'move'
    // Defer so the browser snapshots the element before it's restyled.
    setTimeout(() => setDrag(data), 0)
  }
  const endDrag = () => {
    setDrag(null)
    setInsertAt(null)
    setOverTrash(false)
  }

  const computeInsert = (clientY: number) => {
    const els = stackRef.current?.querySelectorAll<HTMLElement>('[data-index]') ?? []
    let below = 0
    els.forEach((el) => {
      const r = el.getBoundingClientRect()
      if (r.top + r.height / 2 > clientY) below += 1
    })
    return below
  }

  const onStackDragOver = (e: DragEvent) => {
    if (!e.dataTransfer.types.includes(MIME)) return
    e.preventDefault()
    e.dataTransfer.dropEffect = drag?.kind === 'move' ? 'move' : 'copy'
    const at = computeInsert(e.clientY)
    if (at !== insertAt) setInsertAt(at)
  }

  const onStackDrop = (e: DragEvent) => {
    e.preventDefault()
    const raw = e.dataTransfer.getData(MIME)
    const at = insertAt ?? computeInsert(e.clientY)
    endDrag()
    if (!raw) return
    const data = JSON.parse(raw) as DragData
    if (data.kind === 'new') {
      insert(data.type, at)
      flag('dragBlock')
    }
    else {
      setBlocks((bs) => {
        const from = bs.findIndex((b) => b.id === data.id)
        if (from < 0) return bs
        const to = from < at ? at - 1 : at
        if (to === from) return bs
        const next = [...bs]
        const [m] = next.splice(from, 1)
        next.splice(to, 0, m)
        return next
      })
      setSelected(data.id)
      flag('moveBlock')
    }
  }

  const useTemplate = (t: Template) => {
    setDraft((d) => ({
      ...d,
      name: t.name,
      blocks: t.blocks.map((b) => ({ ...b, id: newId() })),
      training: { ...(meta?.defaults ?? {}), ...d.training, ...t.training },
      custom_blocks: undefined,
      origin: 'template',
      edited: false,
    }))
    flag('template')
    setSelected(null)
    setShowTemplates(false)
    toast(`Loaded “${t.name}” — training settings updated too. Undo with ⌘Z.`, 'ok')
  }

  const importFile = async (f: File) => {
    try {
      const json = JSON.parse(await f.text())
      const res = await api.importConfig(json)
      loadConfig(res.config)
      flag('import')
      res.notes.forEach((n) => toast(n))
      if (res.env_diffs.length) toast(`Environment differs from the original run: ${res.env_diffs.join('; ')}`, 'error')
      toast(`Imported “${res.config.name}”`, 'ok')
    } catch (err) {
      toast(`Import failed: ${(err as Error).message}`, 'error')
    }
  }

  // ---- palette ----
  const unlockedSpecs = useMemo(() => Object.values(specs).filter((s) => blockAllowed(access, s.type)), [specs, access])
  const lockedCount = Object.values(specs).filter((s) => s.source === 'builtin').length - unlockedSpecs.filter((s) => s.source === 'builtin').length
  const hasTasks = unlockedSpecs.some((s) => s.tags?.includes('image')) && unlockedSpecs.some((s) => s.tags?.includes('text'))
  const grouped = useMemo(() => {
    const cats = meta?.categories ?? []
    const by: Record<string, BlockSpec[]> = {}
    unlockedSpecs
      .filter((s) => taskFilter === 'all' || s.tags?.includes(taskFilter) || s.tags?.includes('general'))
      .forEach((s) => (by[s.category] ??= []).push(s))
    const order = [...cats, ...Object.keys(by).filter((c) => !cats.includes(c))]
    return order.filter((c) => by[c]?.length).map((c) => [c, by[c]] as const)
  }, [unlockedSpecs, meta, taskFilter])

  const ok = inspect?.ok && blocks.length > 0
  const outShape = inspect?.output_shape
  const outOk = !!outShape && JSON.stringify(outShape) === JSON.stringify(inspect?.expected_output)

  const shapeAt = (gap: number) => {
    if (gap === 0) return inspect?.input_shape
    return reports[blocks[gap - 1]?.id]?.out_shape
  }

  const Flow = ({ gap }: { gap: number }) => {
    const hot = drag != null && insertAt === gap
    const shape = shapeAt(gap)
    return (
      <div className={`flow ${hot ? 'hot' : ''}`}>
        <div className="arrow" />
        {shape && !hot && <span className="shape">{fmtShape(shape)}</span>}
        {hot && <div className="insert-line" />}
      </div>
    )
  }

  return (
    <div className="builder">
      {/* Palette */}
      <aside className="palette" data-tour="palette">
        <div className="row" style={{ marginBottom: 8 }}>
          <span className="small muted grow">Drag blocks onto the stack, or click to add.</span>
          <GuideButton lesson="app.builder" />
        </div>
        {(access.unlocked || hasTasks) && (
          <div className="seg" style={{ width: '100%', marginBottom: 12 }}>
            {(['all', 'image', 'text'] as const).map((k) => (
              <button key={k} className={taskFilter === k ? 'on' : ''} onClick={() => setTaskFilter(k)} style={{ flex: 1 }}>
                {k === 'all' ? 'All' : k === 'image' ? '🖼️ Image' : '📝 Text'}
              </button>
            ))}
          </div>
        )}
        {grouped.map(([cat, list]) => (
          <div key={cat} className="palette-cat">
            <h4>{cat}</h4>
            {list.map((s) => (
              <div
                key={s.type}
                className="palette-item"
                draggable
                onDragStart={(e) => startDrag(e, { kind: 'new', type: s.type })}
                onDragEnd={endDrag}
                onClick={() => insert(s.type, defaultInsertIndex())}
                title={s.description}
              >
                <span className="palette-swatch" style={{ background: s.color, ['--sw' as string]: s.color }} />
                <span className="grow">{s.name}</span>
                {s.source === 'custom' && <span className="pill" style={{ fontSize: 10 }}>py</span>}
                {s.source !== 'custom' && (
                  <button className="ghost sm lesson-link" title={`Lesson: ${s.name}`} onClick={(e) => { e.stopPropagation(); openLesson(lessonForBlock(s.type)) }}>📖</button>
                )}
              </div>
            ))}
          </div>
        ))}
        {lockedCount > 0 && (
          <div className="locked-hint" onClick={() => { sessionStorage.setItem('lego-ai/guide-view', 'skills'); setTab('guide') }}>
            🔒 {lockedCount} more blocks {access.has('skilltree') ? 'in the Skill Tree' : 'unlock as you level up'}
          </div>
        )}
        {access.has('custom.page') && (
          <button className="sm" style={{ width: '100%', justifyContent: 'center' }} onClick={() => setTab('custom')}>
            + Define a block in Python
          </button>
        )}
      </aside>

      {/* Canvas */}
      <section className="canvas" onClick={(e) => e.target === e.currentTarget && setSelected(null)}>
        <div className="canvas-toolbar">
          <input className="name" value={draft.name} onChange={(e) => setDraft((d) => ({ ...d, name: e.target.value }), { undoable: false })} title="Model name" />
          {access.has('datasets') ? (
            <select value={draft.dataset} onChange={(e) => { setDraft((d) => ({ ...d, dataset: e.target.value })); flag('datasetChanged') }} title="Dataset" data-tour="dataset">
              {datasets.map((d) => {
                const ok = !access.datasets || access.datasets.has(d.id)
                return <option key={d.id} value={d.id} disabled={!ok}>{ok ? d.name : `🔒 ${d.name} (${skillForDataset(d.id)?.title ?? 'skill'})`}</option>
              })}
            </select>
          ) : <span className="pill" title="More datasets unlock at level 6">🔢 MNIST</span>}
          {access.has('builder.templates') && <button onClick={() => setShowTemplates(true)} data-tour="templates">Templates</button>}
          <button className="ghost" onClick={() => { setDraft((d) => ({ ...d, blocks: [], custom_blocks: undefined })); setSelected(null) }} disabled={!blocks.length}>
            Clear
          </button>
          <button className="ghost" onClick={undo} disabled={!canUndo} title="Undo (⌘Z)">↶</button>
          <button className="ghost" onClick={redo} disabled={!canRedo} title="Redo (⇧⌘Z)">↷</button>
          <span className="spacer" />
          {access.has('config.io') && <>
          <button className="ghost" onClick={() => fileRef.current?.click()} title="Load a .lego.json run config">Import</button>
          <button className="ghost" data-tour="export" onClick={() => { flag('export'); downloadJSON({ format: 'lego-ai.run', version: 1, name: draft.name, dataset: { id: draft.dataset, train_subset: draft.train_subset }, architecture: { blocks }, training: draft.training }, `${draft.name.replace(/\W+/g, '_')}.lego.json`) }} disabled={!blocks.length}>
            Export
          </button>
          </>}
          <input ref={fileRef} type="file" accept=".json" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) importFile(f); e.target.value = '' }} />
          {blocks.length > 0 && (
            <span className={`pill ${ok ? 'ok' : 'bad'}`} title={inspect?.error ?? ''}>
              {ok ? `✓ valid · ${fmtNum(inspect?.total_params)} params` : '✗ needs fixing'}
            </span>
          )}
          {access.has('train.page') && <button className="primary" disabled={!ok} onClick={() => setTab('train')}>Train →</button>}
        </div>

        {blocks.length === 0 && !drag ? (
          <EmptyState templates={access.has('builder.templates') ? templates : []} specs={specs} onUse={useTemplate} onDragOver={onStackDragOver} onDrop={onStackDrop} dataset={dataset?.name} />
        ) : (
          <div className="stack-wrap" onClick={(e) => e.target === e.currentTarget && setSelected(null)}>
            <div className="stack" data-tour="stack" ref={stackRef} onDragOver={onStackDragOver} onDrop={onStackDrop} onDragLeave={(e) => { if (!stackRef.current?.contains(e.relatedTarget as Node)) setInsertAt(null) }}>
              <div className={`io-node output ${blocks.length && inspect ? (outOk ? 'ok' : 'bad') : ''}`}>
                <div className="io-label">OUTPUT · {describeOutput(dataset)}</div>
                <div className="io-shape">{fmtShape(inspect?.expected_output)}</div>
                {blocks.length > 0 && !outOk && inspect && !inspect.blocks.some((b) => b.error) && (
                  <div className="small bad" style={{ marginTop: 4 }}>{inspect.error}</div>
                )}
              </div>
              <Flow gap={blocks.length} />
              {[...blocks].reverse().map((b) => {
                const i = blocks.indexOf(b)
                return (
                  <div key={b.id} data-index={i}>
                    <Brick
                      spec={specs[b.type]}
                      inst={b}
                      report={reports[b.id]}
                      selected={selected === b.id}
                      dragging={drag?.kind === 'move' && drag.id === b.id}
                      onClick={(e) => { e.stopPropagation(); setSelected(b.id) }}
                      onDragStart={(e) => startDrag(e, { kind: 'move', id: b.id })}
                      onDragEnd={endDrag}
                    />
                    <Flow gap={i} />
                  </div>
                )
              })}
              {blocks.length === 0 && <div className={`drop-hint ${insertAt != null ? 'hot' : ''}`}>Drop here</div>}
              <div className="io-node">
                <div className="io-label">INPUT · {dataset?.name} · {describeInput(dataset)}</div>
                <div className="io-shape">{fmtShape(inspect?.input_shape ?? dataset?.input_shape)}</div>
              </div>
            </div>
          </div>
        )}

        {drag?.kind === 'move' && (
          <div
            className={`trash ${overTrash ? 'hot' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setOverTrash(true) }}
            onDragLeave={() => setOverTrash(false)}
            onDrop={(e) => { e.preventDefault(); remove(drag.id); endDrag() }}
          >
            Drop here to delete
          </div>
        )}
      </section>

      {/* Inspector */}
      <aside className="inspector">
        {sel ? (
          <BlockInspector
            key={sel.id}
            inst={sel}
            spec={specs[sel.type]}
            report={reports[sel.id]}
            onChange={(fn) => setBlocks((bs) => bs.map((b) => {
              if (b.id !== sel.id) return b
              const nb = fn(b)
              if (nb.repeat >= 3) flag('repeat3')
              return nb
            }))}
            onDelete={() => remove(sel.id)}
            onDuplicate={() => duplicate(sel.id)}
          />
        ) : (
          <>{!access.has('train.page') && <QuickTrain valid={!!ok} />}<ModelSummary /></>
        )}
      </aside>

      {showTemplates && (
        <div className="modal-back" onClick={() => setShowTemplates(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h2>Starting points for {dataset?.name}</h2>
              <span className="spacer" />
              <button className="ghost" onClick={() => setShowTemplates(false)}>✕</button>
            </div>
            <div className="modal-body">
              <TemplateGrid templates={templates} specs={specs} onUse={useTemplate} />
              <div className="row" style={{ marginTop: 16 }}>
                <button onClick={() => { setDraft((d) => ({ ...d, blocks: [], name: 'My model' })); setShowTemplates(false) }}>Start from scratch</button>
                <span className="small muted">An empty stack — drag blocks in from the palette.</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function TemplateGrid({ templates, specs, onUse }: { templates: Template[]; specs: Record<string, BlockSpec>; onUse: (t: Template) => void }) {
  return (
    <div className="template-grid">
      {[...templates].sort((a, b) => Number(b.recommended) - Number(a.recommended)).map((t) => (
        <div key={t.id} className="template-card" onClick={() => onUse(t)}>
          <div className="row">
            <h3 className="grow">{t.name}</h3>
            {t.recommended && <span className="pill accent">recommended</span>}
          </div>
          <div className="row" style={{ alignItems: 'flex-end', gap: 14 }}>
            <MiniStack blocks={t.blocks} specs={specs} width={110} />
            <div className="small muted grow" style={{ paddingLeft: 18 }}>{t.description}</div>
          </div>
          <div className="tiny faint mono">
            {Object.entries(t.training).filter(([k]) => ['epochs', 'lr', 'scheduler', 'device'].includes(k)).map(([k, v]) => `${k}=${v}`).join('  ')}
          </div>
        </div>
      ))}
    </div>
  )
}

function EmptyState({ templates, specs, onUse, onDragOver, onDrop, dataset }: {
  templates: Template[]; specs: Record<string, BlockSpec>; onUse: (t: Template) => void
  onDragOver: (e: DragEvent) => void; onDrop: (e: DragEvent) => void; dataset?: string
}) {
  return (
    <div style={{ padding: '28px 32px', maxWidth: 980, margin: '0 auto' }} onDragOver={onDragOver} onDrop={onDrop} data-tour="stack">
      {templates.length ? <>
        <h2>Start with a proven architecture for {dataset}</h2>
        <p className="muted" style={{ marginTop: 4 }}>Pick a template to customise, or drag blocks from the left to build from scratch.</p>
      </> : <>
        <h2>Build your model</h2>
        <p className="muted" style={{ marginTop: 4 }}>Drag blocks from the panel on the left into this space. Data flows <b>up</b>: a simple digit reader is <b>Flatten</b> at the bottom, then <b>Linear</b>, then the <b>Output Head</b> on top.</p>
      </>}
      <div style={{ marginTop: 16 }}>
        <TemplateGrid templates={templates} specs={specs} onUse={onUse} />
      </div>
    </div>
  )
}

function BlockInspector({ inst, spec, report, onChange, onDelete, onDuplicate }: {
  inst: BlockInst
  spec: BlockSpec | undefined
  report: { in_shape: number[] | null; out_shape: number[] | null; params: number; error: string | null } | undefined
  onChange: (fn: (b: BlockInst) => BlockInst) => void
  onDelete: () => void
  onDuplicate: () => void
}) {
  const { access, openLesson } = useLearn()
  if (!spec) {
    return (
      <div className="inspector-body">
        <div className="error-box">Block type “{inst.type}” is not available. If it was a custom block, re-create it on the Custom Blocks tab.</div>
        <button className="danger" onClick={onDelete}>Delete block</button>
      </div>
    )
  }
  const setParam = (k: string, v: unknown) => onChange((b) => ({ ...b, params: { ...b.params, [k]: v } }))
  const setRepeat = (n: number) => onChange((b) => ({ ...b, repeat: Math.max(1, Math.min(64, n || 1)) }))
  const actsUnlocked = blockAllowed(access, 'activation')
  return (
    <>
      <div className="inspector-head">
        <div className="color-bar" style={{ background: spec.color }} />
        <div className="row">
          <h2 className="grow">{spec.name}</h2>
          <span className="pill">{spec.category}</span>
        </div>
        <p className="small muted" style={{ margin: '8px 0 0' }}>{spec.description}</p>
        {spec.paper && <a className="small" href={spec.paper} target="_blank" rel="noreferrer">Original paper ↗</a>}
      </div>
      <div className="inspector-body">
        <div className="stat-grid">
          <div className="stat"><div className="k">In</div><div className="v">{fmtShape(report?.in_shape)}</div></div>
          <div className="stat"><div className="k">Out</div><div className="v">{fmtShape(report?.out_shape)}</div></div>
          <div className="stat"><div className="k">Params</div><div className="v">{fmtNum(report?.params)}</div></div>
        </div>
        {report?.error && <div className="error-box">{report.error}</div>}

        {spec.source === 'builtin' && <button className="sm" onClick={() => openLesson(lessonForBlock(spec.type))}>📖 Learn how {spec.name} works</button>}
        {access.has('builder.repeat') && <label className="field">
          <span className="lbl">Repeat</span>
          <div className="row">
            <div className="stepper">
              <button onClick={() => setRepeat(inst.repeat - 1)} disabled={inst.repeat <= 1}>−</button>
              <input type="number" value={inst.repeat} min={1} max={64} onChange={(e) => setRepeat(parseInt(e.target.value, 10))} />
              <button onClick={() => setRepeat(inst.repeat + 1)}>+</button>
            </div>
          </div>
          <span className="help">Stack this block N times in a row (each copy has its own weights). Input size is inferred for each copy.</span>
        </label>}

        {spec.params.length > 0 && <h4>Parameters</h4>}
        {spec.params.map((p) => (
          <div key={p.name}>
            <ParamField spec={p} value={inst.params[p.name]} onChange={(v) => setParam(p.name, v)} disabled={p.type === 'activation' && !actsUnlocked} />
            {p.type === 'activation' && !actsUnlocked && <span className="tiny faint">🔒 Choosing activations unlocks at level 5</span>}
          </div>
        ))}
        {spec.params.length === 0 && <div className="small muted">This block has no parameters — its sizes are inferred from its input.</div>}

        <div className="row" style={{ marginTop: 6 }}>
          <button onClick={onDuplicate} title="⌘D">Duplicate</button>
          <button onClick={() => onChange((b) => ({ ...b, params: defaultParams(spec) }))}>Reset</button>
          <span className="spacer" />
          <button className="danger" onClick={onDelete} title="Delete / Backspace">Delete</button>
        </div>
      </div>
    </>
  )
}

function ModelSummary() {
  const { draft, inspect, dataset, blocks: specs } = useStore()
  const counts = useMemo(() => {
    const c: Record<string, number> = {}
    draft.blocks.forEach((b) => (c[b.type] = (c[b.type] ?? 0) + b.repeat))
    return c
  }, [draft.blocks])
  return (
    <>
      <div className="inspector-head">
        <h4>Dataset</h4>
        <h2 style={{ marginTop: 6 }}>{dataset?.name}</h2>
        <p className="small muted" style={{ margin: '6px 0 0' }}>{dataset?.description}</p>
      </div>
      <div className="inspector-body">
        <div className="stat-grid">
          <div className="stat"><div className="k">Input</div><div className="v">{fmtShape(dataset?.input_shape)}</div></div>
          <div className="stat"><div className="k">Output</div><div className="v">{fmtShape(inspect?.expected_output)}</div></div>
          <div className="stat"><div className="k">Train</div><div className="v">{fmtNum(dataset?.train_size)}</div></div>
        </div>
        <h4>Model</h4>
        <div className="stat-grid">
          <div className="stat"><div className="k">Blocks</div><div className="v">{draft.blocks.reduce((a, b) => a + b.repeat, 0)}</div></div>
          <div className="stat"><div className="k">Params</div><div className="v">{fmtNum(inspect?.total_params)}</div></div>
          <div className="stat"><div className="k">Status</div><div className={`v ${inspect?.ok ? 'ok' : 'bad'}`}>{draft.blocks.length ? (inspect?.ok ? 'valid' : 'error') : '—'}</div></div>
        </div>
        {inspect?.error && draft.blocks.length > 0 && <div className="error-box">{inspect.error}</div>}
        {Object.keys(counts).length > 0 && (
          <div className="col" style={{ gap: 4 }}>
            {Object.entries(counts).map(([t, n]) => (
              <div key={t} className="row small">
                <span className="chip" style={{ background: specs[t]?.color }} />
                <span className="grow">{specs[t]?.name ?? t}</span>
                <span className="mono muted">×{n}</span>
              </div>
            ))}
          </div>
        )}
        <h4>Tips</h4>
        <ul className="small muted" style={{ margin: 0, paddingLeft: 18, lineHeight: 1.7 }}>
          <li>Click a block to edit its parameters here.</li>
          <li>Data flows <b>upward</b>: the shape after each block is shown on the arrows.</li>
          <li>Input sizes are inferred automatically — you only choose output sizes.</li>
          <li>End with an <b>Output Head</b>; it sizes itself to the dataset.</li>
          <li><span className="kbd">⌫</span> delete · <span className="kbd">⌘D</span> duplicate · <span className="kbd">⌘Z</span> undo</li>
        </ul>
      </div>
    </>
  )
}
