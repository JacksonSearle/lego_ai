import { Fragment, useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import { NumInput } from '../components/NumInput'
import { GuideButton } from '../learn/GuideButton'
import { newId, useStore } from '../store'
import type { LabBench, LabInfo, LabPipeline, RunRow } from '../types'
import { fmtAgo, fmtDur, fmtNum, fmtPct } from '../util'

const STAGE_COLOR: Record<string, string> = { pretrain: '#60a5fa', rl: '#f59e0b', safety: '#34d399' }
const STAGE_LABEL: Record<string, string> = { pretrain: 'Pretrained', rl: 'After RL', safety: 'Safety-tuned' }

/** The pipeline as a row of stations, showing which one the model is in right now. */
function Stations({ info, pipe, live }: { info: LabInfo; pipe: LabPipeline | null; live: boolean }) {
  const stages = pipe?.stages ?? []
  const state = (id: string) => stages.find((s) => s.stage === id)?.status ?? (pipe ? 'skipped' : 'idle')
  const allDone = !!pipe && stages.every((s) => s.status === 'done')
  const benching = live && stages.some((s) => s.status === 'running')
  const cards = [
    { id: 'data', icon: '🌐', name: 'The toy internet', what: `Stories, facts, sums and puzzles about ${info.world.places} invented places. Like the real one, it repeats popular mistakes and leaks secrets.`, st: pipe ? 'done' : 'idle' },
    ...info.stages.map((s) => ({ ...s, st: state(s.id) })),
    { id: 'bench', icon: '📊', name: 'Benchmarks', what: 'After every stage the model sits the same seven tests, so you can see what each stage bought and what it cost.', st: allDone ? 'done' : benching ? 'running' : pipe ? 'waiting' : 'idle' },
  ]
  return (
    <div className="stations">
      {cards.map((c, i) => (
        <Fragment key={c.id}>
          {i > 0 && <div className={`station-arrow ${cards[i - 1].st === 'done' ? 'lit' : ''}`}>→</div>}
          <div className={`station ${c.st}`} style={{ ['--sc' as string]: STAGE_COLOR[c.id] ?? '#94a3b8' }}>
            <div className="row" style={{ gap: 6 }}>
              <span style={{ fontSize: 20 }}>{c.icon}</span>
              <b className="grow">{c.name}</b>
              {c.st === 'running' && c.id !== 'bench' && <span className="pill accent"><span className="spinner" style={{ width: 9, height: 9 }} /> you are here</span>}
              {c.st === 'done' && <span className="ok">✓</span>}
              {c.st === 'skipped' && <span className="pill">skipped</span>}
              {c.st === 'stopped' && <span className="pill warn">stopped</span>}
            </div>
            <div className="tiny muted" style={{ marginTop: 4 }}>{c.what}</div>
          </div>
        </Fragment>
      ))}
    </div>
  )
}

/** One benchmark: a bar per finished stage. */
function BenchRow({ b, stages }: { b: LabInfo['benchmarks'][number]; stages: { stage: string; bench: LabBench }[] }) {
  return (
    <div className="bench-row" title={b.what}>
      <div>
        <div style={{ fontWeight: 600 }}>{b.name}</div>
        <div className="tiny faint">like {b.like}</div>
      </div>
      <div className="col" style={{ gap: 3 }}>
        {stages.map((s, i) => {
          const v = s.bench.scores[b.id] ?? 0
          const prev = i > 0 ? stages[i - 1].bench.scores[b.id] ?? 0 : null
          const d = prev == null ? 0 : v - prev
          return (
            <div key={s.stage} className="row" style={{ gap: 8 }}>
              <div className="bench-bar"><div style={{ width: `${v * 100}%`, background: STAGE_COLOR[s.stage] }} /></div>
              <span className="mono small" style={{ width: 44, textAlign: 'right' }}>{fmtPct(v, 0)}</span>
              <span className={`mono tiny ${d > 0.02 ? 'ok' : d < -0.02 ? 'bad' : 'faint'}`} style={{ width: 40 }}>{prev == null ? '' : `${d >= 0 ? '+' : ''}${Math.round(d * 100)}`}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/** A few sentences on what this pipeline shows, written from its own numbers. */
function takeaways(stages: { stage: string; bench: LabBench }[]): string[] {
  const out: string[] = []
  const get = (id: string) => stages.find((s) => s.stage === id)?.bench
  const pre = get('pretrain'), rl = get('rl'), safe = get('safety')
  const pct = (v: number) => `${Math.round(v * 100)}%`
  if (pre) out.push(`Pretraining alone gave a model that knows the world (${pct(pre.scores.knowledge)} on Knowledge) but repeats the internet’s mistakes (${pct(pre.scores.math)} on Math) and gives away secrets (it refuses ${pct(pre.scores.refusal)} of requests for a vault code).`)
  if (pre && rl) {
    const dm = rl.scores.math - pre.scores.math, dp = rl.scores.puzzles - pre.scores.puzzles
    out.push(dm > 0.02 || dp > 0.02
      ? `Reinforcement learning moved Math from ${pct(pre.scores.math)} to ${pct(rl.scores.math)} and Puzzles from ${pct(pre.scores.puzzles)} to ${pct(rl.scores.puzzles)}, with nobody showing it the right answers: only a checker saying right or wrong. Knowledge went from ${pct(pre.scores.knowledge)} to ${pct(rl.scores.knowledge)}, because the leash kept it close to the pretrained model.`
      : `Reinforcement learning did not move Math or Puzzles much this time (${pct(pre.scores.math)} to ${pct(rl.scores.math)}, ${pct(pre.scores.puzzles)} to ${pct(rl.scores.puzzles)}). Try more rounds.`)
  }
  const before = rl ?? pre
  if (before && safe) {
    out.push(`Safety fine-tuning took refusals of plain requests from ${pct(before.scores.refusal)} to ${pct(safe.scores.refusal)}, and of disguised or reworded ones from ${pct(before.scores.jailbreak)} to ${pct(safe.scores.jailbreak)}${safe.scores.jailbreak < safe.scores.refusal - 0.05 ? ': the gap between those two is the jailbreak problem' : ''}. It still answers ${pct(safe.scores.helpful)} of harmless look-alike questions.`)
    const cost = before.capability - safe.capability
    out.push(cost > 0.015 ? `The price: average capability fell from ${pct(before.capability)} to ${pct(safe.capability)}. Labs call this the safety tax and work hard to shrink it.` : `Capability barely moved (${pct(before.capability)} to ${pct(safe.capability)}): the ordinary questions mixed into the safety data did their job.`)
  }
  return out
}

export function Frontier() {
  const { draft, setDraft, inspect, meta, session, attachJob, toConfig, toast, setTab } = useStore()
  const [info, setInfo] = useState<LabInfo | null>(null)
  const [pipes, setPipes] = useState<LabPipeline[]>([])
  const [runs, setRuns] = useState<RunRow[]>([])
  const [picked, setPicked] = useState<string | null>(null)
  const [opt, setOpt] = useState<Record<string, number | boolean>>({})
  const [from, setFrom] = useState('')
  const [device, setDevice] = useState('cpu')
  const [busy, setBusy] = useState(false)
  const [showWorld, setShowWorld] = useState(false)
  const [failed, setFailed] = useState<string | null>(null)

  const live = session?.kind === 'lab' ? session : null
  const load = useCallback(() => {
    api.labPipelines().then(setPipes).catch(() => {})
    api.runs().then(setRuns).catch(() => {})
  }, [])
  useEffect(() => {
    api.labInfo().then((i) => { setInfo(i); setOpt(i.defaults) }).catch((e) => setFailed((e as Error).message))
  }, [])
  useEffect(() => { load() }, [load, live?.status, live?.pipeline?.stages.filter((s) => s.status === 'done').length])
  useEffect(() => { if (live?.pipeline) setPicked(live.pipeline.id) }, [live?.pipeline?.id])
  useEffect(() => {
    const devs = meta?.environment.available_devices ?? ['cpu']
    setDevice(devs.includes('cuda') ? 'cuda' : devs.includes('mps') ? 'mps' : 'cpu')
  }, [meta])

  // The live pipeline's record is fresher than the stored one.
  const all = useMemo(() => (live?.pipeline ? [live.pipeline, ...pipes.filter((p) => p.id !== live.pipeline!.id)] : pipes), [pipes, live?.pipeline])
  const pipe = all.find((p) => p.id === picked) ?? all[0] ?? null
  const isLive = !!live && live.status === 'running' && pipe?.id === live.pipeline?.id
  const done = (pipe?.stages ?? []).filter((s) => s.bench).map((s) => ({ stage: s.stage, bench: s.bench! }))

  const onLab = draft.dataset === 'lab_world'
  const valid = onLab && !!inspect?.ok && draft.blocks.length > 0
  const foundations = runs.filter((r) => r.dataset === 'lab_world' && r.status === 'completed' && !r.init_from && r.has_checkpoint)
  const running = session?.status === 'running'

  const starter = async () => {
    const t = (await api.templates('lab_world')).find((x) => x.recommended)
    if (!t) return
    setDraft((d) => ({ ...d, name: 'My assistant', dataset: 'lab_world', seq_len: null, train_subset: null, blocks: t.blocks.map((b) => ({ ...b, id: newId() })), training: { ...(meta?.defaults ?? {}), ...t.training }, custom_blocks: undefined, init_from: undefined, origin: 'template', edited: false }))
    toast('Loaded the lab’s starter model: a small GPT. You can change it in the Builder.', 'ok')
  }
  const start = async () => {
    setBusy(true)
    try {
      const cfg = toConfig()
      cfg.training = { ...cfg.training, device }
      const { job } = await api.labStart(cfg, { ...opt, start_from: from || undefined })
      attachJob(job.id, 'lab', draft.name)
      toast('The pipeline has started. Watch the stations light up here, or the live charts on the Train tab.', 'ok')
    } catch (e) { toast((e as Error).message, 'error') } finally { setBusy(false) }
  }
  const set = (k: string, v: number | boolean) => setOpt((o) => ({ ...o, [k]: v }))

  if (failed) return <div className="page"><div className="error-box">Could not open the Frontier Lab: {failed}</div></div>
  if (!info) return <div className="page"><span className="spinner" /></div>
  const probes = done[0]?.bench.samples ?? []
  return (
    <div className="page">
      <div className="row wrap" style={{ marginBottom: 12, gap: 12 }}>
        <h2>🏭 Frontier Lab</h2>
        <GuideButton lesson="adv.frontier" />
        <span className="small muted">How a lab turns a pile of text into an assistant, at toy scale: pretrain, improve with reinforcement learning, teach it to refuse, and measure after every step.</span>
      </div>

      <Stations info={info} pipe={pipe} live={isLive} />
      {isLive && <div className="small muted" style={{ margin: '8px 2px' }}><span className="spinner" style={{ width: 10, height: 10 }} /> {live!.message} · <a style={{ cursor: 'pointer' }} onClick={() => setTab('train')}>watch the live training charts</a></div>}

      <div className="frontier-layout">
        <div className="panel">
          <div className="panel-head"><h3 className="grow">Run a pipeline</h3></div>
          <div className="panel-body col" style={{ gap: 12 }}>
            <div className="small">
              <b>Foundation model</b>
              {valid ? <div className="muted">“{draft.name}”, {fmtNum(inspect?.total_params)} weights · <a style={{ cursor: 'pointer' }} onClick={() => setTab('build')}>edit in Builder</a></div>
                : <div className="muted">{onLab ? 'The model in the Builder is not valid yet.' : 'The Builder holds a model for a different dataset.'}</div>}
              {!valid && <button className="primary sm" style={{ marginTop: 6 }} onClick={starter}>Load the lab’s starter model</button>}
            </div>
            {foundations.length > 0 && (
              <label className="field"><span className="lbl">Or skip pretraining</span>
                <select value={from} onChange={(e) => setFrom(e.target.value)}>
                  <option value="">Pretrain from scratch</option>
                  {foundations.slice(0, 20).map((r) => <option key={r.id} value={r.id}>Reuse “{r.name}” ({fmtAgo(r.created_at)})</option>)}
                </select>
              </label>
            )}
            <div className="col" style={{ gap: 8 }}>
              <div className="row small"><span style={{ width: 22 }}>📚</span><span className="grow">Pretrain for</span><NumInput int style={{ width: 60 }} min={1} max={100} value={Number(opt.pretrain_epochs)} emptyValue={12} onChange={(n) => set('pretrain_epochs', n ?? 12)} /><span className="muted" style={{ width: 62 }}>epochs</span></div>
              <label className="row small"><input type="checkbox" style={{ width: 18 }} checked={!!opt.rl} onChange={(e) => set('rl', e.target.checked)} /><span className="grow">🎯 Reinforcement learning for</span><NumInput int style={{ width: 60 }} min={25} max={2000} value={Number(opt.rl_iterations)} emptyValue={150} onChange={(n) => set('rl_iterations', n ?? 150)} /><span className="muted" style={{ width: 62 }}>rounds</span></label>
              <label className="row small"><input type="checkbox" style={{ width: 18 }} checked={!!opt.safety} onChange={(e) => set('safety', e.target.checked)} /><span className="grow">🛡️ Safety fine-tune for</span><NumInput int style={{ width: 60 }} min={1} max={50} value={Number(opt.safety_epochs)} emptyValue={5} onChange={(n) => set('safety_epochs', n ?? 5)} /><span className="muted" style={{ width: 62 }}>epochs</span></label>
              <div className="row small"><span style={{ width: 22 }}>⚙️</span><span className="grow">Train on</span>
                <select value={device} onChange={(e) => setDevice(e.target.value)}>{(meta?.environment.available_devices ?? ['cpu']).map((d) => <option key={d} value={d}>{d}</option>)}</select>
              </div>
            </div>
            <div className="tiny faint">Untick a stage to see what the assistant is like without it. With the starter model the whole pipeline takes about five minutes on a laptop GPU and longer on a CPU.</div>
            <button className="primary big" disabled={busy || running || (!valid && !from)} onClick={start}>{running ? 'A job is already running' : '▶ Run the pipeline'}</button>
            <button className="sm ghost" onClick={() => setShowWorld(!showWorld)}>{showWorld ? '▾' : '▸'} Look at the training data</button>
            {showWorld && <>
              <div className="tiny muted">The toy internet (stage 1). Notice the wrong sums and the vault codes:</div>
              <pre className="json" style={{ maxHeight: 220, fontSize: 11.5 }}>{info.world.sample}</pre>
              <div className="tiny muted">The safety set (stage 3):</div>
              <pre className="json" style={{ maxHeight: 180, fontSize: 11.5 }}>{info.safety_sample}</pre>
            </>}
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <h3 className="grow">{pipe ? `Results: ${pipe.name}` : 'Results'}</h3>
            {pipe && <span className={`pill ${pipe.status === 'completed' ? 'ok' : pipe.status === 'running' ? 'accent' : 'warn'}`}>{pipe.status}</span>}
            {all.length > 1 && <select value={pipe?.id} onChange={(e) => setPicked(e.target.value)}>{all.map((p) => <option key={p.id} value={p.id}>{p.name} · {fmtAgo(p.created_at)}</option>)}</select>}
            {pipe && !isLive && <button className="sm ghost danger" title="Remove this pipeline from the list (its runs stay on the leaderboard)" onClick={async () => { await api.labDelete(pipe.id); setPicked(null); load() }}>✕</button>}
          </div>
          <div className="panel-body col" style={{ gap: 14 }}>
            {pipe?.error && <div className="error-box">{pipe.error}</div>}
            {done.length === 0 ? (
              <div className="empty">{pipe ? 'The first scores appear when pretraining finishes.' : 'Run a pipeline and its benchmark scores appear here, one bar per stage.'}</div>
            ) : <>
              <div className="row wrap small" style={{ gap: 14 }}>
                {done.map((s) => <span key={s.stage}><span style={{ color: STAGE_COLOR[s.stage] }}>■</span> {STAGE_LABEL[s.stage]}</span>)}
                <span className="spacer" />
                <span className="tiny faint">the small number is the change from the stage before, in points</span>
              </div>
              {(['capability', 'safety'] as const).map((g) => (
                <div key={g} className="col" style={{ gap: 8 }}>
                  <h4>{g === 'capability' ? 'What it can do' : 'How it behaves'}</h4>
                  {info.benchmarks.filter((b) => b.group === g).map((b) => <BenchRow key={b.id} b={b} stages={done} />)}
                </div>
              ))}
              <div className="note-box">
                <b>What this run shows</b>
                <ul style={{ margin: '6px 0 0', paddingLeft: 18, lineHeight: 1.6 }}>{takeaways(done).map((t) => <li key={t}>{t}</li>)}</ul>
              </div>
              <div>
                <h4 style={{ marginBottom: 6 }}>What it actually says</h4>
                <table className="tbl small">
                  <thead><tr><th>Asked</th>{done.map((s) => <th key={s.stage} style={{ color: STAGE_COLOR[s.stage] }}>{STAGE_LABEL[s.stage]}</th>)}</tr></thead>
                  <tbody>
                    {probes.map((p, i) => (
                      <tr key={i}>
                        <td className="mono">{p.q}<div className="tiny faint">{info.benchmarks.find((b) => b.id === p.bench)?.name}</div></td>
                        {done.map((s) => { const a = s.bench.samples[i]?.a ?? ''; return <td key={s.stage} className="mono" style={a.startsWith('i cannot') ? { color: 'var(--green)' } : undefined}>{a.startsWith('i cannot') ? info.refusal : a || '(nothing)'}</td> })}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="tiny faint" style={{ marginTop: 4 }}>The same questions after every stage. To ask your own, open Explore → Chat, pick one of the stage models and type, for example, “Q: 4+6=?” then a new line and “A:”.</div>
              </div>
              <div className="tiny muted">
                {pipe!.stages.map((s) => `${STAGE_LABEL[s.stage]}: ${s.reused ? 'reused' : s.seconds ? fmtDur(s.seconds) : s.status}`).join(' · ')}. Each stage is a run on the Leaderboard (marked 🏭).
              </div>
            </>}
          </div>
        </div>
      </div>
    </div>
  )
}
