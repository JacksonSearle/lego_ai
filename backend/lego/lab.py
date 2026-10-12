"""The Frontier Lab pipeline: pretrain → reinforcement learning → safety fine-tune → benchmarks.

A scaled-down version of how a frontier lab turns a pile of text into an assistant, run on
the toy world in ``labworld.py`` with a model built from blocks:

1. **Pretrain** a foundation model on "the internet" by next-character prediction
   (an ordinary training run on the ``lab_world`` dataset).
2. **Reinforcement learning** with GRPO on tasks a program can check: the model writes
   several answers to each question, and answers that beat the group's average are made
   more likely.
3. **Safety fine-tune**: supervised training on refusals for requests it must not fulfil,
   mixed with ordinary questions (a run on ``lab_safety`` starting from stage 2's weights).
4. **Benchmarks** after every stage, so you can see what each one bought and what it cost.

Every stage is an ordinary run on the leaderboard, tagged with the pipeline it belongs to.
"""
from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import config as cfgmod, datasets, labworld as W, runs
from .jobs import Job, Stopped
from .rng import RngStream
from .train import build_model, configure_determinism, resolve_device, train_run

DIR = Path(__file__).resolve().parent.parent / "data" / "pipelines"
MAX_NEW = 12  # characters the model may write per answer
NEWLINE = W.STOI["\n"]
_FILLER = W.pretrain_text(60, W.SEED + 7)


# ---------------------------------------------------------------------------
# Asking the model questions
# ---------------------------------------------------------------------------
def contexts(questions: list[str], width: int) -> torch.Tensor:
    """Each question as the model sees it: some earlier text, then "Q: ...\\nA: ", all the same length."""
    rows = [W.encode((_FILLER + f"Q: {q}\nA: ")[-width:]) for q in questions]
    return torch.stack(rows)


@torch.no_grad()
def write(net, ctx: torch.Tensor, greedy: bool = True, gen: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Let the model continue each context by MAX_NEW characters.

    Returns (characters, log-probability of each chosen character).
    """
    seq = ctx
    toks, logps = [], []
    for _ in range(MAX_NEW):
        logp = F.log_softmax(net(seq)[:, -1].float(), -1)
        if greedy:
            nxt = logp.argmax(-1)
        else:
            nxt = torch.multinomial(logp.exp().cpu(), 1, generator=gen).squeeze(1).to(logp.device)
        toks.append(nxt)
        logps.append(logp.gather(1, nxt[:, None]).squeeze(1))
        seq = torch.cat([seq, nxt[:, None]], 1)
    return torch.stack(toks, 1), torch.stack(logps, 1)


def answers(tokens: torch.Tensor) -> list[str]:
    """The text of each answer: everything before the first line break."""
    out = []
    for row in tokens.tolist():
        out.append("".join(W.VOCAB[t] for t in row).split("\n")[0])
    return out


def _answer_mask(tokens: torch.Tensor) -> torch.Tensor:
    """1 for the characters that are part of the answer (up to and including its line break)."""
    ended = (tokens == NEWLINE).cumsum(1)
    return ((ended == 0) | ((ended == 1) & (tokens == NEWLINE))).float()


@torch.no_grad()
def evaluate(net, seq_len: int, device, n: int = 60, stop=None) -> dict:
    """Run every benchmark. Returns {"scores": {id: 0..1}, "samples": [...], "capability", "safety"}."""
    was_training = net.training
    net.eval()
    width = seq_len - MAX_NEW
    items = W.benchmark_items(n)
    scores, samples = {}, []
    for bench, qs in items.items():
        if stop:
            stop()
        got = answers(write(net, contexts([q for q, _ in qs], width).to(device))[0])
        if bench in ("refusal", "jailbreak"):
            ok = [W.is_refusal(g) for g in got]
        elif bench == "helpful":
            ok = [g == a for (_, a), g in zip(qs, got)]
        else:
            ok = [g == a for (_, a), g in zip(qs, got)]
        scores[bench] = sum(ok) / len(ok)
    # Language: next-character accuracy on unseen stories.
    rng = random.Random(W.SEED + 55)
    ids = W.encode(" ".join(W.story(rng) for _ in range(120)))
    k = (len(ids) - 1) // seq_len
    x, y = ids[: k * seq_len].view(k, seq_len).to(device), ids[1 : k * seq_len + 1].view(k, seq_len).to(device)
    scores["language"] = (net(x).argmax(-1) == y).float().mean().item()
    # The same handful of prompts after every stage, to read what the model actually says.
    got = answers(write(net, contexts([q for _, q in W.PROBES], width).to(device))[0])
    for (bench, q), g in zip(W.PROBES, got):
        samples.append({"bench": bench, "q": q, "a": g})
    cap = [scores[b["id"]] for b in W.BENCHMARKS if b["group"] == "capability"]
    safe = [scores[b["id"]] for b in W.BENCHMARKS if b["group"] == "safety"]
    if was_training:
        net.train()
    return {"scores": scores, "samples": samples, "capability": sum(cap) / len(cap), "safety": sum(safe) / len(safe)}


# ---------------------------------------------------------------------------
# Stage 2: reinforcement learning with GRPO
# ---------------------------------------------------------------------------
def grpo_step(net, ref, opt, questions: list[tuple[str, str]], group: int, width: int, device, gen: torch.Generator,
              retain: torch.Tensor | None = None, clip: float = 0.2, kl_weight: float = 0.5, inner: int = 2) -> dict:
    """One round of Group Relative Policy Optimization.

    For every question the model samples ``group`` answers. Each answer's reward is compared
    with the average of its own group: better than average is pushed up, worse is pushed
    down. No separate "critic" network is needed, which is the point of GRPO. A penalty
    keeps the model close to the reference (the pretrained model), both on its answers and
    on ordinary text (``retain``), so it does not forget everything else while chasing reward.
    """
    b = len(questions)
    ctx = contexts([q for q, _ in questions], width).to(device).repeat_interleave(group, 0)
    tokens, logp_old = write(net, ctx, greedy=False, gen=gen)
    mask = _answer_mask(tokens)
    texts = answers(tokens)
    rewards = torch.tensor([W.reward(questions[i // group][1], t) for i, t in enumerate(texts)], device=device).view(b, group)
    adv = (rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-4)
    adv = torch.where(rewards.std(1, keepdim=True) < 1e-6, torch.zeros_like(adv), adv).view(-1)  # a group that all agree teaches nothing

    seq = torch.cat([ctx, tokens], 1)
    inp, first = seq[:, :-1], ctx.shape[1] - 1
    with torch.no_grad():
        ref_logp = F.log_softmax(ref(inp)[:, first:].float(), -1)
        retain_ref = F.log_softmax(ref(retain).float(), -1) if retain is not None else None
    old = (logp_old * mask).sum(1)
    loss_value = kl_value = 0.0
    for _ in range(inner):
        logp_all = F.log_softmax(net(inp)[:, first:].float(), -1)
        new = (logp_all.gather(2, tokens[:, :, None]).squeeze(2) * mask).sum(1)
        ratio = (new - old).exp()
        surrogate = torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
        kl = ((ref_logp.exp() * (ref_logp - logp_all)).sum(-1) * mask).sum() / mask.sum()
        if retain is not None:
            # The same leash on plain text from the pretraining corpus: stay the model you were.
            r_new = F.log_softmax(net(retain).float(), -1)
            kl = kl + (retain_ref.exp() * (retain_ref - r_new)).sum(-1).mean()
        loss = -surrogate.mean() + kl_weight * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        loss_value, kl_value = loss.item(), kl.item()
    exact = sum(1 for i, t in enumerate(texts) if t == questions[i // group][1]) / len(texts)
    return {"reward": rewards.mean().item(), "exact": exact, "loss": loss_value, "kl": kl_value}


def rl_run(job: Job, run_id: str, iterations: int, group: int = 8, questions: int = 24, tag: dict | None = None) -> dict:
    """Stage 2 as a run: GRPO from the weights of the run named in ``init_from``.

    Reported like a training run so the usual charts work: one "epoch" is 25 rounds, the
    training curve is the average reward, and validation accuracy is the score on the
    checkable benchmarks (math and puzzles).
    """
    from .combine import best_state

    tag = {"run_id": run_id, **(tag or {})}
    emit = lambda e: job.emit({**tag, **e})  # noqa: E731
    run = runs.get(run_id)
    config = run["config"]
    t = config["training"]
    started = time.time()
    run["status"] = "running"
    config["environment"] = cfgmod.environment()
    runs.save(run)
    try:
        device = resolve_device(t["device"])
        configure_determinism()
        rng_stream = RngStream(t["seed"], device)
        with rng_stream:
            net, info, _ = build_model(config)
        _, start = best_state(config["init_from"]["run_id"])
        net.load_state_dict(start)
        net.to(device).eval()  # no dropout while collecting or learning from samples
        ref = copy.deepcopy(net)
        for p in ref.parameters():
            p.requires_grad_(False)
        data = datasets.load_for(config["dataset"])
        width = info.seq_len - MAX_NEW
        opt = torch.optim.AdamW(net.parameters(), lr=t["lr"], weight_decay=0.0)
        rng = random.Random(t["seed"])
        gen = torch.Generator().manual_seed(t["seed"])
        per_epoch = 25
        epochs = max(1, iterations // per_epoch)
        total = epochs * per_epoch
        n_params = sum(p.numel() for p in net.parameters())
        emit({"type": "start", "name": config["name"], "params": n_params, "epochs": epochs, "steps_per_epoch": per_epoch, "total_steps": total,
              "resume_epoch": 0, "history": None, "device": str(device), "train_examples": total * questions})
        run["summary"].update({"params": n_params, "steps_per_epoch": per_epoch, "train_examples": total * questions})
        history = {"steps": [], "epochs": []}
        best = -1.0
        step = 0
        for epoch in range(epochs):
            ep_start = time.time()
            agg = {"reward": 0.0, "exact": 0.0, "loss": 0.0}
            for i in range(per_epoch):
                if job.stopping:
                    raise Stopped()
                tasks = [W.rl_task(rng) for _ in range(questions)]
                idx = torch.randint(0, len(data.train_x), (32,), generator=gen)
                st = grpo_step(net, ref, opt, tasks, group, width, device, gen, retain=data.train_x[idx].to(device))
                step += 1
                for k in agg:
                    agg[k] += st[k] / per_epoch
                if step % 5 == 0 or i == per_epoch - 1:
                    point = {"step": step, "epoch": epoch + (i + 1) / per_epoch, "loss": st["loss"], "acc": st["reward"], "lr": t["lr"]}
                    history["steps"].append(point)
                    emit({"type": "step", **point})
            bench = evaluate(net, info.seq_len, device, n=40)
            score = (bench["scores"]["math"] + bench["scores"]["puzzles"]) / 2
            ep = {"epoch": epoch + 1, "step": step, "train_loss": agg["loss"], "train_acc": agg["reward"], "val_loss": 1 - agg["exact"], "val_acc": score,
                  "lr": t["lr"], "seconds": time.time() - ep_start}
            history["epochs"].append(ep)
            best = max(best, score)
            torch.save({"model": net.state_dict(), "epoch": epoch + 1, "val": None}, runs.checkpoint_path(run_id, "best"))
            runs.save_metrics(run_id, history)
            run["summary"].update({"epochs_done": epoch + 1, "steps": step, "best_val_acc": best, "best_val_loss": ep["val_loss"], "final_val_acc": score,
                                   "final_val_loss": ep["val_loss"], "final_train_loss": ep["train_loss"], "duration": time.time() - started})
            runs.save(run)
            emit({"type": "epoch", **ep, "best_val_acc": best})
        run["summary"].update({"test_acc": score, "test_loss": ep["val_loss"], "duration": time.time() - started})
        run["status"] = "completed"
        runs.save(run)
        emit({"type": "done", "summary": run["summary"], "status": "completed"})
        return run["summary"]
    except Stopped:
        run = runs.get(run_id)
        run["status"] = "stopped"
        runs.save(run)
        raise
    except Exception as e:
        run = runs.get(run_id)
        run["status"] = "failed"
        run["error"] = f"{type(e).__name__}: {e}"
        runs.save(run)
        raise


# ---------------------------------------------------------------------------
# The whole pipeline
# ---------------------------------------------------------------------------
STAGES = [
    {"id": "pretrain", "name": "Pretrain", "icon": "📚", "what": "Read the toy internet and learn to predict the next character. This is where all its knowledge comes from, mistakes and secrets included."},
    {"id": "rl", "name": "Reinforcement learning", "icon": "🎯", "what": "Answer questions a program can check. Sample several answers each, reward the right ones (GRPO). Fixes mistakes it copied from the internet."},
    {"id": "safety", "name": "Safety fine-tune", "icon": "🛡️", "what": "Train on examples of refusing requests it must not fulfil, mixed with normal questions so it stays helpful."},
]
DEFAULTS = {"pretrain_epochs": 12, "rl_iterations": 150, "safety_epochs": 5, "rl_lr": 1e-4, "safety_lr": 5e-4, "rl": True, "safety": True}


def info() -> dict:
    """What the Frontier Lab page needs to explain itself."""
    p = W.PLACES[0]
    return {
        "stages": STAGES, "benchmarks": W.BENCHMARKS, "defaults": DEFAULTS, "refusal": W.REFUSAL,
        "world": {"places": len(W.PLACES), "example": {"place": p, **W.WORLD["places"][p]}, "sample": W.pretrain_text(9, W.SEED + 11).strip()},
        "safety_sample": W.safety_text(5, W.SEED + 12).strip(),
    }


def _bench_run(run_id: str, device_name: str, stop) -> dict:
    """Benchmark the weights a run ended with (the ones the next stage starts from)."""
    from .combine import best_state

    run, state = best_state(run_id)
    cfg = run["config"]
    net, info_, _ = build_model(cfg, seed=0)
    net.load_state_dict(state)
    device = resolve_device(device_name)
    out = evaluate(net.to(device), info_.seq_len, device, stop=stop)
    run = runs.get(run_id)
    run["summary"]["bench"] = out
    runs.save(run)
    return out


def _save(record: dict):
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = DIR / f"{record['id']}.tmp"
    tmp.write_text(json.dumps(runs.sanitize(record), indent=2))
    tmp.replace(DIR / f"{record['id']}.json")


def pipelines() -> list[dict]:
    if not DIR.exists():
        return []
    out = []
    for p in DIR.glob("*.json"):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda r: r["created_at"], reverse=True)


def delete(pipeline_id: str):
    if not pipeline_id.replace("-", "").isalnum():
        raise ValueError("bad id")
    (DIR / f"{pipeline_id}.json").unlink(missing_ok=True)


def run_pipeline(job: Job, base_config: dict, opts: dict):
    o = {**DEFAULTS, **{k: v for k, v in (opts or {}).items() if k in DEFAULTS and v is not None}}
    base = cfgmod.normalize(base_config)
    base.pop("environment", None)
    base.pop("init_from", None)
    base["dataset"] = {"id": "lab_world", "train_subset": None, "seq_len": base["dataset"].get("seq_len") if base["dataset"]["id"] == "lab_world" else None}
    name = base["name"]
    device = base["training"]["device"]
    pid = runs.new_id()
    plan = ["pretrain"] + (["rl"] if o["rl"] else []) + (["safety"] if o["safety"] else [])
    record = {
        "id": pid, "name": name, "created_at": time.time(), "status": "running", "options": o,
        "stages": [{"stage": s, "run_id": None, "status": "waiting", "bench": None, "seconds": None} for s in plan],
    }
    job.meta.update({"pipeline_id": pid, "name": name})
    _save(record)
    job.emit({"type": "pipeline_start", "pipeline": record})
    prev = (opts or {}).get("start_from")
    try:
        for i, st in enumerate(record["stages"]):
            job.check_stop()
            stage = st["stage"]
            t0 = time.time()
            cfg = copy.deepcopy(base)
            tag = {"id": pid, "stage": stage, "index": i, "n": len(plan), "name": name}
            if stage == "pretrain":
                if prev:  # reuse a foundation model that is already trained
                    src = runs.get(prev)
                    if src["config"]["dataset"]["id"] != "lab_world" or src["status"] != "completed":
                        raise ValueError("The foundation model must be a finished run trained on Lab World.")
                    base["architecture"], base["custom_blocks"] = src["config"]["architecture"], src["config"].get("custom_blocks") or {}
                    base["dataset"]["seq_len"] = src["config"]["dataset"].get("seq_len")
                    if src["config"].get("init_stream"):
                        base["init_stream"] = src["config"]["init_stream"]
                    st["run_id"] = prev
                    st["reused"] = True
                else:
                    cfg["name"] = f"{name} · pretrained"
                    cfg["training"]["epochs"] = int(o["pretrain_epochs"])
                    run = runs.create(cfg, pipeline=tag)
                    st["run_id"] = run["id"]
            elif stage == "rl":
                cfg["name"] = f"{name} · after RL"
                cfg["init_from"] = {"run_id": prev, "name": runs.get(prev)["name"]}
                cfg["training"].update({"lr": float(o["rl_lr"]), "epochs": max(1, int(o["rl_iterations"]) // 25), "scheduler": "none", "warmup_epochs": 0, "weight_decay": 0.0})
                run = runs.create(cfg, pipeline=tag)
                st["run_id"] = run["id"]
            else:
                cfg["name"] = f"{name} · safety-tuned"
                cfg["dataset"]["id"] = "lab_safety"
                cfg["init_from"] = {"run_id": prev, "name": runs.get(prev)["name"]}
                cfg["training"].update({"lr": float(o["safety_lr"]), "epochs": int(o["safety_epochs"]), "scheduler": "cosine", "warmup_epochs": 0.5, "max_steps": None})
                run = runs.create(cfg, pipeline=tag)
                st["run_id"] = run["id"]
            st["status"] = "running"
            _save(record)
            job.emit({"type": "stage_start", "stage": stage, "index": i, "run_id": st["run_id"], "n": len(plan)})
            if not st.get("reused"):
                if stage == "rl":
                    rl_run(job, st["run_id"], int(o["rl_iterations"]), tag={"stage": stage})
                else:
                    train_run(job, st["run_id"], tag={"stage": stage})
            job.emit({"type": "status", "message": f"Running the benchmarks after “{stage}”…", "run_id": st["run_id"]})
            st["bench"] = _bench_run(st["run_id"], device, job.check_stop)
            st.update(status="done", seconds=time.time() - t0)
            prev = st["run_id"]
            _save(record)
            job.emit({"type": "stage_end", "stage": stage, "index": i, "run_id": st["run_id"], "bench": st["bench"]})
        record["status"] = "completed"
    except Stopped:
        record["status"] = "stopped"
        raise
    except Exception as e:
        record["status"] = "failed"
        record["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        for st in record["stages"]:
            if st["status"] == "running":
                st["status"] = "stopped"
        _save(record)
        job.emit({"type": "pipeline_end", "pipeline": record})
    return record
