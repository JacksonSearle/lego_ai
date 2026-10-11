"""Hugging Face Hub: import datasets, and publish and import trained models.

* Datasets are read from the Hub's automatic Parquet export into the same table form as an
  uploaded file (see ``userdata.py``), so the rest of the import is identical.
* A published model is a small repository holding the run's config, its weights, a model
  card and, when possible, the standalone training script. Importing one creates a local run.

Weights travel as a plain dictionary of tensors and are loaded with ``weights_only=True``,
which cannot run code. A config's custom blocks *are* code, so importing a model that uses
them needs the user's explicit go-ahead.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import torch

from . import config as cfgmod, custom, datasets, export, model as modelmod, runs, userdata

TAG = "lego-ai"
MAX_DATASET_ROWS = 20000


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def _friendly(e: Exception) -> str:
    name = type(e).__name__
    if "RepositoryNotFound" in name or "404" in str(e)[:200]:
        return "Not found on the Hub. Check the name (it looks like owner/name). Private repositories need you to be logged in."
    if "LocalTokenNotFound" in name or "401" in str(e)[:200]:
        return "You are not logged in to Hugging Face, or the token is not valid."
    if "403" in str(e)[:200]:
        return "Your token is not allowed to do this. Publishing needs a token with write access."
    if "Connect" in name or "Timeout" in name or "resolve" in str(e).lower():
        return "Could not reach huggingface.co. Check your internet connection."
    return f"{name}: {str(e)[:300]}"


class HubError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------
def status() -> dict:
    from huggingface_hub import get_token

    if not get_token():
        return {"logged_in": False, "user": None}
    try:
        who = _api().whoami()
    except Exception as e:  # noqa: BLE001
        return {"logged_in": False, "user": None, "error": _friendly(e)}
    return {"logged_in": True, "user": who.get("name"), "orgs": [o.get("name") for o in who.get("orgs") or []]}


def login(token: str) -> dict:
    """Check a token and, if it works, save it where every Hugging Face tool looks for it."""
    from huggingface_hub import login as hf_login

    token = (token or "").strip()
    if not token:
        raise HubError("Paste an access token.")
    try:
        from huggingface_hub import HfApi

        HfApi(token=token).whoami()
        hf_login(token=token, add_to_git_credential=False)
    except Exception as e:  # noqa: BLE001
        raise HubError(_friendly(e)) from None
    return status()


def logout() -> dict:
    from huggingface_hub import logout as hf_logout

    try:
        hf_logout()
    except Exception:  # noqa: BLE001
        pass
    return status()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def dataset_table(repo: str, config: str | None = None) -> dict:
    """Read (the first part of) a Hub dataset into a table. Returns the staged table."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo = repo.strip().removeprefix("https://huggingface.co/datasets/").strip("/")
    if repo.count("/") > 1 or not repo:
        raise HubError("A dataset name looks like owner/name, for example scikit-learn/iris.")
    try:
        files = [f for f in _api().list_repo_files(repo, repo_type="dataset", revision="refs/convert/parquet") if f.endswith(".parquet")]
    except Exception as e:  # noqa: BLE001
        raise HubError(_friendly(e)) from None
    if not files:
        raise HubError("The Hub has no table version of this dataset, so it cannot be imported here.")
    configs = sorted({f.split("/")[0] for f in files})
    cfg = config if config in configs else "default" if "default" in configs else configs[0]
    by_split: dict[str, list[str]] = {}
    for f in sorted(files):
        parts = f.split("/")
        if parts[0] == cfg and len(parts) >= 3:
            by_split.setdefault(parts[1], []).append(f)
    train = "train" if "train" in by_split else sorted(by_split)[0]
    test = next((s for s in ("test", "validation", "valid", "dev") if s in by_split and s != train), None)

    columns: list[str] = []
    class_names: dict[str, list[str]] = {}
    rows, marks = [], []

    def read(split: str, mark: str, cap: int):
        nonlocal columns
        got = 0
        for f in by_split[split]:
            if got >= cap:
                break
            try:
                path = hf_hub_download(repo, f, repo_type="dataset", revision="refs/convert/parquet")
            except Exception as e:  # noqa: BLE001
                raise HubError(_friendly(e)) from None
            pf = pq.ParquetFile(path)
            meta = json.loads((pf.schema_arrow.metadata or {}).get(b"huggingface", b"{}"))
            for col, feat in ((meta.get("info") or {}).get("features") or {}).items():
                if isinstance(feat, dict) and feat.get("_type") == "ClassLabel" and feat.get("names"):
                    class_names[col] = feat["names"]
            for batch in pf.iter_batches(batch_size=1000):
                if not columns:
                    columns = list(batch.schema.names)
                data = batch.to_pydict()
                for i in range(batch.num_rows):
                    row = []
                    for c in columns:
                        v = data.get(c, [None] * batch.num_rows)[i]
                        if isinstance(v, dict):  # an image: {"bytes": ..., "path": ...}
                            v = v.get("bytes")
                        elif isinstance(v, (list, tuple)):
                            v = json.dumps(v)[:500]
                        row.append(v)
                    rows.append(row)
                    marks.append(mark)
                got += batch.num_rows
                if got >= cap:
                    break

    read(train, "train", MAX_DATASET_ROWS)
    if test:
        read(test, "test", MAX_DATASET_ROWS // 4)
    table = userdata._table(columns, rows, f"huggingface.co/datasets/{repo}" + (f" ({cfg})" if len(configs) > 1 else ""), marks if test else None, class_names)
    table["hub"] = {"repo": repo, "config": cfg, "configs": configs, "truncated": len(rows) >= MAX_DATASET_ROWS}
    return table


# ---------------------------------------------------------------------------
# Models: publish
# ---------------------------------------------------------------------------
def _card(run: dict, info: datasets.DatasetInfo, repo: str, has_script: bool) -> str:
    s, cfg = run["summary"], run["config"]
    blocks = " → ".join(f"{b['type']}{'×' + str(b['repeat']) if b.get('repeat', 1) > 1 else ''}" for b in cfg["architecture"]["blocks"])
    pct = lambda v: "n/a" if v is None else f"{v * 100:.2f}%"  # noqa: E731
    t = cfg["training"]
    lines = [
        "---", "library_name: pytorch", "tags:", f"- {TAG}", f"- {info.task}", "---", "",
        f"# {run['name']}", "",
        f"A model built from blocks and trained with Lego AI on **{info.name}**.", "",
        "| | |", "|---|---|",
        f"| Validation accuracy (best) | {pct(s.get('best_val_acc'))} |",
        f"| Test accuracy | {pct(s.get('test_acc'))} |",
    ]
    if s.get("shift_acc") is not None:
        lines.append(f"| Shifted test accuracy | {pct(s.get('shift_acc'))} |")
    lines += [
        f"| Weights | {s.get('params', 0):,} |",
        f"| Epochs | {s.get('epochs_done', 0)} |",
        "", "## Architecture", "", f"Input {'×'.join(map(str, info.input_shape))} → {blocks}", "",
        "## Training", "",
        f"Optimizer {t['optimizer']}, learning rate {t['lr']}, batch size {t['batch_size']}, schedule {t['scheduler']}, seed {t['seed']}.", "",
        "## Files", "",
        "- `config.lego.json`: the full recipe (dataset, blocks, every training setting). Import it in Lego AI to rebuild or retrain the model.",
        "- `model.pt`: the trained weights, a plain dictionary of tensors (`torch.load(..., weights_only=True)`).",
    ]
    if has_script:
        lines.append("- `train.py`: a standalone PyTorch script that defines the same model and trains it from scratch.")
    lines += ["", "## Use it in Lego AI", "", f"Leaderboard → Hugging Face → Import a model → `{repo}`", ""]
    if info.custom:
        lines += [f"Note: this model was trained on a dataset its author imported (“{info.name}”), which is not part of this repository.", ""]
    return "\n".join(lines)


def prepare_folder(run_id: str, repo: str, folder: Path) -> list[str]:
    """Write everything a published model consists of into ``folder``. Returns the file names."""
    run = runs.get(run_id)
    path = runs.checkpoint_path(run_id, "best")
    if not path.exists():
        path = runs.checkpoint_path(run_id, "last")
    if not path.exists():
        raise HubError("This run has no saved weights to publish.")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise HubError("An ensemble is several models, which cannot be published as one. Publish its members instead.")
    info = datasets.info_for(run["config"]["dataset"])
    torch.save({k: v.clone() for k, v in ckpt["model"].items()}, folder / "model.pt")
    cfg = {**run["config"], "result": run["summary"], "published": {"name": run["name"], "at": time.time()}}
    (folder / "config.lego.json").write_text(json.dumps(runs.sanitize(cfg), indent=2))
    has_script = False
    if not info.custom:
        try:
            files = export.export_scripts(run["config"])
            for name in ("train.py", "generate.py", "predict.py"):
                if name in files:
                    (folder / name).write_text(files[name])
            has_script = True
        except Exception:  # noqa: BLE001 - the script is a bonus; fine-tuned runs cannot have one
            pass
    (folder / "README.md").write_text(_card(run, info, repo, has_script))
    return sorted(p.name for p in folder.iterdir())


def publish(run_id: str, name: str, private: bool = False) -> dict:
    st = status()
    if not st["logged_in"]:
        raise HubError("Log in to Hugging Face first.")
    name = name.strip().strip("/")
    if not name:
        raise HubError("Give the repository a name.")
    repo = name if "/" in name else f"{st['user']}/{name}"
    api = _api()
    with tempfile.TemporaryDirectory() as tmp:
        files = prepare_folder(run_id, repo, Path(tmp))
        try:
            api.create_repo(repo, repo_type="model", private=bool(private), exist_ok=True)
            api.upload_folder(repo_id=repo, folder_path=tmp, repo_type="model", commit_message=f"Publish from Lego AI: {runs.get(run_id)['name']}")
        except Exception as e:  # noqa: BLE001
            raise HubError(_friendly(e)) from None
    url = f"https://huggingface.co/{repo}"
    run = runs.get(run_id)
    run["hub"] = {"repo": repo, "url": url, "published_at": time.time(), "private": bool(private)}
    runs.save(run)
    return {"repo": repo, "url": url, "files": files}


# ---------------------------------------------------------------------------
# Models: find and import
# ---------------------------------------------------------------------------
def search(query: str = "", limit: int = 30) -> list[dict]:
    try:
        models = _api().list_models(filter=TAG, search=query.strip() or None, limit=limit, sort="downloads")
        return [{"repo": m.id, "downloads": getattr(m, "downloads", None), "likes": getattr(m, "likes", None), "updated": str(getattr(m, "last_modified", "") or "")[:10]} for m in models]
    except Exception as e:  # noqa: BLE001
        raise HubError(_friendly(e)) from None


def import_folder(folder: Path, repo: str, allow_code: bool = False) -> dict:
    """Create a local run from a published model's files."""
    cfg_path, w_path = folder / "config.lego.json", folder / "model.pt"
    if not cfg_path.exists() or not w_path.exists():
        raise HubError("This repository was not published from Lego AI: it needs config.lego.json and model.pt.")
    raw = json.loads(cfg_path.read_text())
    cfg = cfgmod.normalize(raw)
    cfg.pop("init_from", None)  # the starting run exists only on the author's computer
    blocks_code = cfg.get("custom_blocks") or {}
    if blocks_code and not allow_code:
        return {"needs_confirm": True, "repo": repo, "custom_blocks": {k: v["source"] for k, v in blocks_code.items()}}
    ds_id = cfg["dataset"]["id"]
    if ds_id not in datasets.DATASETS:
        raise HubError(f"This model was trained on a dataset you do not have (“{ds_id}”). Import that dataset first, with the same name.")
    custom.ensure_from_config(blocks_code)
    info = datasets.info_for(cfg["dataset"])
    overlay = custom.build_registry_overlay(blocks_code)
    net, report = modelmod.build(cfg["architecture"], info.input_shape, info.ctx(), strict=True, overlay=overlay)
    state = torch.load(w_path, map_location="cpu", weights_only=True)  # tensors only: cannot run code
    try:
        net.load_state_dict(state)
    except RuntimeError:
        raise HubError("The weights do not fit the architecture in the config, so this model cannot be loaded.") from None
    result = raw.get("result") or {}
    name = (raw.get("published") or {}).get("name") or cfg["name"]
    cfg["name"] = name
    run = runs.create(cfg, hub={"repo": repo, "url": f"https://huggingface.co/{repo}", "imported_at": time.time()})
    torch.save({"model": net.state_dict(), "epoch": result.get("epochs_done", 0), "val": None}, runs.checkpoint_path(run["id"], "best"))
    keep = ("params", "epochs_done", "best_val_acc", "best_val_loss", "final_val_acc", "final_val_loss", "test_acc", "test_loss", "shift_acc", "shifts", "train_examples")
    run["summary"] = {**{k: result[k] for k in keep if k in result}, "params": report.total_params}
    run["status"] = "completed"
    runs.save(run)
    return {"run_id": run["id"], "name": name}


def import_model(repo: str, allow_code: bool = False) -> dict:
    from huggingface_hub import hf_hub_download

    repo = repo.strip().removeprefix("https://huggingface.co/").strip("/")
    if repo.count("/") != 1:
        raise HubError("A model name looks like owner/name.")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            for f in ("config.lego.json", "model.pt"):
                hf_hub_download(repo, f, repo_type="model", local_dir=tmp)
        except Exception as e:  # noqa: BLE001
            if "EntryNotFound" in type(e).__name__:
                raise HubError("This repository was not published from Lego AI: it needs config.lego.json and model.pt.") from None
            raise HubError(_friendly(e)) from None
        return import_folder(Path(tmp), repo, allow_code)
