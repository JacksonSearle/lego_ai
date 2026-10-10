"""Learner progress (XP, level, unlocks, achievements) for the guided mode.

The rules live in the frontend curriculum; the backend only persists the
state so it survives browser changes. Stored at data/progress.json.
"""
from __future__ import annotations

import json
from pathlib import Path

PROGRESS_FILE = Path(__file__).resolve().parent.parent / "data" / "progress.json"


def load() -> dict | None:
    if not PROGRESS_FILE.exists():
        return None
    try:
        return json.loads(PROGRESS_FILE.read_text())
    except json.JSONDecodeError:
        return None


def save(state: dict):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(PROGRESS_FILE)


def reset():
    PROGRESS_FILE.unlink(missing_ok=True)
