"""Learner profiles and their progress (XP, level, unlocks, achievements).

The rules live in the frontend curriculum; the backend only stores the state.
Several profiles can exist side by side (for example one guided profile and
one with everything unlocked); exactly one is active. Trained runs and the
leaderboard are shared between profiles.

Stored in data/profiles.json::

    {"active": "default", "profiles": {"default": {"name": "Me", "progress": {...} | null}}}
"""
from __future__ import annotations

import json
import secrets
import threading
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
PROFILES_FILE = DATA / "profiles.json"
LEGACY_FILE = DATA / "progress.json"  # single-profile format used by earlier versions
_lock = threading.Lock()


def _read() -> dict:
    if PROFILES_FILE.exists():
        try:
            return json.loads(PROFILES_FILE.read_text())
        except json.JSONDecodeError:
            pass
    progress = None
    if LEGACY_FILE.exists():
        try:
            progress = json.loads(LEGACY_FILE.read_text())
        except json.JSONDecodeError:
            progress = None
    return {"active": "default", "profiles": {"default": {"name": "Me", "progress": progress}}}


def _write(state: dict):
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = PROFILES_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(PROFILES_FILE)


def load() -> tuple[str, dict | None]:
    """The active profile's id and progress."""
    with _lock:
        s = _read()
        return s["active"], s["profiles"][s["active"]]["progress"]


def save(progress: dict, pid: str | None = None):
    """Store progress for profile ``pid`` (default: the active one).

    Pages pass the profile they loaded from, so a save that arrives just after
    switching profiles can never overwrite the wrong one. Saves for a profile
    that no longer exists are dropped.
    """
    with _lock:
        s = _read()
        pid = pid or s["active"]
        if pid not in s["profiles"]:
            return
        s["profiles"][pid]["progress"] = progress
        _write(s)


def reset():
    with _lock:
        s = _read()
        s["profiles"][s["active"]]["progress"] = None
        _write(s)


def list_profiles() -> dict:
    with _lock:
        s = _read()
        out = []
        for pid, p in s["profiles"].items():
            pr = p.get("progress") or {}
            out.append({
                "id": pid, "name": p["name"], "active": pid == s["active"], "mode": pr.get("mode"),
                "xp": pr.get("xp", 0), "avatar": pr.get("avatar", "🐣"), "demo": bool(p.get("demo")),
            })
        return {"active": s["active"], "profiles": out}


def create(name: str, *, activate: bool = True, demo: bool = False, progress: dict | None = None) -> str:
    name = (name or "").strip()[:40] or "New profile"
    with _lock:
        s = _read()
        pid = secrets.token_hex(4)
        s["profiles"][pid] = {"name": name, "progress": progress, "demo": demo}
        if activate:
            s["active"] = pid
        _write(s)
        return pid


def activate(pid: str):
    with _lock:
        s = _read()
        if pid not in s["profiles"]:
            raise KeyError("No such profile")
        s["active"] = pid
        _write(s)


def rename(pid: str, name: str):
    with _lock:
        s = _read()
        if pid not in s["profiles"]:
            raise KeyError("No such profile")
        s["profiles"][pid]["name"] = (name or "").strip()[:40] or s["profiles"][pid]["name"]
        _write(s)


def delete(pid: str):
    with _lock:
        s = _read()
        if pid not in s["profiles"]:
            raise KeyError("No such profile")
        if len(s["profiles"]) == 1:
            raise ValueError("You can't delete the only profile. Reset its progress instead.")
        del s["profiles"][pid]
        if s["active"] == pid:
            s["active"] = next(iter(s["profiles"]))
        _write(s)
