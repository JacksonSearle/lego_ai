"""Small local settings file (data/settings.json): things that differ per install.

    server_url   the shared leaderboard this copy submits to and sends feedback to
    player       the name shown next to this user's submissions
    workers      other computers (for example a rented GPU machine) that can train for this one
    update_repo  the GitHub repository whose releases this copy updates from
"""
from __future__ import annotations

import json
import os
import secrets
import threading
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "data" / "settings.json"
ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULTS = {
    "server_url": os.environ.get("LEGO_SERVER_URL", ""),
    "player": "",
    "workers": [],
    "update_repo": "JacksonSearle/lego_ai",
    "feedback_email": "",
    "submissions": {},  # run id → {"id", "track", "server"}
}
_lock = threading.Lock()


def load() -> dict:
    with _lock:
        try:
            stored = json.loads(PATH.read_text())
        except (OSError, ValueError):
            stored = {}
        return {**DEFAULTS, **stored}


def save(changes: dict) -> dict:
    with _lock:
        try:
            cur = json.loads(PATH.read_text())
        except (OSError, ValueError):
            cur = {}
        cur.update(changes)
        PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(cur, indent=2))
        tmp.replace(PATH)
    return load()


def public() -> dict:
    """Settings as shown to the page: worker tokens are never sent back."""
    s = load()
    return {
        "server_url": s["server_url"], "player": s["player"], "update_repo": s["update_repo"], "feedback_email": s["feedback_email"],
        "workers": [{"id": w["id"], "name": w["name"], "url": w["url"], "has_token": bool(w.get("token"))} for w in s["workers"]],
    }


def version() -> str:
    try:
        return (ROOT / "VERSION").read_text().strip()
    except OSError:
        return "dev"


def _access_token() -> str:
    """The token that protects this copy's own API when other computers can reach it.

    Set with LEGO_TOKEN. A leaderboard host is always reachable by others, so it gets a
    generated one (kept in data/host/access_token) if none was given.
    """
    token = os.environ.get("LEGO_TOKEN", "")
    if token or os.environ.get("LEGO_HOST") != "1":
        return token
    path = PATH.parent / "host" / "access_token"
    try:
        return path.read_text().strip()
    except OSError:
        path.parent.mkdir(parents=True, exist_ok=True)
        token = secrets.token_urlsafe(18)
        path.write_text(token)
        return token


ACCESS_TOKEN = _access_token()
