"""Is a newer version of Lego AI available?

Releases are published on GitHub. This module only *looks*; the launcher (``launch.py`` in
the project folder) installs an update the next time the app is started, because code
cannot safely replace itself while it is running.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

from . import settings

_cache: dict = {"at": 0.0, "value": None}


def parse(v: str) -> tuple:
    return tuple(int(p) if p.isdigit() else 0 for p in v.strip().lstrip("vV").split(".")[:4])


def latest(repo: str, timeout: float = 6) -> dict | None:
    """The newest release of ``repo``: {"version", "notes", "url", "zip"} or None if there is none."""
    url = os.environ.get("LEGO_UPDATE_URL") or f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "lego-ai"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        rel = json.loads(r.read())
    if not rel.get("tag_name"):
        return None
    assets = [a for a in rel.get("assets") or [] if str(a.get("name", "")).endswith(".zip")]
    return {
        "version": rel["tag_name"].lstrip("vV"), "notes": (rel.get("body") or "")[:2000], "url": rel.get("html_url"),
        # A release built with make_release.sh carries a ready-to-run zip (with the page already built).
        "zip": assets[0]["browser_download_url"] if assets else None,
    }


def check(force: bool = False) -> dict:
    cur = settings.version()
    if not force and _cache["value"] and time.time() - _cache["at"] < 3600:
        return _cache["value"]
    out = {"current": cur, "latest": None, "available": False, "notes": "", "url": None, "error": None}
    try:
        rel = latest(settings.load()["update_repo"])
        if rel:
            out.update(latest=rel["version"], notes=rel["notes"], url=rel["url"], available=parse(rel["version"]) > parse(cur) and bool(rel["zip"]))
    except Exception as e:  # noqa: BLE001 - offline, no releases yet, private repository...
        out["error"] = f"{type(e).__name__}"
    _cache.update(at=time.time(), value=out)
    return out
