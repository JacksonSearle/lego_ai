"""Start Lego AI, updating it first if a newer release exists.

Run by the double-click launchers ("Lego AI.command" on a Mac, "Lego AI.bat" on Windows).
It uses only Python's standard library, so it works before anything else is installed.

    1. If this folder is a release download (not a developer's git checkout), look for a newer
       release and install it over this folder. Your data (backend/data) is never touched.
    2. Start the server and open the app in the browser.

Environment: PORT (default 8000), LEGO_NO_UPDATE=1 to skip step 1.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEEP = {"backend/data", "backend/.venv", ".git"}  # never replaced by an update


def version() -> str:
    try:
        return (ROOT / "VERSION").read_text().strip()
    except OSError:
        return "0"


def parse(v: str) -> tuple:
    return tuple(int(p) if p.isdigit() else 0 for p in v.strip().lstrip("vV").split(".")[:4])


def repo() -> str:
    try:
        return json.loads((ROOT / "backend" / "data" / "settings.json").read_text()).get("update_repo") or "JacksonSearle/lego_ai"
    except (OSError, ValueError):
        return "JacksonSearle/lego_ai"


def update():
    if os.environ.get("LEGO_NO_UPDATE") == "1":
        return
    if (ROOT / ".git").exists():
        print("Developer checkout (git): not updating automatically. Use git pull.")
        return
    url = os.environ.get("LEGO_UPDATE_URL") or f"https://api.github.com/repos/{repo()}/releases/latest"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "lego-ai"})
        with urllib.request.urlopen(req, timeout=8) as r:
            rel = json.loads(r.read())
        latest = str(rel.get("tag_name") or "").lstrip("vV")
        assets = [a for a in rel.get("assets") or [] if str(a.get("name", "")).endswith(".zip")]
        if not latest or parse(latest) <= parse(version()) or not assets:
            print(f"Lego AI {version()} is up to date.")
            return
        print(f"Updating Lego AI {version()} → {latest}…")
        with urllib.request.urlopen(assets[0]["browser_download_url"], timeout=120) as r:
            blob = r.read()
    except Exception as e:  # noqa: BLE001 - offline or no releases: start what we have
        print(f"Could not check for updates ({type(e).__name__}). Starting the installed version.")
        return
    try:
        with tempfile.TemporaryDirectory() as tmp:
            zipfile.ZipFile(io.BytesIO(blob)).extractall(tmp)
            # The zip holds one top-level folder with the app inside.
            tops = [p for p in Path(tmp).iterdir() if p.is_dir()]
            src = tops[0] if len(tops) == 1 and (tops[0] / "VERSION").exists() else Path(tmp)
            if not (src / "backend" / "lego").exists() or not (src / "frontend" / "dist" / "index.html").exists():
                raise ValueError("the download is not a complete Lego AI release")
            for item in src.rglob("*"):
                rel_path = item.relative_to(src).as_posix()
                if any(rel_path == k or rel_path.startswith(k + "/") for k in KEEP):
                    continue
                target = ROOT / rel_path
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)
            # Remove program files that no longer exist in the new version.
            for folder in ("backend/lego", "frontend/dist"):
                for old in (ROOT / folder).rglob("*"):
                    if old.is_file() and "__pycache__" not in old.parts and not (src / old.relative_to(ROOT)).exists():
                        old.unlink()
        print(f"Updated to {latest}.")
    except Exception as e:  # noqa: BLE001
        print(f"The update could not be installed ({e}). Starting the installed version.")


def uv() -> str:
    found = shutil.which("uv")
    if found:
        return found
    for cand in (Path.home() / ".local" / "bin" / "uv", Path.home() / ".cargo" / "bin" / "uv", Path.home() / ".local" / "bin" / "uv.exe"):
        if cand.exists():
            return str(cand)
    sys.exit("Lego AI needs “uv” to manage Python. Install it from https://docs.astral.sh/uv/ and start Lego AI again.")


def main():
    update()
    if not (ROOT / "frontend" / "dist" / "index.html").exists():
        sys.exit("This copy has no built page (frontend/dist). Download a release zip, or run ./run.sh with Node.js installed.")
    port = os.environ.get("PORT", "8000")
    url = f"http://localhost:{port}"
    print(f"Starting Lego AI {version()} at {url}  (the first start downloads PyTorch, which can take several minutes)")
    proc = subprocess.Popen([uv(), "run", "uvicorn", "lego.server:app", "--host", os.environ.get("HOST", "127.0.0.1"), "--port", port], cwd=ROOT / "backend")
    try:
        for _ in range(1800):  # wait for the server to answer, then open the browser
            if proc.poll() is not None:
                sys.exit("Lego AI stopped while starting. The messages above say why.")
            try:
                urllib.request.urlopen(url + "/api/version", timeout=1)
                break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        webbrowser.open(url)
        print("Lego AI is running. Keep this window open; close it to quit.")
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
