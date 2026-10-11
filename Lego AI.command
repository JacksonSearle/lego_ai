#!/bin/bash
# Double-click to start Lego AI on a Mac. Keeps itself up to date.
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "First start: installing uv, the tool that manages Python for Lego AI…"
  curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "Could not install uv. See https://docs.astral.sh/uv/"; read -r -p "Press Return to close."; exit 1; }
fi
uv run --no-project --python 3.12 python launch.py
read -r -p "Lego AI has stopped. Press Return to close this window."
