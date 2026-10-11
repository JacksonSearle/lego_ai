#!/usr/bin/env bash
# Build the frontend (if needed) and serve the whole app on http://localhost:8000
# (a release download already contains the built page, so it needs no Node.js)
set -euo pipefail
cd "$(dirname "$0")"

if command -v npm >/dev/null 2>&1 && [ -d frontend/src ]; then
  if [ ! -d frontend/node_modules ]; then
    (cd frontend && npm install)
  fi
  if [ ! -f frontend/dist/index.html ] || [ -n "$(find frontend/src -newer frontend/dist/index.html -print -quit)" ]; then
    (cd frontend && npm run build)
  fi
elif [ ! -f frontend/dist/index.html ]; then
  echo "The page has not been built and Node.js is not installed. Download a release zip, or install Node.js." >&2
  exit 1
fi

cd backend
# HOST=0.0.0.0 makes this copy reachable from other computers. Do that only together with
# LEGO_TOKEN (a worker others train on) or LEGO_HOST=1 (a shared leaderboard): see docs/.
exec uv run uvicorn lego.server:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
