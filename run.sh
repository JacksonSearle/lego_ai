#!/usr/bin/env bash
# Build the frontend (if needed) and serve the whole app on http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d frontend/node_modules ]; then
  (cd frontend && npm install)
fi
if [ ! -f frontend/dist/index.html ] || [ -n "$(find frontend/src -newer frontend/dist/index.html -print -quit)" ]; then
  (cd frontend && npm run build)
fi

cd backend
exec uv run uvicorn lego.server:app --host 127.0.0.1 --port "${PORT:-8000}"
