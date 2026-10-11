#!/usr/bin/env bash
# Build a release zip that playtesters can download and double-click.
#
#   scripts/make_release.sh            builds dist-release/lego-ai-<VERSION>.zip
#
# It contains the app's source, the already-built page (so no Node.js is needed) and the
# launchers. Publish it as a GitHub release whose tag is the version, with the zip attached:
# every installed copy then updates itself the next time it starts.
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION="$(tr -d '[:space:]' < VERSION)"
NAME="lego-ai-$VERSION"
OUT="dist-release"

(cd frontend && [ -d node_modules ] || npm install)
(cd frontend && npm run build)

rm -rf "$OUT/$NAME" "$OUT/$NAME.zip"
mkdir -p "$OUT/$NAME"
# Everything tracked by git except backend/data (so no personal data, runs, settings or
# custom blocks), plus the built page.
git ls-files -z | while IFS= read -r -d '' f; do
  case "$f" in backend/data/*) continue ;; esac   # your own custom blocks stay yours
  mkdir -p "$OUT/$NAME/$(dirname "$f")"
  cp "$f" "$OUT/$NAME/$f"
done
for f in VERSION launch.py "Lego AI.command" "Lego AI.bat"; do cp "$f" "$OUT/$NAME/$f"; done
mkdir -p "$OUT/$NAME/frontend" && cp -R frontend/dist "$OUT/$NAME/frontend/dist"
chmod +x "$OUT/$NAME/Lego AI.command" "$OUT/$NAME/run.sh"
(cd "$OUT" && zip -qr "$NAME.zip" "$NAME")
rm -rf "$OUT/$NAME"

echo "Built $OUT/$NAME.zip ($(du -h "$OUT/$NAME.zip" | cut -f1))"
echo
echo "To publish it (makes it public to everyone who can see the repository):"
echo "  1. Commit and push, so the code on GitHub matches."
echo "  2. gh release create v$VERSION $OUT/$NAME.zip --title \"Lego AI $VERSION\" --notes \"What changed\""
echo "     or: github.com → Releases → Draft a new release → tag v$VERSION → attach the zip."
echo "For the next release, raise the number in the VERSION file first."
