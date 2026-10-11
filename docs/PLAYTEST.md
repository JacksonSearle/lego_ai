# Giving Lego AI to playtesters

Playtesters get a zip, double-click a launcher, and from then on their copy updates itself
and has a Feedback button that reaches you.

## How it works

- **A release** is a zip built by `scripts/make_release.sh`. It contains the source, the
  already-built page (no Node.js needed) and two launchers.
- **The launchers** are `Lego AI.command` (Mac) and `Lego AI.bat` (Windows). On first start
  they install `uv`, which then downloads the right Python and PyTorch (a few minutes and
  about 2 GB). After that they run `launch.py`.
- **Updating**: every start, `launch.py` asks GitHub for the newest release of the
  repository. If its version is higher than the local `VERSION` file, it downloads that
  release's zip and copies it over the program files. `backend/data` (models, datasets,
  progress, settings) is never touched. While the app is open, a banner appears when a
  newer release exists and asks the user to restart.
- **Feedback**: the 💬 button in the top bar. The message is saved in the tester's
  `backend/data/feedback.jsonl` and, if their copy is connected to a shared leaderboard
  host, sent there. You read it on the host's **Host → Feedback** tab.

## Publishing a release

1. Raise the number in `VERSION` (for example `0.11.0` to `0.12.0`). Commit and push.
2. `scripts/make_release.sh` builds `dist-release/lego-ai-<version>.zip`.
3. On GitHub: Releases → Draft a new release → tag `v<version>` → attach the zip → Publish.
   With the GitHub CLI: `gh release create v0.12.0 dist-release/lego-ai-0.12.0.zip --title "Lego AI 0.12.0" --notes "What changed"`.

Send testers the zip from the release page. They unzip it anywhere and double-click the
launcher. Later releases reach them automatically.

Requirements and limits:

- **The repository's releases must be public** for the update check to work without a
  login. A private repository returns "not found" to the app, which then simply starts the
  installed version.
- To make feedback reach you with no setup on the tester's side, start a host (see
  `HOSTING.md`) and have testers enter its address once under Leaderboard → Shared. To ship
  it preconfigured, set `LEGO_SERVER_URL` in the launcher or put the address in
  `DEFAULTS["server_url"]` in `backend/lego/settings.py` before building the release.
- **These are scripts, not signed applications.** On a Mac the first double-click may be
  blocked: right-click the file → Open → Open. On Windows, SmartScreen may say "Windows
  protected your PC": More info → Run anyway. Removing those warnings needs paid code
  signing (an Apple Developer ID and a Windows certificate) and packaging as a real
  application, which is a separate project.
- The Windows launcher has not been tested on a Windows machine yet.
- Developers working from a git checkout are never auto-updated; use `git pull`.
