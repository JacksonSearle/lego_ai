# Hosting a shared leaderboard

One person (the host) runs a copy of Lego AI in host mode. Everyone else runs the normal
app on their own computer, trains there, and submits to the host.

## The design in one paragraph

**The host never trusts a score it is sent.** A submission is a run's *config*: the
complete recipe (dataset, blocks, every training setting, the seed). No weights and no
numbers that count. The host checks the recipe against the track's limits, trains it
from scratch on its own computer on the CPU, and lists the score it measured there.
That answers three questions at once:

- *Is the result real?* Yes, the host produced it.
- *Is it fair?* Every entry was measured on the same machine, so nobody wins by owning a
  bigger GPU, and time limits mean the same thing for everyone.
- *Is it reproducible?* Training in this app is deterministic on the CPU, so the same
  config gives the same score every time the host runs it. Any player can press Open on
  an entry to load its recipe and retrain it too.

The submitter's own score is shown next to the host's. On identical hardware and library
versions they match to the last digit; across different computers they usually differ in
the third or fourth decimal place.

## Tracks

A track is a leaderboard with limits. Limits keep entries quick to check and make it a
contest of ideas, not hardware. Each track has: a dataset, a maximum number of weights,
epochs and training examples, a time limit (measured on the host), and whether custom
blocks are allowed. Four tracks are created the first time the host starts; edit them on
the Host tab.

Checking cost is what limits a leaderboard. With the default tracks an entry takes from a
few seconds to a few minutes on a laptop, so one computer can serve a class or a club.

**Custom blocks are code.** Checking an entry that contains one runs that Python on the
host. Tracks refuse them by default. If you allow them, the Host tab shows the code and
asks before running it. Only do this for people you know.

## Running the host

```bash
LEGO_HOST=1 LEGO_HOST_NAME="Room 12 leaderboard" LEGO_ADMIN_TOKEN="choose-a-long-secret" HOST=0.0.0.0 ./run.sh
```

- `LEGO_HOST=1` turns on the public `/host/...` endpoints and the **Host** tab.
- `HOST=0.0.0.0` lets other computers connect.
- In host mode the rest of the app (`/api/...`: training, your runs, custom blocks) is
  locked behind an access token, because other people can now reach this computer. When
  the host starts it prints a line like `http://localhost:8000/?token=...`. Open that
  address once in your own browser and you are in; the token is remembered. Set your own
  with `LEGO_TOKEN="..."`, otherwise one is generated and kept in
  `backend/data/host/access_token`.
- `LEGO_ADMIN_TOKEN` is optional: a second secret that unlocks only the Host tab's actions
  (queue, tracks, feedback), for a helper who should not get the whole app.

Open the app on the host with that address and go to the **Host** tab:

- **Submissions**: the queue. *Check* trains one entry; *Check all waiting* works through
  everything without custom code, oldest first, one at a time. *Reject* removes an entry
  with a reason the submitter sees.
- **Tracks**: add, edit and remove tracks.
- **Feedback**: messages from players' Feedback buttons.

Everything is stored in `backend/data/host/` as plain JSON. Back up that folder.

## Letting players reach it

Players enter the host's address under Leaderboard → Shared. Options, simplest first:

| Where the players are | What to do | Address players enter |
|---|---|---|
| Same room or Wi-Fi | Nothing more. Find the host's local IP address. | `http://192.168.x.x:8000` |
| Anywhere, host stays on your computer | Run a tunnel, for example `cloudflared tunnel --url http://localhost:8000` (free, no account for quick tunnels) or Tailscale Funnel. | the `https://...` address the tunnel prints |
| Anywhere, always on | Rent a small server (2 CPU cores and 4 GB of memory is enough for the default tracks), install `uv`, copy a release zip there and start it as above, behind a reverse proxy with HTTPS such as Caddy. | `https://your-domain` |

Notes:

- A tunnel address from a free quick tunnel changes each time you start it.
- Use HTTPS for anything beyond a local network. The tunnel options provide it.
- The built-in protections are modest: size limits, a cap on waiting submissions per
  name, duplicate detection, and no code execution without your say-so. It is meant for a
  community you know, not the open internet at scale.
- Players should run the same release as the host. A different PyTorch version can change
  scores slightly, and blocks added in a newer version do not exist on an older host.

## What is not built

- Accounts. A player is a name they type. Two people can pick the same name.
- Automatic checking without the host pressing a button. *Check all waiting* is one click;
  leaving it fully unattended is a deliberate non-feature while custom code can exist.
- Several hosts sharing one queue.
