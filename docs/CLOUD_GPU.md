# Training on a cloud GPU

Lego AI can send a training run to another computer, a **worker**, and bring the results
back. A worker is the same app, started with an access token, on any machine with a GPU:
a rented cloud machine, a desktop at home, a lab computer.

Your laptop keeps the Builder, the live charts and the leaderboard. The worker only does
the training.

## 1. Start a worker

On the GPU machine (Linux with an NVIDIA GPU is the usual case):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh       # once: installs uv
# copy a Lego AI release zip to the machine and unzip it, then inside the folder:
LEGO_TOKEN="choose-a-long-random-secret" HOST=0.0.0.0 PORT=8000 ./run.sh
```

`LEGO_TOKEN` is what keeps strangers out: every request must carry it. Use a long random
string and treat it like a password.

## 2. Connect to it

**Safest: an SSH tunnel.** The worker stays invisible to the internet.

```bash
ssh -N -L 8010:localhost:8000 user@the-gpu-machine      # leave this running on your laptop
```

In the app: Train → Hardware → **Add another computer** → address `http://localhost:8010`
and the token. (With a tunnel you can even start the worker with `HOST=127.0.0.1`.)

**Direct:** if the provider gives the machine a public address or a proxied URL, enter
that address. Use this only with a strong token, and prefer an `https://` address.

## 3. Train

Pick the worker under **Where to train** and press Train. The run appears on your Train
page with live charts. When it ends, the metrics and weights are copied to your computer
and the run sits on your leaderboard like any other, marked with a cloud.

- Datasets you imported are sent to the worker automatically the first time.
- Custom blocks travel inside the config.
- Works for single runs. Optuna, seeds, several datasets and fine-tuning still run locally.
- A run trained on a GPU is not bit-for-bit reproducible, and will differ slightly from
  the same run on a CPU.

## Where to get a GPU

- **A computer you already have access to**: a gaming desktop, a university cluster node.
- **Rent by the hour**: GPU clouds and marketplaces rent machines with a single modern
  GPU. Prices vary a lot between providers and GPU models, so compare. The Hardware tab's
  size advisor tells you roughly how long an epoch would take on common GPUs.
- **Free notebooks** (Google Colab, Kaggle) can run a worker too, but they have no public
  address: you need a tunnel tool from inside the notebook, and sessions end after a few
  hours.

**Rented machines bill while they are on, used or not. Shut the machine down when you
finish.** Lego AI does not start or stop cloud machines for you: that would need your
provider account and payment details, which the app deliberately never handles.
