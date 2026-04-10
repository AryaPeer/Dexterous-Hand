# RunPod Training Setup

## 1. System setup

```bash
apt-get update && apt-get install -y tmux git
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Clone and install

```bash
cd ~
git clone https://github.com/AryaPeer/Dexterous-Hand.git dexterous_hand
cd dexterous_hand
source $HOME/.local/bin/env
uv sync
```

## 3. Train

```bash
tmux new-session -s train
```

```bash
cd ~/dexterous_hand
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=disabled
```

Other tasks:

```bash
uv run python main.py train --n-envs 32 --total-timesteps 100000000
uv run python main.py train --n-envs 32 --total-timesteps 150000000
uv run python main.py train-reorient --n-envs 32 --total-timesteps 400000000
uv run python main.py train-peg --n-envs 32 --total-timesteps 150000000
uv run python main.py train-tactile --n-envs 32 --total-timesteps 150000000 --variant both
uv run python main.py train-tactile --n-envs 32 --total-timesteps 150000000 --variant tactile
```

In a new terminal, start a persistent watcher session that copies the full local `runs` content into `/workspace/runs/` (overwriting only matching files/folders) and then stops the pod:

```bash
tmux new-session -s watcher
while pgrep -f "main.py" > /dev/null; do sleep 60; done && command cp -rf ~/dexterous_hand/runs/. /workspace/runs/ && echo "RUNPOD_POD_ID=$RUNPOD_POD_ID" && runpodctl stop pod "$RUNPOD_POD_ID"
```

Attach to the watcher at any time with `tmux attach -t watcher` (detach with `Ctrl+b d`).
