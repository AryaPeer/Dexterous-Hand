# RunPod Training Setup

## 1. System setup

```bash
apt-get update && apt-get install -y tmux git
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Clone and install

```bash
cd ~
git clone https://github.com/AryaPeer/Dexterous-Hand.git
cd dexterous_hand
uv sync
```

## 3. Train

```bash
tmux new-session -s train
```

```bash
cd ~/dexterous_hand
export OPENBLAS_NUM_THREADS=32
export WANDB_MODE=disabled
```

Other tasks:

```bash
uv run python main.py train --n-envs 256 --total-timesteps 50000000
uv run python main.py train-reorient --n-envs 256 --total-timesteps 200000000
uv run python main.py train-peg --n-envs 256 --total-timesteps 100000000
uv run python main.py train-tactile --n-envs 256 --total-timesteps 100000000
```

In a new terminal, start a watcher that copies runs and stops the pod when training finishes:

```bash
tmux new-session -d -s watcher 'while pgrep -f "main.py" > /dev/null; do sleep 60; done && mkdir -p /workspace/runs && cp -r ~/dexterous_hand/runs/* /workspace/runs/ && echo "Done." && runpodctl stop pod'
```