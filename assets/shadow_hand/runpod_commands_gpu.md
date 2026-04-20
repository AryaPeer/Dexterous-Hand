# RunPod GPU Training Setup (MJX + SBX)

Use this guide for the MJX-accelerated training path (PPO/SAC on JAX via SBX). For CPU-only training see `runpod_commands.md`.

Pick a pod with **CUDA 12.x** and at least 24 GB VRAM (e.g. RTX 6000 Ada, RTX 4090, L40S, H100). 2048 parallel MJX envs fit in ~16–20 GB at default config.

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
uv sync --extra mjx
```

The `mjx` extra pulls in `mujoco-mjx`, `sbx-rl`, `jax[cuda12]`, `flax`, `optax`, and `chex`.

Verify JAX sees the GPU before training:

```bash
uv run python -c "import jax; print(jax.devices())"
# expected: [CudaDevice(id=0)]
```

## 3. Train

```bash
tmux new-session -s train
```

### GPU environment variables

```bash
cd ~/dexterous_hand
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=disabled
```

Do **not** set `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` / `MKL_NUM_THREADS` here — those only matter for the CPU path.

### Grasp + reorient

```bash
uv run python main.py train-grasp-mjx --num-envs 2048 --total-timesteps 30000000
uv run python main.py train-reorient-mjx --num-envs 2048 --total-timesteps 400000000
```

### Peg + tactile

```bash
uv run python main.py train-peg-mjx --num-envs 2048 --total-timesteps 40000000
uv run python main.py train-tactile-mjx --num-envs 2048 --total-timesteps 40000000
```

If VRAM is tight on the chosen pod (e.g. 24 GB card under load), drop `--num-envs` to 1024 or 512 — the configs scale linearly.

### Sanity check a short run first

```bash
uv run python main.py train-grasp-mjx --num-envs 256 --total-timesteps 200000
```

Should complete in a couple minutes on any modern GPU and print non-zero reward terms.

### Watcher + auto-stop

In a second terminal, start a watcher that copies the local `runs/` into `/workspace/runs/` once training exits, then stops the pod:

```bash
tmux new-session -s watcher
while pgrep -f "main.py" > /dev/null; do sleep 60; done && command cp -rf ~/dexterous_hand/runs/. /workspace/runs/ && echo "RUNPOD_POD_ID=$RUNPOD_POD_ID" && runpodctl stop pod "$RUNPOD_POD_ID"
```

Attach to the watcher at any time with `tmux attach -t watcher` (detach with `Ctrl+b d`).

## 4. Evaluate

Evaluation still runs on CPU MuJoCo for rendering; no GPU needed. Load the SBX checkpoint with the `-mjx` eval commands:

```bash
uv run python main.py evaluate-grasp-mjx --run-dir runs/grasp_mjx_2048env_42
uv run python main.py evaluate-peg-mjx --run-dir runs/peg_mjx_2048env_42
uv run python main.py evaluate-reorient-mjx --run-dir runs/reorient_mjx_2048env_42
uv run python main.py evaluate-tactile-mjx --run-dir runs/tactile_mjx_2048env_42
```
