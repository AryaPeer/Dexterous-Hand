# RunPod GPU Training Setup (MJX + SBX)

Use this guide for the MJX-accelerated training path (PPO/SAC on JAX via SBX). For CPU-only training see runpod_commands.md.

Pick a pod with CUDA 12.4 or newer and at least 24 GB VRAM (RTX 6000 Ada, RTX 4090, L40S, H100).

The Shadow Hand model has many mesh collision geoms plus the table, object, hole, and walls, so MJX constraint and contact tensors are heavy under vmap. Recommended envs per GPU (validated empirically):

| GPU VRAM                        | grasp/peg num-envs | reorient num-envs | Notes                              |
|---------------------------------|--------------------|-------------------|------------------------------------|
| 80 GB (H100/H200)               | 2048               | 4096              | default                            |
| 40–48 GB (A100, L40S, 6000 Ada) | 1024               | 1024              | comfortable                        |
| 32 GB (RTX 5090)                | 768                | 768               | reorient 2048 OOMs (~50 GB I/O)    |
| 24 GB (4090)                    | 512                | 512               | sanity uses 256                    |

Reorient I/O args grow ~24 MB/env at JIT trace time, so anything below
~30 GB free can't fit 1024 envs. The 5090's 32 GB sits in the awkward
middle; 768 is the safe operating point.

If you see XLA scheduler warnings like "byte size of input/output arguments exceeds the base limit" followed by long hangs or OOM at JIT time, halve num-envs and retry. The warnings themselves are cosmetic; hangs and OOMs are not.


## 1. System setup

### Picking the pod

On RunPod the CUDA driver version is set by the pod's base image — you cannot install a newer driver from inside the container. When launching:

- Prefer templates named RunPod Pytorch 2.4 (CUDA 12.4) or newer, or the generic CUDA 12.4+ templates.
- Avoid the older CUDA 11.x community images.
- Verify after ssh'ing in:

      nvidia-smi | head -3
      # CUDA Version should be >= 12.4. Driver Version should be >= 550.

If you see a pytorch warning "driver is too old" while training, it is cosmetic in this codebase — pytorch is unused on GPU, MJX uses JAX. If you see a JAX crash about CUDA version mismatch, then you need a newer pod.

### Inside the pod

    apt-get update && apt-get install -y tmux git
    curl -LsSf https://astral.sh/uv/install.sh | sh


## 2. Clone and install

    cd ~
    git clone https://github.com/AryaPeer/Dexterous-Hand.git dexterous_hand
    cd dexterous_hand
    source $HOME/.local/bin/env
    uv sync --extra mjx

The mjx extra pulls in mujoco-mjx, sbx-rl, jax[cuda12], flax, optax, and chex.

Verify JAX sees the GPU before training:

    uv run python -c "import jax; print(jax.devices())"
    # expected: [CudaDevice(id=0)]


## 3. Train

    tmux new-session -s train

### GPU environment variables

    cd ~/dexterous_hand
    export CUDA_VISIBLE_DEVICES=0
    export JAX_PLATFORMS=cuda
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
    export WANDB_MODE=disabled

`PREALLOCATE=true` + `MEM_FRACTION=0.7` is the configuration that fixed
the `CUDA_ERROR_ILLEGAL_ADDRESS` from the original sanity bundle on
reorient — keep both. They also eliminate the long-tail XLA
fragmentation crashes that hit multi-hour SAC runs.

Do not set OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, or MKL_NUM_THREADS here — those only matter for the CPU path.

### Sanity check a short run first

Always do this before launching a full run. It compiles the step graph on the pod, which is where most problems (OOM, missing sensors, bad contact counts, etc.) surface:

    uv run python main.py train-grasp-mjx --num-envs 256 --total-timesteps 200000

Should complete in a couple minutes on any modern GPU and print non-zero reward terms. If it hangs more than 10 min or OOMs during compile, halve num-envs and retry — compile is the hard part; once it runs, it runs.

### Grasp + reorient (PPO)

Per-pod env counts after the round-3 5090 OOM finding (2048 envs of
reorient need ~50 GB of I/O args alone; the 5090's 32 GB doesn't fit):

| GPU       | grasp num-envs | reorient num-envs |
| --------- | -------------- | ----------------- |
| H100 80GB | 2048           | 4096              |
| A100/L40S | 1024           | 1024              |
| 5090 32GB | 768            | 768               |
| 4090 24GB | 512            | 512               |

    # full grasp (70 M, current default)
    uv run python main.py train-grasp-mjx --num-envs 768 --total-timesteps 70000000

    # reorient 180° target — see assets/shadow_hand/runpod_reorient_full.md
    uv run python main.py train-reorient-mjx \
        --num-envs 768 \
        --total-timesteps 200000000 \
        --curriculum-reference-timesteps 200000000

### Peg (SAC)

SAC is gradient-heavy and the audit-H1 fix bumped `gradient_steps` to
128 by default. The UTD ratio at 256 envs is now ~0.5 (was 0.03 before).
Do not push num-envs above what the GPU comfortably fits — gradient
cost per env-step is higher than for PPO.

    # full peg (60 M, current default)
    uv run python main.py train-peg-mjx --num-envs 256 --total-timesteps 60000000

Round-3 confirmed 256 envs on a 4090 produces stable training in 1.5 hr
per 1 M timesteps. Bumping to 512 on an A100 / 1024 on an H100 is fine
but watch `train/critic_loss` — high UTD with too many envs can
oscillate.

### Watcher + auto-stop

In a second terminal, start a watcher that copies the local runs/ into /workspace/runs/ once training exits, then stops the pod:

    tmux new-session -s watcher
    while pgrep -f "main.py" > /dev/null; do sleep 60; done && command cp -rf ~/dexterous_hand/runs/. /workspace/runs/ && echo "RUNPOD_POD_ID=$RUNPOD_POD_ID" && runpodctl stop pod "$RUNPOD_POD_ID"

Attach to the watcher at any time with tmux attach -t watcher (detach with Ctrl+b d).


## 4. Evaluate

Evaluation still runs on CPU MuJoCo for rendering; no GPU needed. Load the SBX checkpoint with the -mjx eval commands. Point --model-path at the saved final_model.zip and (recommended) --vec-normalize-path at the saved VecNormalize stats so observations are normalized the same way they were during training:

    uv run python main.py evaluate-grasp-mjx \
        --model-path runs/grasp_mjx_2048env_42/final_model.zip \
        --vec-normalize-path runs/grasp_mjx_2048env_42/vec_normalize.pkl

    uv run python main.py evaluate-peg-mjx \
        --model-path runs/peg_mjx_2048env_42/final_model.zip \
        --vec-normalize-path runs/peg_mjx_2048env_42/vec_normalize.pkl

    uv run python main.py evaluate-reorient-mjx \
        --model-path runs/reorient_mjx_2048env_42/final_model.zip \
        --vec-normalize-path runs/reorient_mjx_2048env_42/vec_normalize.pkl

    uv run python main.py evaluate-tactile-mjx \
        --tactile-model runs/tactile_mjx_2048env_42/final_model.zip \
        --baseline-model runs/peg_mjx_2048env_42/final_model.zip \
        --tactile-vec-normalize runs/tactile_mjx_2048env_42/vec_normalize.pkl \
        --baseline-vec-normalize runs/peg_mjx_2048env_42/vec_normalize.pkl
