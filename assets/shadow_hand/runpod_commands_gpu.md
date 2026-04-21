# RunPod GPU Training Setup (MJX + SBX)

Use this guide for the MJX-accelerated training path (PPO/SAC on JAX via SBX). For CPU-only training see runpod_commands.md.

Pick a pod with CUDA 12.4 or newer and at least 24 GB VRAM (RTX 6000 Ada, RTX 4090, L40S, H100).

The Shadow Hand model has many mesh collision geoms plus the table, object, hole, and walls, so MJX constraint and contact tensors are heavy under vmap. Recommended envs per GPU:

| GPU VRAM                        | num-envs | Notes                                 |
|---------------------------------|----------|---------------------------------------|
| 80 GB (H100)                    | 2048     | default                               |
| 40–48 GB (A100, L40S, 6000 Ada) | 1024     | comfortable                           |
| 24 GB (4090)                    | 512      | if compilation OOMs, drop to 256      |

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
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export WANDB_MODE=disabled

If you hit OOM early and are confident about env size, you can switch back to XLA_PYTHON_CLIENT_PREALLOCATE=true with XLA_PYTHON_CLIENT_MEM_FRACTION=0.7. Leaving headroom matters for MJX because XLA sometimes needs extra scratch buffers during compilation of the vmapped step.

Do not set OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, or MKL_NUM_THREADS here — those only matter for the CPU path.

### Sanity check a short run first

Always do this before launching a full run. It compiles the step graph on the pod, which is where most problems (OOM, missing sensors, bad contact counts, etc.) surface:

    uv run python main.py train-grasp-mjx --num-envs 256 --total-timesteps 200000

Should complete in a couple minutes on any modern GPU and print non-zero reward terms. If it hangs more than 10 min or OOMs during compile, halve num-envs and retry — compile is the hard part; once it runs, it runs.

### Grasp + reorient (PPO)

    uv run python main.py train-grasp-mjx --num-envs 1024 --total-timesteps 30000000
    uv run python main.py train-reorient-mjx --num-envs 1024 --total-timesteps 400000000

### Peg + tactile (SAC)

SAC is gradient-heavy; do not blindly push num-envs here. At 1024 envs the default config already does ~32 gradient steps per env step.

    uv run python main.py train-peg-mjx --num-envs 1024 --total-timesteps 40000000
    uv run python main.py train-tactile-mjx --num-envs 1024 --total-timesteps 40000000

Bump up to num-envs 2048 only on an 80 GB H100 after the sanity check passes.

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
