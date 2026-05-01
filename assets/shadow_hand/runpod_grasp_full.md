# RunPod GPU — Grasp full training (70 M)

Targets the round-8 grasp full run. `RewardConfig.lift_target` was lowered
from 0.04 to 0.012 to match the empirical geometric plateau (7 cm cube +
fixed-z wrist caps delta-lift at ~1.2 cm); with that change the smooth
success gate `(lift_factor * contact_factor * speed_factor) >= 0.85`
becomes reachable, so `is_success` and the one-shot `success_bonus`
finally fire. The 5 M sanity (round-7, 2026-04-30) ran clean with
reward 25 → 1465 monotone climb, contacts 2.4–2.7, no NaN — same
infrastructure validated; round-8 only changes the success threshold
binding. No re-sanity; ship the full run.

## 1. GPU choice

| GPU            | $/hr   | num-envs | est. wall  | est. cost   | notes                      |
| -------------- | ------ | -------- | ---------- | ----------- | -------------------------- |
| **RTX 5090**   | $0.99  | 768      | ~28 hr     | **~$28**    | recommended                |
| RTX 4090       | $0.69  | 512      | ~42 hr     | ~$29        | cheaper $/hr but slower    |
| H100 SXM       | $2.99  | 2048     | ~11 hr     | ~$33        | fastest single pod         |

Recommendation: **RTX 5090 at 768 envs, 70 M timesteps.** ~$28 total,
~28 hr wallclock. Cost-neutral with the 4090 path; 5090 is the call
for $/hr predictability and faster turnaround.

**2048 envs is not validated on 5090** — the JIT-compiled batched reset
I/O for the Shadow Hand grasp env is borderline at 32 GB. 768 is the
safe operating point. If you hit OOM at 768, drop to 512.

Anything below 24 GB VRAM won't fit 512+ envs of the Shadow Hand model.

## 2. Pod prerequisites

- CUDA 12.4+ image (RunPod PyTorch 2.4 / CUDA 12.4 templates).
- Verify after ssh:
  ```
  nvidia-smi | head -3
  # CUDA Version >= 12.4, Driver Version >= 550
  ```

## 3. System setup (fresh pod)

```
apt-get update && apt-get install -y tmux git
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

## 4. Clone and install

```
cd ~
git clone -b cleanup-dead-code https://github.com/AryaPeer/Dexterous-Hand.git dexterous_hand
cd dexterous_hand
uv sync --extra mjx
mkdir -p runs
```

Verify JAX sees the GPU:

```
uv run python -c "import jax; print(jax.devices())"
# expected: [CudaDevice(id=0)]
```

## 5. Launch training

```
tmux new-session -s grasp
```

Inside tmux:

```
cd ~/dexterous_hand

export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export WANDB_MODE=disabled

uv run python main.py train-grasp-mjx \
    --num-envs 768 \
    --total-timesteps 70000000 \
    2>&1 | tee runs/grasp_full_stdout.log
```

If you're on RTX 4090 use `--num-envs 512`. On H100 SXM use
`--num-envs 2048` (validated headroom on 80 GB).

`PREALLOCATE=true` + `MEM_FRACTION=0.7` is the configuration that fixed
the `CUDA_ERROR_ILLEGAL_ADDRESS` from the original sanity bundle — keep
both.

Detach with `Ctrl+b d`, reattach later with `tmux attach -t grasp`.

## 6. Watcher + auto-stop

In a second tmux session:

```
tmux new-session -s watcher
```

Inside:

```
while pgrep -f "main.py train-" > /dev/null; do sleep 60; done \
  && cp -rf ~/dexterous_hand/runs/. /workspace/runs/ \
  && echo "RUNPOD_POD_ID=$RUNPOD_POD_ID" \
  && runpodctl stop pod "$RUNPOD_POD_ID"
```

Copies `runs/` to persistent `/workspace/runs/` once training exits, then
stops the pod so you stop paying. Detach with `Ctrl+b d`.

## 7. What to watch

The CSV is at `runs/grasp_mjx_768env_42/logs/progress.csv`. Tail it
from a third tmux pane:

```
tail -f runs/grasp_mjx_768env_42/logs/progress.csv
```

Healthy run, every ~few minutes:

- `train/metrics/object_height` rising. Targets:
  - ≥ 0.448 (1.2 cm delta-lift, the geometric plateau) by 5 M
    — this is the round-7 plateau, should be reached very early
  - ≥ 0.450 sustained by 10 M
- `train/metrics/num_finger_contacts` plateauing ≥ 2.5 throughout.
- `train/metrics/success_hold_steps` > 0 sometime past 5 M
  (this is the round-8 deliverable — round-7 was stuck at 0).
- `train/reward/success` non-zero at least once past 5 M
  (one-shot rising-edge bonus).
- `train/rollout/ep_len_mean` < 200 once successes are firing
  (C2 truncation: successful episodes terminate early via `is_success`).
- `train/metrics/nan_rate` < 0.01 throughout.
- `train/approx_kl` 0.01–0.05, `train/clip_fraction` 0.1–0.4
  (PPO healthy ranges).

If `metrics/object_height` is stuck below 0.44 after 20 M, kill the
run — the geometry plateau is broken. If `train/metrics/nan_rate` > 0.01
sustained, kill and check the fail modes in
`runpod_sanity_all_tasks.md` §9.

## 8. After training finishes

The model is saved automatically. Check:

```
ls /workspace/runs/grasp_mjx_768env_42/
# expected: best/  checkpoints/  final_model.zip  vec_normalize.pkl  logs/
```

Evaluate on CPU (no GPU needed):

```
uv run python main.py evaluate-grasp-mjx \
    --model-path /workspace/runs/grasp_mjx_768env_42/final_model.zip \
    --vec-normalize-path /workspace/runs/grasp_mjx_768env_42/vec_normalize.pkl
```

Pass bar at eval: `success_rate > 0.10`, `mean_ep_length < 200`
(early termination on success).

## 9. Pass / partial-pass

**Pass:** `eval/success_rate > 0.10`, episode length truncating below
200, `obj_height` > 0.448 sustained. Ship the policy.

**Partial pass** (reward grew, success still 0): the smooth gate's 0.85
threshold is the next thing to relax. Try lowering to 0.75 in
`rewards/{cpu,gpu}/grasp_reward.py:113` and re-running 5 M sanity.

**Fail** (reward stalls or contacts collapse): geometry is genuinely the
binding constraint. Cube revert is the round-9 escape hatch — restore
`small_cube` 4 cm in `scene_builder.py` (see `runpod_sanity_all_tasks.md`
§9). Different grasp dynamics — rerun sanity from scratch.

## 10. If you need to resume after a crash

`final_model.zip` is only written at the end. Mid-run state lives in
`runs/grasp_mjx_768env_42/checkpoints/` (saved every 500 K vec-steps).
SBX does not auto-resume; you would need to point a one-off script at
the latest checkpoint and call `model.learn(total_timesteps=remaining,
reset_num_timesteps=False)`. Easiest path is to restart from scratch if
the crash is early; checkpoint recovery is only worth it past ~25 M.

## 11. Extending past 70 M (resume from final_model)

If the 70 M run finishes and the policy is "nearly there" — `eval/success_rate`
> 0 but below your bar, or reward still trending up at the final eval — you
do **not** need to relaunch from scratch. SBX policies pickle their full
optimizer state, and `model.learn(reset_num_timesteps=False)` continues
the run as if it had simply been longer.

A wired CLI is provided. To add 30 M on top of the existing 70 M:

```
uv run python main.py resume-grasp-mjx \
    --model-path /workspace/runs/grasp_mjx_768env_42/final_model.zip \
    --vec-normalize-path /workspace/runs/grasp_mjx_768env_42/vec_normalize.pkl \
    --additional-timesteps 30000000 \
    --num-envs 768 \
    --seed 42
```

Output lands in `<input_dir>_resumed/` by default (override with
`--output-dir`). Three things that matter under the hood:

1. **`reset_num_timesteps=False`** — without this, `model.num_timesteps`
   resets to 0 and any timestep-based callbacks (curriculum, eval cadence)
   re-fire from scratch. Grasp has no curriculum so this is mostly cosmetic
   here, but make a habit of always setting it on resume.
2. **Reload `vec_normalize.pkl`** into the new env via `VecNormalize.load`.
   Otherwise observation running stats reset → the policy sees a different
   input distribution than the one it trained on → you can lose 5-10 M
   timesteps of progress to re-stabilization.
3. **Resume `total_timesteps` is the *additional* budget**, not the
   cumulative target. Pass `30_000_000` to add 30 M on top.

Cost: 30 M resume ≈ 12 hr / $12 on 5090. Adding 30 M to a 70 M base is
strictly cheaper than retraining 100 M from scratch ($40), and you keep
the 70 M of learning already in the policy.

## 12. Cost summary

Single-pod cost: ~$28 on 5090. If grasp completes successfully, the
round-8 program looks like:

| run                     | cost   |
| ----------------------- | ------ |
| grasp full (this doc)   | ~$28   |
| peg sanity              | ~$1    |
| peg full 150 M (post-sanity, see `runpod_peg_full.md`) | ~$42 |
| reorient 200 M (in flight) | ~$80 |
| **program total**       | **~$151** |
