# RunPod GPU — Reorient training (200 M, 180° final stage)

Targets the 180° reorient stage, the IsaacGymEnvs `ShadowHand` reference
benchmark target. The cube visibly flips over — visually unmistakable as
multi-finger coordination. **The 180° goal is the end-state for this
project; we are not extending to full SO(3).** §11 documents the round-9
resume path that fixes the H2 drop-cliff and lets the in-flight 200 M
run converge on a stable 180° policy.

Reorient is the only task already verified clean (3 M sanity on
2026-04-26 hit no NaN / no `CUDA_ERROR_ILLEGAL_ADDRESS`). Ship the
200 M run without re-sanity. Peg + grasp ship separately after their
own sanity bundle.

## 1. GPU choice

| GPU            | $/hr   | num-envs | est. wall  | est. cost   | notes                      |
| -------------- | ------ | -------- | ---------- | ----------- | -------------------------- |
| **RTX 5090**   | $0.99  | 768      | ~80 hr     | **~$80**    | recommended                |
| RTX 4090       | $0.69  | 512      | ~120 hr    | ~$80        | cheaper $/hr but slower    |
| H100 SXM       | $2.99  | 4096     | ~30 hr     | ~$90        | fastest single pod         |
| H200 SXM       | $3.99  | 4096     | ~25 hr     | ~$100       | only if H100 SXM unavailable |

Recommendation: **RTX 5090 at 768 envs, 200 M timesteps.** ~$80 total,
~80 hr wallclock, achieves stage-2 (180°) reorient. At 200 M timesteps
the GPU choice is roughly cost-neutral; 5090 is the call for $/hr
predictability.

**Important: RTX 5090 cannot fit 2048 envs of reorient.** The JIT-compiled
batched reset I/O args alone need ~50 GB; the 5090's 32 GB doesn't
have headroom. 768 envs is the safe operating point. If you hit OOM
even at 768 envs (shouldn't with `MEM_FRACTION=0.7`), drop to 512.

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
tmux new-session -s reorient
```

Inside tmux:

```
cd ~/dexterous_hand

export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export WANDB_MODE=disabled

uv run python main.py train-reorient-mjx \
    --num-envs 768 \
    --total-timesteps 200000000 \
    2>&1 | tee runs/reorient_180deg_stdout.log
```

The default `curriculum_reference_timesteps` on `MjxReorientTrainConfig`
is now `200_000_000` (matches the default `total_timesteps`), so the
curriculum stages fire at their literal step thresholds — no override
needed for the recommended 200 M run.

If you're on RTX 4090 use `--num-envs 512`. On H100 SXM use
`--num-envs 4096`. Pass `--curriculum-reference-timesteps <N>` only if
you change the total timestep budget and want to scale stage advance
proportionally.

`PREALLOCATE=true` + `MEM_FRACTION=0.7` is the configuration that fixed
the `CUDA_ERROR_ILLEGAL_ADDRESS` from the original sanity bundle — keep
both.

Detach with `Ctrl+b d`, reattach later with `tmux attach -t reorient`.

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

The CSV is at `runs/reorient_mjx_768env_42/logs/progress.csv`. Tail it
from a third tmux pane:

```
tail -f runs/reorient_mjx_768env_42/logs/progress.csv
```

Healthy run, every ~few minutes:

- `train/metrics/angular_distance` trending DOWN. Targets:
  - ≤ 0.5 rad sustained by ~30 M (stage 0 + entering stage 1 cleared)
  - ≤ 1.0 rad sustained by ~100 M (stage 1 cleared, in stage 2)
- `train/reward/angular_progress` averaging > 0.
- `train/metrics/num_finger_contacts` ≥ 0.5 throughout.
- `train/metrics/success_steps` rising. **Stop-the-pod milestone:**
  `≥ 0.20` (20% success rate at stage 2's 180° target) by 180 M.

If `metrics/angular_distance` drifts UP past 1.8 rad and stays there, or
if the log shows `CUDA_ERROR_ILLEGAL_ADDRESS`, kill the run and check
the fail modes in `runpod_sanity_all_tasks.md` §9.

## 8. After training finishes

The model is saved automatically. Check:

```
ls /workspace/runs/reorient_mjx_768env_42/
# expected: best/  checkpoints/  final_model.zip  vec_normalize.pkl  logs/
```

Evaluate on CPU (no GPU needed):

```
uv run python main.py evaluate-reorient-mjx \
    --model-path /workspace/runs/reorient_mjx_768env_42/final_model.zip \
    --vec-normalize-path /workspace/runs/reorient_mjx_768env_42/vec_normalize.pkl
```

## 9. Curriculum reference

`MjxReorientTrainConfig.curriculum_stages` (in `dexterous_hand/config.py`)
caps at 180° — the SO(3) stage was removed because we don't intend to
train past 180° in this run, and leaving it in would have wasted budget
on an objective that can't converge in the remaining timesteps.

`scale_stage_starts` computes `scaled_threshold = base_threshold *
(total_timesteps / reference_total_timesteps)`. With the new default
`reference_total_timesteps = 200_000_000` matching the recommended
`total_timesteps = 200_000_000`, scaled thresholds equal the base.

| stage | base threshold | scaled @ 200 M total | max angle | time-in-stage |
| ----- | -------------- | -------------------- | --------- | ------------- |
| 0     | 0              | 0                    | 30°       | 20 M          |
| 1     | 20 M           | 20 M                 | 90°       | 40 M          |
| 2     | 60 M           | 60 M                 | 180°      | **140 M**     |

Stage 2 gets all the trailing budget — well above the ~100–150 M rule
of thumb for 180° convergence. If you reduce `--total-timesteps`,
`scale_stage_starts` shrinks the earlier-stage budgets first, so stage
2 retains the bulk of the run.

## 10. If you need to resume after a crash

`final_model.zip` is only written at the end. Mid-run state lives in
`runs/reorient_mjx_768env_42/checkpoints/` (saved every 500 K vec-steps).
Use the wired CLI to point at the latest checkpoint:

```
uv run python main.py resume-reorient-mjx \
    --model-path /workspace/runs/reorient_mjx_768env_42/checkpoints/<latest>.zip \
    --vec-normalize-path /workspace/runs/reorient_mjx_768env_42/vec_normalize.pkl \
    --additional-timesteps <remaining> \
    --num-envs 768 \
    --seed 42
```

`vec_normalize.pkl` is only saved at end-of-run, so for crash recovery
you may need to reuse the one from a prior completed run on the same
config or restart from scratch. Checkpoint recovery is only worth the
hassle past ~50 M.

## 11. Round-9 H2 resume — recover a stable 180° policy

**Why this section exists.** The in-flight 200 M run hit `success_steps`
≈ 0.135 around 72 M (early stage 2), then partially regressed by 121 M
to ~0.083 with `angular_distance` drifting from 1.05 → 1.40 rad (~80°).
PPO health is pristine throughout (`approx_kl` 0.006, `value_loss` 0.07,
`explained_variance` 0.987, `nan_rate` 0), so this is not an optimizer
failure — it's a reward-landscape failure mode flagged in `gpu_audit.md`
as **H2: drop-penalty cliff**. The reward function had a binary
`where(dropped, drop_penalty=-20, 0)` term. As stage 2 introduced
180° goals at 60 M, aggressive rotations occasionally dropped the cube,
each drop fired a -100 reward (after 5× weight), and the policy
converged to a "hold cube safely, rotate slowly" local minimum that
minimizes drop frequency at the cost of solving the goal.

**What the round-9 fix does.** `dexterous_hand/rewards/{cpu,gpu}/reorient_reward.py`
now takes a continuous `drop_factor` instead of a binary `dropped` flag.
The env (`envs/{cpu,gpu}/reorient_env.py`) computes a clamped smoothstep
over the `drop_height_offset` margin: 0 when cube is at safe palm height,
1 when it crosses the drop threshold, smooth ramp in between. The reward
function applies it as a direct multiplier on `drop_penalty`. Net effect:
the policy now receives **gradient warning** as the cube descends toward
the drop boundary, instead of a binary impulse only at the cliff. This
breaks the "hover safely" minimum because the policy can lean back from
the edge without paying the full cliff cost — the gradient says "you're
getting low, climb a bit" rather than "cube is fine until it's not".

The binary `dropped` flag is still used inside the env for episode
termination (separate from reward shaping). Tests `tests/cpu/test_rewards.py`,
`tests/gpu/test_rewards.py`, and `tests/test_reward_parity.py` were
updated to exercise the smooth path; all 153 tests green.

**Resume procedure.** Wait for the 200 M run to finish naturally (don't
kill — the policy state is recoverable, the gradient signal just
shifted). Then:

1. Confirm the round-9 commit is live on the pod's branch:
   ```
   cd ~/dexterous_hand && git pull && uv run pytest -q
   ```
   Expect 153 passed, 3 skipped.

2. Resume from the saved 200 M checkpoint:
   ```
   uv run python main.py resume-reorient-mjx \
       --model-path /workspace/runs/reorient_mjx_768env_42/final_model.zip \
       --vec-normalize-path /workspace/runs/reorient_mjx_768env_42/vec_normalize.pkl \
       --additional-timesteps 50000000 \
       --num-envs 768 \
       --seed 42
   ```

   The resume uses `reset_num_timesteps=False` (cumulative timesteps
   preserved) and `VecNormalize.load` (obs running stats preserved).
   Curriculum is anchored at the original schedule; cumulative timesteps
   start at 200 M, so the resume is pure stage-2 horizon at 180°.

**Important reload caveat.** The reward function changed, so the policy's
value function will need a few M timesteps to refit. Expect:
- `value_loss` to spike briefly (5-10× pre-resume baseline) then settle
- `explained_variance` to drop transiently (down to ~0.7) then recover
- `success_steps` and `angular_distance` may dip for the first ~5 M
  before improving as the policy escapes the "hover" minimum

If `success_steps` doesn't recover above the pre-resume 0.083 by 20 M
of resume (= 220 M cumulative), kill — the H2 fix wasn't enough and
round-10 needs additional intervention (likely curriculum relaxation).

**Pass bar at 250 M cumulative (= 50 M resume completed):**
- `success_steps` ≥ 0.20 (the original 200 M target)
- `angular_distance` ≤ 1.0 rad sustained
- `eval/success_rate` > 0.10

Cost: ~$50 for the 50 M resume on 5090. Total reorient program: ~$80
(in-flight 200 M) + ~$50 (round-9 resume) = ~$130.

**No SO(3) extension.** The 180° policy is the deliverable; do not
re-add the `(120M, math.pi)` stage to `curriculum_stages`. If the round-9
resume hits the pass bar, ship the policy and end the program.
