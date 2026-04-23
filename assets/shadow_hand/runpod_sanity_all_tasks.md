# RunPod GPU — 2M sanity on cleanup-dead-code

All three tasks (grasp, reorient, peg) need a fresh 2M sanity. Every reward
changed on this branch: action-rate deleted everywhere, action_penalty
standardized to IsaacGymEnvs's `-0.0002·||a||²`, peg drop decoupled from
regrasp, reorient `orientation_tracking`+`orientation_success` collapsed to
one `orientation` term, domain randomization on by default, initial-state
noise bumped to Dactyl scale. Peg's prior 2M pass on main doesn't carry over.

~1.5–2 hr total, ~$0.70 on a 4090.

## 1. One-time: clone + sync on a fresh pod

```
cd ~ && rm -rf dexterous_hand && git clone -b cleanup-dead-code \
    https://github.com/AryaPeer/Dexterous-Hand.git dexterous_hand
cd ~/dexterous_hand && uv sync --extra mjx
mkdir -p runs
```

## 2. Start a tmux session and paste the training block

```
tmux new-session -s sanity
```

Then inside that tmux session:

```
cd ~/dexterous_hand

export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled

(
  echo "==================== train-grasp-mjx ===================="
  uv run python main.py train-grasp-mjx --num-envs 256 --total-timesteps 2000000 || true
  echo ""

  echo "==================== train-peg-mjx ===================="
  uv run python main.py train-peg-mjx --num-envs 256 --total-timesteps 2000000 || true
  echo ""

  echo "==================== train-reorient-mjx ===================="
  uv run python main.py train-reorient-mjx \
      --num-envs 256 \
      --total-timesteps 2000000 \
      --curriculum-reference-timesteps 20000000 || true
  echo ""
) 2>&1 | tee runs/sanity_stdout.log
```

Detach any time with `Ctrl+b d`. Reattach later with `tmux attach -t sanity`.

## 3. In a second tmux session, paste the watcher

Open a new session so the watcher runs alongside the training:

```
tmux new-session -s watcher
```

Inside that session:

```
while pgrep -f "main.py train-" > /dev/null; do sleep 60; done \
  && cp -rf ~/dexterous_hand/runs/. /workspace/runs/ \
  && echo "RUNPOD_POD_ID=$RUNPOD_POD_ID" \
  && runpodctl stop pod "$RUNPOD_POD_ID"
```

This copies `runs/` to `/workspace/runs/` once all three trainings finish,
then stops the pod. Detach with `Ctrl+b d`.

## 4. After the pod stops, paste back the log

```
/workspace/runs/sanity_stdout.log
```

Or just the tail if it's huge:

```
tail -n 500 /workspace/runs/sanity_stdout.log
```

## 5. Expected timeline (4090)

| task      | wall time |
| --------- | --------- |
| grasp     | 25–35 min |
| peg       | 30–40 min |
| reorient  | 30–40 min |
| **total** | **1.5–2 hr** |

`--curriculum-reference-timesteps 20000000` pins reorient to stage 0 (30°)
for the full 2M: with ref=20M and total=2M, the first stage change at 20M
nominal scales to `20/20 × 2 = 2M = end`. You're evaluating "can stage 0 be
learned at all under the new reward", not the full curriculum.

## 6. Pass bar — per task, by step 1.5M

**grasp** (PPO, 200-step episodes):

- `metrics/num_finger_contacts` ≥ 0.3 on average
- `metrics/object_height` > table+0.1 on best episodes (table = 0.40 →
  peaks > 0.50 m visible in extrema)
- `ep_rew_mean` > +150 by 2M
- `info["is_success"]` / `metrics/success_hold_steps` > 0 at least once

**peg** (SAC, 500-step episodes):

- `metrics/num_finger_contacts` ≥ 1.5
- `metrics/stage` reaches ≥ 1 (fingers on peg) by 1M
- `ep_rew_mean` climbing, not flat at initial value
- `info["is_success"]` fires at least once by 2M (stage 3 + insertion_depth
  above threshold for 10 consecutive steps)
- no single-env blow-up from mass × 1.3 / friction × 0.7

**reorient** (PPO, 400-step episodes):

- `metrics/num_finger_contacts` ≥ 0.3
- `metrics/angular_distance` trending DOWN over time (not drifting up past 1.8 rad)
- `reward/orientation` > 0.1 on average (exp(−2·0.5)·gate ≈ 0.14 within
  0.5 rad of target, inside the stage-0 30° window)
- `metrics/success_steps` non-zero at least once

## 7. Fail modes and per-task rollback

If **any** task regresses, do NOT launch the full budget. Rollback per task:

- **grasp** contacts still < 0.1 at 2M → `action_penalty` scale too aggressive
  even at -0.0002. Drop to `-0.00005` in `rewards/{cpu,gpu}/grasp_reward.py`.
- **peg** stage stuck at 0 at 2M → DR too aggressive or A4 drop penalty
  firing too early. Try `DomainRandomization(enabled=False)` in
  `MjxPegTrainConfig.dr` and re-run just peg.
- **reorient** contacts < 0.2 or angular_distance drifting up → k=2
  exp-shape still too steep for exploration. Either raise
  `orientation_contact_alpha` to 0.7 (more unconditional signal) or switch
  the reward shape to IsaacGymEnvs's `1/(d+0.1)` in
  `rewards/{cpu,gpu}/reorient_reward.py`.
- **any** task NaN → `obs_noise_std` may be interacting badly with
  VecNormalize running stats on the just-reset envs. Disable both
  `DomainRandomization.enabled` and `obs_noise_std=0.0` on the failing
  task's config and re-run.

## 8. Cost

| phase                              | budget                              | wall time  | cost at $0.35/hr |
| ---------------------------------- | ----------------------------------- | ---------- | ---------------- |
| 2M sanity (grasp + peg + reorient) | 2M each, 256 envs                   | 1.5–2 hr   | ~$0.70           |
| full training (after approval)     | grasp 30M + peg 40M + reorient 200M | 34–43 hr   | ~$12–15          |
