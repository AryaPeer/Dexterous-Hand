# RunPod GPU — sanity on cleanup-dead-code (fresh pod)

All three tasks (peg, grasp, reorient) need a fresh sanity. Every reward
changed on this branch: action-rate deleted everywhere, action_penalty
standardized to IsaacGymEnvs's `-0.0002·||a||²`, peg drop decoupled from
regrasp, reorient `orientation_tracking`+`orientation_success` collapsed to
one `orientation` term, peg DR off, reorient DR on, initial-state noise
bumped to Dactyl scale, NaN guard + pre-grasp settle on peg and reorient.

Order: peg first (cheapest verification of NaN bundle — if it dies, it dies in 90 min, not 9 hr), then grasp, then reorient.

~14 hr total wallclock at ~160 fps. ~$5 on a 4090.

## 1. Pod prerequisites

Pick a pod with CUDA 12.4+ and ≥24 GB VRAM (4090, L40S, 6000 Ada, A100, H100).

## 2. System setup (fresh pod)

```
apt-get update && apt-get install -y tmux git
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

## 3. Clone and install

```
cd ~
git clone -b cleanup-dead-code https://github.com/AryaPeer/Dexterous-Hand.git dexterous_hand
cd dexterous_hand
uv sync --extra mjx
mkdir -p runs
```

## 4. Start tmux session and paste the training block

```
tmux new-session -s sanity
```

Then inside the tmux session:

```
cd ~/dexterous_hand

export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled

(
  echo "==================== train-peg-mjx ===================="
  uv run python main.py train-peg-mjx --num-envs 256 --total-timesteps 1000000 || true
  echo ""

  echo "==================== train-grasp-mjx ===================="
  uv run python main.py train-grasp-mjx --num-envs 256 --total-timesteps 5000000 || true
  echo ""

  echo "==================== train-reorient-mjx ===================="
  uv run python main.py train-reorient-mjx \
      --num-envs 256 \
      --total-timesteps 3000000 \
      --curriculum-reference-timesteps 30000000 || true
  echo ""
) 2>&1 | tee runs/sanity_stdout.log
```

Detach any time with `Ctrl+b d`. Reattach later with `tmux attach -t sanity`.

`--curriculum-reference-timesteps 30000000` pins reorient to stage 0 (30°)
for the full 3M: with ref=30M and total=3M, the first stage change at 20M
nominal scales to `20/30 × 3 = 2M` — i.e. you'd cross into stage 1 only at
2M of 3M. You're evaluating "can stage 0 be learned at all", not the full
curriculum.

## 5. In a second tmux session, paste the watcher

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

Copies `runs/` to `/workspace/runs/` once all three trainings finish, then
stops the pod. Detach with `Ctrl+b d`.

## 6. After the pod stops, paste back the log

```
/workspace/runs/sanity_stdout.log
```

Or just the tail:

```
tail -n 500 /workspace/runs/sanity_stdout.log
```

## 7. Expected timeline (4090, ~160 fps)

| task     | steps | wall time |
| -------- | ----- | --------- |
| peg      | 1M    | ~1.6 hr   |
| grasp    | 5M    | ~8.7 hr   |
| reorient | 3M    | ~5.2 hr   |
| **total** | **9M** | **~15.5 hr** |

## 8. Pass bar — per task

**peg** (SAC, 500-step episodes):

- total reward stays finite for the full 1M (no NaN cascade)
- `metrics/num_finger_contacts` ≥ 1.5 by 500K
- `metrics/insertion_depth` rising above lateral-distance noise floor
- NaN guard fires < 1% of steps if at all
- no single-env blow-up under fixed `ent_coef=0.05` + halved slider forcerange

**grasp** (PPO, 200-step episodes):

- `metrics/num_finger_contacts` plateauing ≥ 3.0 (full grip)
- `metrics/object_height` peaks ≥ 0.50 m (table=0.40, lift_target=0.10)
- raw `lifting` (= `reward/lifting / 8.0`) > 0.1 by 3M
- `ep_rew_mean` > 600 by 5M
- `metrics/success_hold_steps` > 0 at least once

**reorient** (PPO, 400-step episodes):

- `metrics/angular_distance` trending DOWN (not drifting up past 1.8 rad)
- `reward/angular_progress` averaging > 0 (cube actually rotating toward target)
- `reward/orientation` > 1.0 weighted on average (within stage-0 30° window)
- `metrics/num_finger_contacts` ≥ 0.3 (not dropping cube into orientation-only optimum)
- `metrics/success_steps` non-zero at least once

## 9. Fail modes and per-task rollback

If **any** task regresses, do NOT launch the full budget. Rollback per task:

- **peg** still NaNs at 1M → NaN guard misbehaving. Check
  `runs/peg_mjx_*/progress.csv` for the exact step. If happens during
  pre-grasp init, lower `MjxPegTrainConfig.curriculum_stages[0]`'s
  `p_pre_grasped` from `1.0` to `0.3`.
- **peg** stage stuck at 0 at 1M → `ent_coef=0.05` too low for exploration.
  Try `0.1` or revert to `"auto"` and re-run just peg.
- **grasp** contacts still < 2 at 5M → the contact-scale gate is the
  bottleneck. Soften: `contact_scale_lift = 0.3 + 0.7 * min(n_contacts/3, 1)`
  in `rewards/gpu/grasp_reward.py` so partial grip still gives partial lift
  signal.
- **grasp** lift raw > 0.1 but `ep_rew_mean` flat → `action_penalty` scale
  too aggressive. Drop to `-0.00005` in `rewards/{cpu,gpu}/grasp_reward.py`.
- **reorient** contacts < 0.2 or angular_distance drifting up → k=2
  exp-shape too steep. Either raise `orientation_contact_alpha` to 0.7
  (more unconditional signal) or switch reward shape to
  IsaacGymEnvs's `1/(d+0.1)` in `rewards/{cpu,gpu}/reorient_reward.py`.
- **any** task NaN → `obs_noise_std` may be interacting badly with
  VecNormalize running stats on just-reset envs. Disable both
  `DomainRandomization.enabled` and `obs_noise_std=0.0` on the failing
  task's config and re-run.

## 10. Cost

| phase                              | budget                              | wall time  | cost at $0.35/hr |
| ---------------------------------- | ----------------------------------- | ---------- | ---------------- |
| sanity (peg 1M + grasp 5M + reorient 3M) | 9M total, 256 envs            | ~15.5 hr   | ~$5.50           |
| full training (after approval)     | grasp 30M + peg 40M + reorient 200M | 34–43 hr   | ~$12–15          |
