# RunPod GPU — sanity on cleanup-dead-code (fresh pod)

Round-4 sanity bundle. Re-verifies peg + grasp after applying the audit
H2 reward-shape fixes that round-3 skipped (smooth grasp success gate,
`tanh` contact_scale, smooth peg complete bonus, peg `insertion_drive`
term, lateral_gate_k 10→5, sigmoid steepness 50/100→20/20, lift_target
0.10→0.07). Reorient is verified separately in §4b and is no longer in
this bundle (last 3 M re-run on 2026-04-26 was clean).

What changed previously and is locked in:
- action-rate deleted everywhere; action_penalty = `-0.0002·||a||²`
- peg drop decoupled from regrasp; reorient orientation terms collapsed
- peg `norm_reward=False` (audit C1), grasp `norm_reward=False` (H3)
- peg `gradient_steps=128` (UTD bump, audit H1)
- grasp `is_success` plumbed; one-shot `success_bonus` rising-edge
- peg + grasp `gamma` 0.99 → 0.997 / 0.995 (audit M5)
- NaN guard + pre-grasp settle on peg and reorient
- new this round: `metrics/nan_rate` logged so the < 1 % bar is checkable

Order: peg first (cheapest — if it dies, it dies in ~1.5 hr, not 6+ hr),
then grasp.

~7 hr total wallclock at ~200 fps. ~$2.50 on a 4090.

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
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export WANDB_MODE=disabled

(
  echo "==================== train-peg-mjx ===================="
  uv run python main.py train-peg-mjx --num-envs 256 --total-timesteps 1000000 || true
  echo ""

  echo "==================== train-grasp-mjx ===================="
  uv run python main.py train-grasp-mjx --num-envs 256 --total-timesteps 5000000 || true
  echo ""
) 2>&1 | tee runs/sanity_stdout.log
```

Detach any time with `Ctrl+b d`. Reattach later with `tmux attach -t sanity`.

## 4b. Reorient-only re-run (after CUDA crash fix)

Use this if peg + grasp already passed in a previous bundle and you only
need to verify that the `XLA_PYTHON_CLIENT_PREALLOCATE=true` +
cube-quat normalization fix lets reorient run to 3M without faulting.
Writes to a separate log so the bundled `sanity_stdout.log` isn't
overwritten.

```
tmux new-session -s reorient
```

Inside:

```
cd ~/dexterous_hand

export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export WANDB_MODE=disabled

(
  echo "==================== train-reorient-mjx (rerun) ===================="
  uv run python main.py train-reorient-mjx \
      --num-envs 256 \
      --total-timesteps 3000000 \
      --curriculum-reference-timesteps 30000000 || true
  echo ""
) 2>&1 | tee runs/sanity_reorient_stdout.log
```

Detach with `Ctrl+b d`. Pass criterion: reaches the full 3M without
`CUDA_ERROR_ILLEGAL_ADDRESS` and `metrics/angular_distance` keeps
trending down. The previous run got to 1.27M in ~73 min; budget ~3 hr
for this rerun.

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

Copies `runs/` to `/workspace/runs/` once peg + grasp finish, then
stops the pod. Detach with `Ctrl+b d`.

## 6. After the pod stops, paste back the log

```
/workspace/runs/sanity_stdout.log
```

Or just the tail:

```
tail -n 500 /workspace/runs/sanity_stdout.log
```

## 7. Expected timeline (4090, ~200 fps after H1 UTD bump)

| task     | steps | wall time |
| -------- | ----- | --------- |
| peg      | 1 M   | ~1.5 hr   |
| grasp    | 5 M   | ~6.0 hr   |
| **total** | **6 M** | **~7.5 hr** |

Round-3 actuals: peg 1 M in 1:28, grasp 5 M in 6:09. With H2 reward
shape changes (cheap `tanh`, sigmoids) throughput should be unchanged.

Reorient already verified clean in §4b (3 M in 2:35 with no NaN, no CUDA
crash); not re-running here.

## 8. Pass bar — per task (round-4 reward shapes)

**peg** (SAC, 500-step episodes):

- total reward stays finite for the full 1 M (no NaN cascade)
- `train/metrics/nan_rate` < 0.01 (verifiable now that round-4 logs it)
- `train/ent_coef` follows auto-schedule, NOT collapsing below 0.01 by
  1 M (regression test for audit C1 + H1; round-3 hit 0.024)
- `metrics/num_finger_contacts` ≥ 1.0 by 500 K and not collapsing
  (round-3 hit 2.4 — bar bumped accordingly)
- `metrics/stage` advances past 0.5 by 1 M (round-3 hit 0.93)
- `metrics/insertion_depth` rising above noise floor (≥ 0.001 m by 1 M).
  This is the round-4 deliverable: the smooth `complete_bonus` and
  `insertion_drive` term should let depth crack open past round-3's
  0.0002 m plateau.
- `reward/insertion_drive` non-zero whenever peg is aligned above hole
  (sanity check that the new term actually fires)

**grasp** (PPO, 200-step episodes):

- `metrics/num_finger_contacts` plateauing ≥ 2.5 (round-3 hit 2.5–3.2;
  with the new `tanh(n/2)` contact_scale, ≥ 2.5 is genuine progress)
- `metrics/object_height` peaks ≥ 0.50 m (table=0.40, lift_target=0.07).
  Note the lift_target reduction: 5 cm of clearance is now full-target,
  not the 10 cm bar we had before. Bar lowered to 0.50 m to leave
  margin (0.40 + 0.07 + 0.03 buffer).
- `reward/lifting` > 0.5 by 5 M (with `tanh(n/2)` and lower target,
  the dense lift term actually pays off now)
- `metrics/success_hold_steps` > 0 at least once (the smooth-product
  success gate at threshold 0.85 should be reachable now)
- `reward/success` non-zero at least once (one-shot rising-edge bonus)
- `ep_rew_mean` > 500 by 5 M (looser than before because bonus is now
  one-shot instead of paid-per-step for the rest of the episode)

**reorient** (PPO, 400-step episodes — verified separately in §4b):

- `metrics/angular_distance` trending DOWN (not drifting up past 1.8 rad)
- `reward/angular_progress` averaging > 0 (cube actually rotating toward target)
- `reward/orientation` > 1.0 weighted on average (within stage-0 30° window)
- `metrics/num_finger_contacts` ≥ 0.3 (not dropping cube into orientation-only optimum)
- `metrics/success_steps` non-zero at least once

## 9. Fail modes and per-task rollback (round-4)

If **any** task regresses, do NOT launch the full budget. Rollback per task:

- **peg** NaNs at 1 M → NaN guard misbehaving. Check
  `train/metrics/nan_rate` (now logged); if > 0.01 around the failure,
  lower `MjxPegTrainConfig.curriculum_stages[0]`'s `p_pre_grasped`
  from `1.0` to `0.3`.
- **peg** ent_coef crashes below 0.005 inside 500 K → C1 regression
  (`norm_reward=True` snuck back in, or the gradient_steps bump
  reverted). Verify `config.py:313` is `False` and `:308` is `128`.
- **peg** policy collapses (contacts climb then drop, actor_loss
  diverges) → SAC stuck in idle-hover local optimum despite
  `ent_coef="auto"`. Set `ent_coef=0.1` (fixed floor).
- **peg** `insertion_depth` flat after 1 M with the new `insertion_drive`
  term → align_weight gate may be too tight. In
  `rewards/gpu/peg_reward.py`, the `insertion_drive` is multiplied by
  `align_weight = sigmoid((peg_clearance - 0.02) * 150)`. If peg
  hovers below 2 cm clearance most of the episode, drive never fires.
  Either lower the clearance gate to 0.01 or remove `align_weight`
  multiplier (let drive depend on `axis_align * lateral_factor` only).
- **peg** `reward/complete` non-zero before any real insertion → smooth
  bonus's lower tail leaking. Tighten the inner sigmoid: change
  `sigmoid(20*(frac - 0.7))` to `sigmoid(40*(frac - 0.7))` in both
  CPU and GPU peg_reward.
- **grasp** contacts plateau ≤ 2.0 at 5 M → `tanh(n/2)` contact_scale
  not pulling hard enough toward more contacts. Switch to
  `contact_scale = jnp.minimum(n_contacts / 3.0, 1.0)` (binary at 3)
  for stronger marginal pull, or strengthen `weights.opposition`.
- **grasp** `success_hold_steps` still 0 at 5 M with smooth gate →
  threshold of 0.85 may be unreachable. Drop to 0.7 in
  `rewards/{cpu,gpu}/grasp_reward.py`. Or lower `lift_target` from
  0.07 to 0.05 if the agent is at ~3 cm of lift consistently.
- **grasp** lift raw > 0.1 but `ep_rew_mean` flat → `action_penalty`
  scale too aggressive. Drop to `-0.00005` in
  `rewards/{cpu,gpu}/grasp_reward.py`.
- **reorient** `CUDA_ERROR_ILLEGAL_ADDRESS` mid-run → likely XLA
  fragmentation. Confirm `XLA_PYTHON_CLIENT_PREALLOCATE=true` and
  `MEM_FRACTION=0.7` are set (above). If still crashing, drop
  `--num-envs` from 256 to 128 and re-run reorient alone.
- **reorient** contacts < 0.2 or angular_distance drifting up → k=2
  exp-shape too steep. Either raise `orientation_contact_alpha` to 0.7
  (more unconditional signal) or switch reward shape to
  IsaacGymEnvs's `1/(d+0.1)` in `rewards/{cpu,gpu}/reorient_reward.py`.
- **any** task NaN → `obs_noise_std` may be interacting badly with
  VecNormalize running stats on just-reset envs. Disable both
  `DomainRandomization.enabled` and `obs_noise_std=0.0` on the failing
  task's config and re-run.

## 10. Cost

| phase                          | budget                                        | wall time  | cost at $0.69/hr (4090) |
| ------------------------------ | --------------------------------------------- | ---------- | ----------------------- |
| sanity (peg 1 M + grasp 5 M)   | 6 M total, 256 envs                           | ~7.5 hr    | ~$5                     |
| full peg                       | 60 M, 256 envs                                | ~14 hr     | ~$10                    |
| full grasp                     | 70 M, 256 envs                                | ~17 hr     | ~$12                    |
| full reorient (180°, separate) | 200 M, 768 envs on 5090 (`runpod_reorient_full.md`) | ~80 hr | ~$80 (5090 @ $0.99) |
| **program**                    | sanity + full peg + full grasp + 180° reorient | —          | **~$110**               |

If grasp full could be moved to a 5090 (PPO scales fine and 5090
throughput at 768 envs is ~2× a 4090 at 256), it'd run in ~6 hr at
~$6 — pick whichever is on hand. Peg full SAC is sensitive to UTD;
keep it on a 4090 with `gradient_steps=128` rather than scaling envs.
