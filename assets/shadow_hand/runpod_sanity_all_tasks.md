# RunPod GPU — 2M sanity on cleanup-dead-code

All three tasks (grasp, reorient, peg) need a fresh 2M sanity. Every reward
changed on this branch: action-rate coefficient raised ~100×, peg drop penalty
decoupled from regrasp, reorient `orientation_tracking`+`orientation_success`
collapsed to one `orientation` term, dead terms deleted, domain randomization
turned on by default. Peg's prior 2M pass on main doesn't carry over.

~1.5–2 hr total, ~$0.70 on a 4090.

## TL;DR copy-paste

One block. Pick a pod, paste, walk away. The watcher auto-copies `runs/` to
`/workspace/runs/` and stops the pod when all three tasks finish.

```bash
# --- once, on a fresh pod ---
cd ~ && rm -rf dexterous_hand && git clone -b cleanup-dead-code \
    https://github.com/AryaPeer/Dexterous-Hand.git dexterous_hand
cd ~/dexterous_hand && uv sync --extra mjx

# --- every run ---
tmux new-session -d -s sanity bash
tmux send-keys -t sanity "cd ~/dexterous_hand && \
export CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled && \
mkdir -p runs && ( \
  echo '==================== train-grasp-mjx ====================' && \
  uv run python main.py train-grasp-mjx --num-envs 256 --total-timesteps 2000000 || true; \
  echo '' && \
  echo '==================== train-peg-mjx ====================' && \
  uv run python main.py train-peg-mjx --num-envs 256 --total-timesteps 2000000 || true; \
  echo '' && \
  echo '==================== train-reorient-mjx ====================' && \
  uv run python main.py train-reorient-mjx --num-envs 256 --total-timesteps 2000000 \
      --curriculum-reference-timesteps 20000000 || true; \
  echo '' \
) 2>&1 | tee runs/sanity_stdout.log" Enter

tmux new-session -d -s watcher bash
tmux send-keys -t watcher "while pgrep -f 'main.py train-' > /dev/null; do sleep 60; done && \
  cp -rf ~/dexterous_hand/runs/. /workspace/runs/ && \
  echo RUNPOD_POD_ID=\$RUNPOD_POD_ID && \
  runpodctl stop pod \"\$RUNPOD_POD_ID\"" Enter

# attach to follow along:
tmux attach -t sanity
```

Detach any time with `Ctrl+b d`. When the pod stops, the log is at
`/workspace/runs/sanity_stdout.log`.

## What's being checked

Every task hits a shared set of reward changes so all three need a fresh
baseline. Specifically the branch adds:

- action-rate / smoothness coefficient `-5e-3` → `-0.5` (grasp, peg);
  `-0.002` → `-0.1` (reorient action_rate);
  `-0.005` → `-0.01` (reorient action_penalty). At weight 0.3–0.5 these now
  contribute `O(-0.1)` per step instead of `O(-1e-4)`.
- domain randomization on by default: per-env body_mass × U(0.7, 1.3),
  geom_friction × U(0.7, 1.3), actuator_gainprm × U(0.85, 1.15). resampled
  on every reset.
- obs_noise_std = 0.005 (additive Gaussian on the policy input).
- peg: `complete_bonus` 600 → 2000; `force_threshold` 5 → 15 N; drop
  penalty decoupled from regrasp; spawn matches GPU radial sampling on CPU
  too; `gradient_steps` 32 → 8 at num_envs=512; success is now truncation
  (info["is_success"]=1, TimeLimit.truncated=True) so SAC bootstraps from
  the success terminal obs.
- reorient: `orientation_tracking`+`orientation_success` collapsed into
  one `orientation` term (weight 7.0, `orientation_contact_alpha = 3/7`);
  `velocity_penalty` / `fingertip_distance` / `position_stability`
  computations deleted; `tracking_k = 2.0` from the prior commit;
  `ent_coef = 0.01` from the prior commit.
- grasp: holding gate no longer saturates at 0.5 (added +0.04 offset);
  hold_velocity_k 20 → 100.

## Pass bar (per task, by step 1.5M)

**grasp** (PPO, 200-step episodes):

- `metrics/num_finger_contacts` ≥ 0.3 on average
- `metrics/object_height` > table+0.1 on best episodes (table = 0.40, so
  peaks > 0.50 m visible in extrema)
- `ep_rew_mean` > +150 by 2M
- no NaN / no tracebacks

**peg** (SAC, 500-step episodes):

- `metrics/num_finger_contacts` ≥ 1.5
- `metrics/stage` reaches ≥ 1 (fingers on peg) by 1M
- ep_rew_mean climbing, not flat at initial value
- `info["is_success"]` fires at least once by 2M (you'll see it as a
  success-flagged truncation in logs — stage 3 plus insertion_depth above
  threshold for 10 consecutive steps)
- DR sanity: no single-env blow-up from mass × 1.3 / friction × 0.7

**reorient** (PPO, 400-step episodes):

- `metrics/num_finger_contacts` ≥ 0.3
- `metrics/angular_distance` trending DOWN over time (not drifting up past 1.8 rad)
- `reward/orientation` > 0.1 on average (exp(−2·0.5)·gate ≈ 0.14 within
  0.5 rad of target, which is inside the stage-0 30° window)
- `metrics/success_steps` non-zero at least once

`--curriculum-reference-timesteps 20000000` pins reorient to stage 0 (30°)
for the full 2M: with ref=20M and total=2M, the first stage change at 20M
nominal scales to `20/20 × 2 = 2M = end`. You're evaluating "can stage 0
be learned at all under the new reward", not the full curriculum.

## Fail modes (structural fix needed — do NOT launch full)

- grasp: contacts still < 0.1 at 2M → action-rate bump overshot; drop to
  `-0.05` on the coefficient.
- peg: stage stuck at 0 at 2M → either A4 drop fix made the penalty
  unreachable (lift never crosses 0.10) or DR too aggressive (try
  `mass_range=(0.9, 1.1)` + `friction_range=(0.9, 1.1)`).
- reorient: contacts < 0.2 at 2M, or angular_distance drifting up → k=2
  still too steep for exploration; try `orientation_contact_alpha=0.7` to
  reward orientation progress unconditionally (less contact-gating).
- any: NaN → obs_noise_std may be interacting with VecNormalize running
  stats; disable DR (`DomainRandomization(enabled=False)`) and re-run one
  task to isolate.

## What to paste back

Just the combined stdout is enough:

```
/workspace/runs/sanity_stdout.log
```

If the log is huge:

```
tail -n 500 /workspace/runs/sanity_stdout.log
```

## Cost

| phase                              | budget                              | wall time  | cost at $0.35/hr |
| ---------------------------------- | ----------------------------------- | ---------- | ---------------- |
| 2M sanity (grasp + peg + reorient) | 2M each, 256 envs                   | 1.5–2 hr   | ~$0.70           |
| full training (after approval)     | grasp 30M + peg 40M + reorient 200M | 34–43 hr   | ~$12–15          |
