# RunPod GPU — Extended Sanity (2M steps) for grasp and reorient

Prior rounds:
- Grasp passed 500k sanity (rew 82 → 138, contacts 0.14 → 0.40) but has not been 2M-sanitied since the audit/scaffolding changes.
- Peg + reorient 500k were stuck in exploration local minima.
- Peg 2M (current main) passed: contacts 2.0, stage 1.0, rew climbed 32 → 1021. Cleared for full 40M — NOT re-run here.
- Reorient 2M (current main) FAILED: contacts dropped 0.11 → 0.07, angular_distance drifted 1.47 → 1.85 rad — gradient-death of exp(-5·ang_dist) past 1 rad plus curriculum compressed to 30% of the 2M run.

This round runs grasp + reorient on main after:
- adding CPU init-qpos scaffolding for reorient/peg eval symmetry
- reorient reward retune (tracking_k 5→2, orientation_success_k 5→2) to fix gradient death
- reorient train ent_coef 0.002 → 0.01 to escape the passive attractor
- reorient sanity passes `--curriculum-reference-timesteps 20_000_000` so stage 0 (30°) holds for the full 2M run

Peg is deliberately skipped (already passed 2M on current main).

~1-1.5 hr total, ~$0.50 on a 4090.

Per-task pass bar:

- grasp: contacts > 0.3 by 1.5M, object_height peaks > table+0.1 on best episodes, ep_rew_mean > +150.
- reorient: contacts > 0.3 by 1.5M, angular_distance trending DOWN over time (not stuck at 1.8 rad), reward/orientation_tracking visibly above 0.1 on average.

## 1. Pull latest and set env vars

```
cd ~/dexterous_hand && git pull

export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled

mkdir -p runs
```

## 2. Run the two sanity tasks under one shell

Grasp has no curriculum; reorient's curriculum reference is overridden so stage 0 (30°) holds for the full 2M run. Using a for loop with `|| true` so a crash on one task does not skip the other.

```
tmux new-session -s sanity
cd ~/dexterous_hand
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled

(
  echo "==================== train-grasp-mjx ===================="
  uv run python main.py train-grasp-mjx --num-envs 256 --total-timesteps 2000000 || true
  echo ""

  echo "==================== train-reorient-mjx ===================="
  uv run python main.py train-reorient-mjx \
      --num-envs 256 \
      --total-timesteps 2000000 \
      --curriculum-reference-timesteps 20000000 || true
  echo ""
) 2>&1 | tee runs/sanity_stdout.log
```

Expected timeline on a 4090:

- grasp: 30-40 min
- reorient: 30-40 min
- total: 1-1.5 hr

`--curriculum-reference-timesteps 20_000_000` (20M) keeps reorient at stage 0 (30°) for the full 2M run — the first stage change is anchored at 20M nominal, scaled to the 2M run it would land at `20M/20M * 2M = 2M = end`, so stages 1-3 are never entered. This lets us evaluate whether the policy can *learn stage 0 at all* under the new reward k=2; if it clears that, the full 200M run will have the standard 400M-ref curriculum where stages unfold naturally.

## 3. While it runs: start the watcher in a second tmux session

```
tmux new-session -s watcher
while pgrep -f "main.py" > /dev/null; do sleep 60; done && command cp -rf ~/dexterous_hand/runs/. /workspace/runs/ && echo "RUNPOD_POD_ID=$RUNPOD_POD_ID" && runpodctl stop pod "$RUNPOD_POD_ID"
```

This auto-copies runs/ to /workspace/runs/ and stops the pod once the three sanity runs finish.

## 4. What to paste back after the pod stops

Just the combined stdout is enough:

```
/workspace/runs/sanity_stdout.log
```

If the log is huge, tail is fine:

```
tail -n 200 /workspace/runs/sanity_stdout.log
```

## 5. What I will check in that output

Pass criteria for "launch full budget":

- grasp: ep_rew_mean > +150 by 2M, metrics/num_finger_contacts > 0.3 by 1.5M, metrics/object_height > table+0.1 on best episodes.
- reorient: metrics/num_finger_contacts > 0.3 by 1.5M, metrics/angular_distance trending DOWN over time (target is in [0.15, 0.52] rad, so the policy should be able to close to < 0.5), reward/orientation_tracking > 0.1 average (exp(-2·0.5) ≈ 0.37 is in range once the policy is within 0.5 rad of target).
- Both: no tracebacks, no NaN, no absurd magnitudes.

Fail criteria (structural fix needed, do NOT launch full):

- grasp: contacts still under 0.1 at 2M → something regressed vs the 500k pass.
- reorient: contacts still under 0.2 at 2M, or angular_distance still drifting away → the tracking_k + ent_coef retune didn't help; next step is probably simpler reward (Dactyl-style progress-only) or more aggressive exploration.

## 6. Cost summary


| phase                          | budget                            | wall time | cost at $0.35/hr |
| ------------------------------ | --------------------------------- | --------- | ---------------- |
| extended sanity (grasp + reorient) | 2M each, 256 envs             | 1-1.5 hr  | ~$0.50           |
| full training (after approval) | grasp 30M + peg 40M + reorient 200M | 34-43 hr | ~$12-15          |


