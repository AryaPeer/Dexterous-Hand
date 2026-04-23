# RunPod GPU — Extended Sanity (2M steps) for peg and reorient

Grasp already passed sanity at 500k (rew 82 → 138, contacts 0.14 → 0.40) — skipping it here.

The 500k sanity on peg and reorient showed both policies still stuck in exploration local minima after reward retunes:

- peg: contacts 0.02, stage stuck at 0, policy hovers near peg without grasping
- reorient: contacts 0.11 (barely up from 0.07), angular distance drifting to 1.84 rad, cube balanced flat on palm

500k is too short for SAC to reliably escape exploration. This extended sanity runs each task for 2M steps to see if contact behavior actually emerges given more samples. ~1.5 hr total, ~$0.55 on a 4090.

If peg contacts exceed ~0.3 and stage advances past 0 by 2M, SAC is learning and full 40M budget is worth it. If peg is still stuck at contacts <0.05 at 2M, no amount of additional training will help — we need structural changes (init-qpos scaffolding, or switch to PPO with strong curriculum).

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

Using a for loop with `|| true` so a crash on one task does not skip the other — we want ALL tracebacks if anything fails.

```
tmux new-session -s sanity
cd ~/dexterous_hand
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled

(
  for task in train-peg-mjx train-reorient-mjx; do
    echo "==================== $task ===================="
    uv run python main.py $task --num-envs 256 --total-timesteps 2000000 || true
    echo ""
  done
) 2>&1 | tee runs/sanity_stdout.log
```

Expected timeline on a 4090:

- peg: 30-40 min
- reorient: 30-40 min
- total: 1-1.5 hr

At 2M total steps each, curriculum stages scale to:

- peg: stage 0 through stage 4 (p_pre_grasped 1.0 → 0.2) all get exercised
- reorient: all 4 curriculum angle stages (30° → 90° → 180° → π) get visited

So you also see whether the policy survives the curriculum transitions and whether contact behavior holds when the task gets harder.

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

- peg: metrics/num_finger_contacts > 0.3 by step ~1.5M, metrics/stage advances above 0, metrics/peg_height climbs above 0.5 (peg actually lifted)
- reorient: metrics/num_finger_contacts > 0.3, metrics/angular_distance trending DOWN over time (not stuck at 1.8 rad)
- Both: no tracebacks, no NaN, no absurd magnitudes

Fail criteria (structural fix needed, do NOT launch full):

- peg: contacts still under 0.1 at 2M → SAC exploration wall, needs init-qpos scaffolding
- reorient: contacts still under 0.2 at 2M, or angular_distance still drifting away from target → need init-qpos grip + higher PPO entropy

## 6. Cost summary


| phase                          | budget                            | wall time | cost at $0.35/hr |
| ------------------------------ | --------------------------------- | --------- | ---------------- |
| extended sanity (2 tasks)      | 2M each, 256 envs                 | 1-1.5 hr  | ~$0.55           |
| full training (after approval) | grasp 30M + peg 40M + reorient 200M | 34-43 hr | ~$12-15          |


