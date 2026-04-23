# RunPod GPU — Sanity Check for peg, reorient, tactile

Grasp already passed sanity (rew climbed 82 → 138, contacts 0.14 → 0.40 over 500k). This run re-verifies the other three after:

- peg-wall collision fix (peg and tactile should no longer OOM on 4090)
- reorient reward retune (position_stability 2.0→0.5, contact_bonus 0.1→0.5, drop_penalty −20→−10)

500k steps at 256 envs each, ~30-40 min total, under $0.20 on a 4090.


## 1. Pull latest and set env vars

    cd ~/dexterous_hand && git pull

    export CUDA_VISIBLE_DEVICES=0
    export JAX_PLATFORMS=cuda
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export WANDB_MODE=disabled

    mkdir -p runs


## 2. Run the three sanity tasks under one shell

Using a for loop with `|| true` so a crash on one task does not skip the others — we want ALL tracebacks if anything fails.

    tmux new-session -s sanity
    cd ~/dexterous_hand
    export CUDA_VISIBLE_DEVICES=0
    export JAX_PLATFORMS=cuda
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export WANDB_MODE=disabled

    (
      for task in train-peg-mjx train-reorient-mjx train-tactile-mjx; do
        echo "==================== $task ===================="
        uv run python main.py $task --num-envs 256 --total-timesteps 500000 || true
        echo ""
      done
    ) 2>&1 | tee runs/sanity_stdout.log

Expected timeline on a 4090:

- peg: 10-14 min
- reorient: 10-14 min
- tactile: 10-14 min
- total: 30-45 min

The curriculum boundaries inside 500k are reached for peg (1st stage transition at ~100k) and reorient (stages at ~25k, 75k, 150k timesteps) — both exercise the curriculum hook code paths.


## 3. While it runs: start the watcher in a second tmux session

    tmux new-session -s watcher
    while pgrep -f "main.py" > /dev/null; do sleep 60; done && command cp -rf ~/dexterous_hand/runs/. /workspace/runs/ && echo "RUNPOD_POD_ID=$RUNPOD_POD_ID" && runpodctl stop pod "$RUNPOD_POD_ID"

This auto-copies runs/ to /workspace/runs/ and stops the pod once the three sanity runs finish.


## 4. What to paste back after the pod stops

Just the combined stdout is enough:

    /workspace/runs/sanity_stdout.log

If the log is huge, tail is fine:

    tail -n 200 /workspace/runs/sanity_stdout.log


## 5. What I will check in that output

- No traceback anywhere in sanity_stdout.log — peg and tactile especially, since those OOM'd last time
- ep_rew_mean trending in the right direction for each
- Reward components behave as expected:
  - peg: no OOM, reward/reach climbing, reward/grasp rising, metrics/stage advancing past 0 (curriculum kicked in), reward/lift > 0 in later rollouts
  - reorient: metrics/num_finger_contacts > 0.5 (was stuck at 0.07 before the retune), ep_len climbing, reward/orientation_tracking holding or climbing, reward/angular_progress mostly positive
  - tactile: no OOM, same reward signals as peg, plus the tactile triple — change channel should now have non-zero values when contacts occur (confirms the earlier previous_tactile fix)
- Curriculum log lines like "[Curriculum] Stage 1: ..." appearing in peg and reorient
- No NaN or absurd magnitudes in any reward key


## 6. Cost summary

| phase | budget | wall time | cost at $0.35/hr |
|-------|--------|-----------|-------------------|
| sanity 3 tasks | 500k each, 256 envs | 30-45 min | ~$0.20 |
| full training (after approval) | see runpod_option_b.md | 37-46h | ~$13-16 |
