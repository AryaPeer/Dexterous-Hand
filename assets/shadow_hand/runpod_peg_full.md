# RunPod GPU — Peg full training (150 M)

Targets the round-8 peg full run after the SAC→PPO migration sanity
passed (1 M sanity 2026-05-01: contacts 0.10 → 2.23, ep_rew 0.74 → 375
monotone, value_loss 0.93 → 32.8, no NaN, no auto-α death spiral). The
budget is **150 M**, bumped from the original 100 M plan after the
sanity result + literature review (IsaacGym ShadowHand cube benchmark
runs 600 M at 16,384 envs ≈ 1 M total episodes; our 768-env config at
150 M gives ~520 episodes/env / ~400 K total episodes — comparable
gradient-signal to reorient at 200 M, which is the closest in-codebase
analog).

The reward function is unchanged from round-8 (round-7 gates retained:
`insertion_drive` 4-gate, `complete` 5-gate). PPO removes the SAC
auto-α failure mode that crashed rounds 6 and 7. Sanity verified the
PPO bootstrap path works; this doc ships the full curriculum.

## 1. GPU choice

| GPU            | $/hr   | num-envs | est. wall  | est. cost   | notes                     |
| -------------- | ------ | -------- | ---------- | ----------- | ------------------------- |
| **RTX 5090**   | $0.99  | 768      | ~42 hr     | **~$42**    | recommended               |
| RTX 4090       | $0.69  | 512      | ~63 hr     | ~$43        | cheaper $/hr but slower   |
| H100 SXM       | $2.99  | 2048     | ~17 hr     | ~$51        | fastest single pod        |

Recommendation: **RTX 5090 at 768 envs, 150 M timesteps.** ~$42 total,
~42 hr wallclock. Cost-neutral with 4090; 5090 wins on $/hr predictability
and turnaround. Same env-count as grasp full (`runpod_grasp_full.md`)
and reorient (in flight) — keeps the comparable-config story clean.

**2048 envs is not validated on 5090** — peg has the highest VRAM
pressure of the three tasks (longest episodes, deepest curriculum
state). 768 is the safe operating point. If you hit OOM at 768, drop
to 512.

Anything below 24 GB VRAM won't fit 512+ envs of the Shadow Hand peg
env.

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
tmux new-session -s peg
```

Inside tmux:

```
cd ~/dexterous_hand

export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export WANDB_MODE=disabled

uv run python main.py train-peg-mjx \
    --num-envs 768 \
    --total-timesteps 150000000 \
    2>&1 | tee runs/peg_full_stdout.log
```

(Or just `uv run python main.py train-peg-mjx` — the dataclass defaults
match this exactly after the round-8 bump.)

If you're on RTX 4090 use `--num-envs 512`. On H100 SXM use
`--num-envs 2048` (validated headroom on 80 GB).

`PREALLOCATE=true` + `MEM_FRACTION=0.7` is the configuration that fixed
the `CUDA_ERROR_ILLEGAL_ADDRESS` from the original sanity bundle — keep
both.

Detach with `Ctrl+b d`, reattach later with `tmux attach -t peg`.

## 6. Watcher + auto-stop

Second tmux session:

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

## 7. Curriculum stage layout

`scale_stage_starts(stages, total=150M, ref=100M)` applies a 1.5×
multiplier to the base stage starts:

| stage | start  | clearance | p_pre_grasped | budget   | what it teaches                                           |
| ----- | ------ | --------- | ------------- | -------- | --------------------------------------------------------- |
| 0     | 0      | 0.004 m   | 1.0           | 12 M     | reach + insert with peg already in grip, generous hole    |
| 1     | 12 M   | 0.004 m   | 0.7           | 12 M     | start adding "from non-grasped" trials, easy hole         |
| 2     | 24 M   | 0.003 m   | 0.5           | 12 M     | half-grasp distribution, hole tightens                    |
| 3     | 36 M   | 0.002 m   | 0.3           | 12 M     | mostly-from-random, hole at 2 mm                          |
| 4     | 48 M   | 0.001 m   | 0.2           | **102 M** | the final task: 1 mm clearance, 80 % start non-grasped   |

Stage 4 is the load-bearing one — final task spec, longest budget.
102 M is comparable to reorient stage 2's 140 M for the comparably-hard
180° target.

## 8. What to watch

The CSV is at `runs/peg_mjx_768env_42/logs/progress.csv`. Tail it
from a third tmux pane:

```
tail -f runs/peg_mjx_768env_42/logs/progress.csv
```

### PPO health (must hold throughout)

- `train/approx_kl` median 0.01–0.05, max < 0.1
- `train/clip_fraction` 0.1–0.4 (PPO clipping engaged but not saturated)
- `train/value_loss` bounded (< 1e4 throughout — round-7 SAC hit 1.3e12)
- `train/explained_variance` climbing toward > 0.9 by mid-run
- `train/std` 0.8–1.2 (no entropy collapse)
- `train/metrics/nan_rate` < 0.01

### Per-stage milestones (task signal)

| at timestep | check                                                   | round-7 SAC ref |
| ----------- | ------------------------------------------------------- | --------------- |
| 10 M        | `metrics/num_finger_contacts` ≥ 2.0 (was 2.23 at 1 M sanity) | 0.11 ❌         |
| 12 M        | `metrics/stage` advances past 0.5 (entering stage 1)    | 0.05 stuck      |
| 24 M        | stage advances past 1.5 (entering stage 2)              | n/a             |
| 36 M        | stage advances past 2.5 (entering stage 3)              | n/a             |
| 48 M        | stage advances past 3.5 (entering stage 4)              | n/a             |
| 60 M        | `metrics/insertion_depth` rising sustained > 0.003      | n/a             |
| 100 M       | `metrics/insertion_hold_steps` rising > 1.0 sustained   | n/a             |
| 150 M       | `reward/complete` non-zero events fired                 | n/a             |

If contacts drop below 1.0 sustained for a 5 M window, kill the run —
that's a regression mode the gates are supposed to prevent. If
`approx_kl` climbs past 0.1 sustained, drop `learning_rate` 3e-4 → 1e-4
and resume.

### Stage-transition sanity

When `metrics/stage` jumps to a new integer, expect a transient dip in
`reward/total` (1–2 M timesteps) as the policy adapts to the harder
distribution. If the dip exceeds 50 % of pre-transition reward and
doesn't recover within 5 M of the transition, the curriculum is too
aggressive — note which transition (stage 3→4 is the most likely
offender at clearance 0.002 → 0.001) and consider a curriculum-stagger
intervention (M9 in `gpu_audit.md`).

## 9. After training finishes

The model is saved automatically. Check:

```
ls /workspace/runs/peg_mjx_768env_42/
# expected: best/  checkpoints/  final_model.zip  vec_normalize.pkl  logs/
```

Evaluate on CPU (no GPU needed):

```
uv run python main.py evaluate-peg-mjx \
    --model-path /workspace/runs/peg_mjx_768env_42/final_model.zip \
    --vec-normalize-path /workspace/runs/peg_mjx_768env_42/vec_normalize.pkl
```

Pass bar at eval: `success_rate > 0.30` (1 mm clearance from random
start is genuinely hard — 30 % is the ship line; OpenAI's dexterous
manipulation work hit similar success rates on the equivalent-difficulty
cube task).

## 10. Pass / partial-pass / fail

**Pass** (`eval/success_rate > 0.30`, stage 4 reached, complete events
firing): ship the policy. Optionally extend with a 50 M resume to push
success rate higher.

**Partial pass** (stage 4 reached, contacts/grip stable, but
`insertion_depth` peaks at 0.003–0.005 without crossing the
`success_threshold = 0.7` → `complete` events rare): the geometry is
working but the final mm of insertion isn't paying off. Resume +50 M
is the right move; the policy is in the basin and just needs more time.

**Partial pass** (stage 3 reached but stage 4 collapses contacts): the
1 mm clearance is too sharp a jump. Soften by inserting an interpolated
stage at clearance=0.0015 (round-9 candidate). Don't blanket-resume
without the curriculum patch — the policy will keep hitting the same
wall.

**Fail** (didn't reach stage 4 by 60 M, or contacts collapsed mid-run
in stage 0–3): something regressed vs the sanity. Compare
`runs/peg_full_stdout.log` to the sanity log at the same `total_timesteps`
checkpoint — if PPO health metrics diverge (`value_loss` blowing up,
`approx_kl` runaway, `std` collapsing), it's a learning-dynamics
regression, not a reward-function issue. Drop `learning_rate` to 1e-4
and resume from the last clean checkpoint, or restart from scratch with
the lower LR if too far gone.

## 11. Extending past 150 M (resume from final_model)

If the 150 M run finishes and the policy is "nearly there" — `eval/success_rate`
> 0 but below 0.30, or `reward/complete` rising at the final checkpoint
without saturation — resume rather than retrain. SBX policies pickle
their full optimizer state, and `model.learn(reset_num_timesteps=False)`
continues the run as if it had simply been longer.

A wired CLI is provided. To add 50 M on top of the existing 150 M:

```
uv run python main.py resume-peg-mjx \
    --model-path /workspace/runs/peg_mjx_768env_42/final_model.zip \
    --vec-normalize-path /workspace/runs/peg_mjx_768env_42/vec_normalize.pkl \
    --additional-timesteps 50000000 \
    --num-envs 768 \
    --seed 42
```

Output lands in `<input_dir>_resumed/` by default (override with
`--output-dir`). Three things that matter under the hood, peg-specific:

1. **`reset_num_timesteps=False`** — keeps `model.num_timesteps`
   cumulative across the resume. The curriculum callback fires on
   cumulative timesteps, so without this it re-enters stage 0 and you
   retrain through stages you already passed. Critical for peg because
   the curriculum is the load-bearing structure here.
2. **Reload `vec_normalize.pkl`** via `VecNormalize.load`. Without
   this, observation running stats reset and the policy sees a shifted
   input distribution — costs 5–10 M timesteps of re-stabilization.
3. **Curriculum stays on the original schedule.** `resume_peg.py`
   reconstructs the stages from `MjxPegTrainConfig` defaults
   (`scale_stage_starts(stages, total=150M, ref=100M)` → stage 4 at
   48 M cumulative). Combined with cumulative `model.num_timesteps`,
   resuming after 150 M means the policy is already deep in stage 4 —
   the resume budget is pure stage-4 horizon. There is no re-scaling
   based on `--additional-timesteps`; the curriculum schedule is fixed
   at the original run's design and only stage 4 extends.

Cost: 50 M resume ≈ 14 hr / $14 on 5090. Strictly cheaper than
retraining 200 M from scratch ($56), and you keep the 150 M of learning
already in the policy.

## 12. Cost summary

Single-pod cost: ~$42 on 5090. Round-8 program (assuming peg sanity
already shipped):

| run                              | cost     |
| -------------------------------- | -------- |
| peg sanity (already done)        | ~$1      |
| peg full 150 M (this doc)        | ~$42     |
| grasp full 70 M                  | ~$28     |
| reorient 200 M (in flight)       | ~$80     |
| **program total**                | **~$151** |

Optional resume (if 150 M lands in the partial-pass band):

| run                              | cost     |
| -------------------------------- | -------- |
| peg resume +50 M (200 M total)   | ~$14     |
| **program total with resume**    | **~$165** |
