# RunPod GPU — Peg sanity (1 M, post PPO migration)

Validates the round-9 algorithm swap: peg moved from SAC (sbx) to PPO
(sbx) to escape the recurring auto-α death spiral that crashed rounds 6
and 7. Round-7 SAC at 1 M sanity hit `ent_coef=1972`, `critic_loss=1.3e12`,
contacts collapsed to 0.11 — classic auto-temperature divergence under
sparse reward. PPO (no auto-α, fixed `ent_coef=0.01`) is what already
works on grasp and reorient on this codebase.

This sanity is one-and-done before the 100 M full run — it verifies
the new algorithm/env combination is healthy. The reward function is
unchanged from round-8 (round-7 gates retained: `insertion_drive` has
all four gates, `complete` has all five — both random-policy probe
verified).

## 1. GPU choice

| GPU            | $/hr   | num-envs | est. wall  | est. cost   |
| -------------- | ------ | -------- | ---------- | ----------- |
| **RTX 4090**   | $0.69  | 256      | ~1.5 hr    | **~$1**     |
| RTX 5090       | $0.99  | 256      | ~1.0 hr    | ~$1         |

Recommendation: **RTX 4090 at 256 envs.** Cheapest sanity-class pod.
This is just a pre-flight check; throughput doesn't matter — diagnostics
do.

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

## 5. Launch sanity

```
tmux new-session -s peg-sanity
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
    --num-envs 256 \
    --total-timesteps 1000000 \
    2>&1 | tee runs/peg_sanity_stdout.log
```

`PREALLOCATE=true` + `MEM_FRACTION=0.7` keeps XLA from fragmenting under
the 500-step episode + 5-stage curriculum.

Detach with `Ctrl+b d`, reattach with `tmux attach -t peg-sanity`.

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

## 7. Early-exit checks (kill the run if these fire)

The whole point of sanity is to catch divergence in 25 min instead of 90.
Tail the CSV:

```
tail -f runs/peg_mjx_256env_42/logs/progress.csv
```

**Kill at 250 K timesteps if:**

- `train/value_loss` > 1e6 (PPO equivalent of SAC's critic explosion)
- `train/approx_kl` > 0.2 sustained (policy update too aggressive)
- `train/metrics/nan_rate` > 0.05 (NaN guard firing constantly)
- `train/metrics/num_finger_contacts` < 0.1 sustained from the start
  (no grasp signal — gates may need re-softening)

PPO does not auto-tune entropy, so the round-7 `ent_coef → 1972`
divergence cannot recur. If `value_loss` is bounded and `approx_kl` is
in PPO's healthy range (0.01–0.05), training is on track regardless of
how slow contacts climb.

## 8. Pass criteria at 1 M

**PPO health (algorithm-side, must hold):**

- `train/approx_kl` median 0.01–0.05, max < 0.1
- `train/clip_fraction` 0.1–0.4 (PPO clipping engaged but not saturated)
- `train/value_loss` bounded (< 1e4 throughout — round-7 SAC hit 1.3e12)
- `train/metrics/nan_rate` < 0.01

**Task-side (the reason we're doing this):**

- `train/metrics/num_finger_contacts` ≥ 1.0 by 1 M
  (round-7 SAC: 0.11 ❌, round-3 SAC reference: 2.4 ✅)
- `train/metrics/stage` advances past 0.1 by 1 M
  (round-7 SAC: 0.05 ❌)
- `train/metrics/insertion_depth` rising above 0.001 m by 1 M
  (sustained, not peak-then-collapse)
- `train/reward/insertion_drive` non-zero whenever peg is held + aligned
  (sanity check that the 4-gate term still fires when conditions are met)
- `train/reward/complete` ≈ 0 throughout
  (gates close the leak — the round-7 probe verified this)
- `train/rollout/ep_rew_mean` finite, not collapsing

A 1 M sanity at 256 envs gives only ~30 PPO updates total, so the policy
won't be near optimal — we're checking for early-warning of structural
problems, not learning convergence.

## 9. Pass / fail decision tree

**All pass:** ship the 100 M full run. Same env count (256) is fine if
you want to keep cost under $5; bump to 768 envs for ~$28 / ~28 hr.

**PPO health pass, task signal slow** (contacts 0.5–1.0, no stage
advance): bump exploration. Edit `MjxPegTrainConfig.ent_coef` from 0.01
to 0.05 in `dexterous_hand/config.py`. Re-run sanity. PPO peg may need
more entropy than grasp/reorient because the reward landscape is
sparser at episode start.

**PPO health fail** (`approx_kl` runaway, `value_loss` exploding):
the gradient update is too aggressive for this reward magnitude. Drop
`learning_rate` from 3e-4 to 1e-4, or `n_epochs` from 10 to 5. Also
worth checking that `norm_reward=False` is still the default — VecNormalize
on a sparse-reward run is the recurring foot-gun.

**Contacts stay at 0** (round-7-style collapse despite different algo):
the round-7 reward gates are also too strict for PPO. Fall back to
soft contact_scale gating — in `rewards/{cpu,gpu}/peg_reward.py`
replace `contact_scale = min(n_contacts / 3.0, 1.0)` with
`contact_scale = tanh(n_contacts / 1.5)` so the no-contact state has
non-zero reward gradient. This restores some bootstrap signal at the
cost of a little leak; acceptable since PPO doesn't have auto-α to
amplify the leak into divergence.

## 10. What sanity does NOT cover

- Stage transitions (curriculum advances at 8 M / 16 M / 24 M / 32 M
  — none reachable in 1 M sanity). Stage-transition regressions
  surface only in the full run.
- Long-horizon collapse (round-6 SAC peaked at ~350 K then collapsed by
  700 K — a slow drift that PPO is structurally less prone to but not
  immune from). If sanity passes, `value_loss` and `approx_kl` are the
  early-warning signals to monitor on the full run.

## 11. After sanity

If pass: read this doc's §9 "All pass" line, kill the pod, launch the
full 100 M run. Default config now produces a working 100 M command:

```
uv run python main.py train-peg-mjx \
    --num-envs 768 \
    --total-timesteps 100000000
```

(or just `uv run python main.py train-peg-mjx` — the dataclass defaults
match this exactly).

Same `PREALLOCATE=true` + `MEM_FRACTION=0.7` env vars; full run takes
~40 hr / ~$40 on 5090 at 768 envs. With the bumped curriculum reference
(100 M), the curriculum stages stay literal at [0, 8 M, 16 M, 24 M,
32 M], leaving the hardest stage (clearance=0.001) **68 M** of training
budget — comparable to reorient stage 2's 140 M for the comparably-hard
180° target.

## 12. Extending past 100 M (resume from final_model)

If the 100 M full run finishes and the policy is close — eval success
rate > 0 but below target, or reward still climbing at the final
checkpoint — you can resume rather than retrain. Same mechanism reorient
uses for SO(3) extension (`runpod_reorient_full.md` §11).

A wired CLI is provided. To add 50 M on top of the existing 100 M:

```
uv run python main.py resume-peg-mjx \
    --model-path /workspace/runs/peg_mjx_768env_42/final_model.zip \
    --vec-normalize-path /workspace/runs/peg_mjx_768env_42/vec_normalize.pkl \
    --additional-timesteps 50000000 \
    --num-envs 768 \
    --seed 42
```

Output lands in `<input_dir>_resumed/` by default (override with
`--output-dir`). Three things that matter under the hood:

1. **`reset_num_timesteps=False`** — keeps `model.num_timesteps` cumulative
   so the curriculum callback fires correctly. Without this, the curriculum
   re-enters stage 0 and you re-train through easy stages you already passed.
2. **Reload `vec_normalize.pkl`** via `VecNormalize.load`. Without this,
   observation running stats reset and the policy sees a shifted input
   distribution — costs 5-10 M timesteps of re-stabilization.
3. **Re-supply curriculum callback.** Loaded models do not auto-restore
   callbacks, only optimizer/network state. If you skip this the env
   stays pinned to the final loaded stage with no further advance.

Cost: 50 M resume ≈ 20 hr / $20 on 5090. Strictly cheaper than retraining
150 M from scratch ($60).
