# RL Codebase Audit — GPU/MJX Path (Grasp PPO + Peg SAC)

## Status (2026-04-28)

Round-4 application status:

| item | status | landed in |
|------|--------|-----------|
| C1   peg `norm_reward=False`                                | ✅ shipped | round-3 |
| C2   grasp `is_success` plumbed + one-shot rising-edge      | ✅ shipped | round-3 |
| H1   peg `gradient_steps=128` (UTD bump 0.004 → 0.5)        | ✅ shipped | round-3 |
| H2   sigmoid steepness 50/100 → 20/20                       | ✅ shipped | round-4 |
| H2   smooth-product success gate at 0.85                    | ✅ shipped | round-4 |
| H2   contact_scale `tanh(n/2)` for grasp                    | ✅ shipped | round-4 |
| H2   peg complete_bonus smooth (sigmoid product)            | ✅ shipped | round-4 |
| H2   peg lateral_gate_k 10 → 5                              | ✅ shipped | round-4 |
| H2   peg `insertion_drive` (downward velocity term)         | ✅ shipped | round-4 |
| H3   grasp `norm_reward=False`                              | ✅ shipped | round-3 |
| M2   `metrics/nan_rate` logged                              | ✅ shipped | round-4 |
| M3   `activation_fn` threaded into MJX configs              | ✅ shipped | round-4 |
| M5   peg γ=0.997, grasp γ=0.995                             | ✅ shipped | round-3 |
| H2   drop penalty cliff → continuous shaping                | ❌ deferred | — |
| M7   buffer_size 1 M → 5 M                                  | ❌ deferred | — |
| M8   running rollout success_rate callback                  | ❌ deferred | — |
| M9   peg curriculum stagger                                 | ❌ deferred | — |

Lift-related round-4 knobs that are NOT in the audit but landed
alongside H2:
- `RewardConfig.lift_target` 0.10 → 0.07 m (grasp)
- Smooth product success gate uses `n_contacts ≥ 3` (was hard `≥ 4`)

The deferred items are observability / data-efficiency improvements,
not learning-impact blockers.

## Context

The user trains two MuJoCo MJX (JAX) policies on RunPod 4090s: **grasp** (PPO via SBX) and **peg insertion** (SAC via SBX). This audit focuses exclusively on the GPU/MJX path. Findings are tied to file:line and ranked by severity. CPU findings appear only when they share the *exact* code path used on GPU (the rewards module is JAX‑ported from the CPU version).

Files inspected:

- `dexterous_hand/envs/gpu/mjx_vec_env.py` — base MJX VecEnv with auto‑reset, NaN guard, DR, obs‑noise
- `dexterous_hand/envs/gpu/grasp_env.py`, `peg_env.py` — task subclasses
- `dexterous_hand/rewards/gpu/grasp_reward.py`, `peg_reward.py` — JAX reward functions
- `scripts/training/gpu/train_grasp.py`, `train_peg.py` — SBX training entrypoints
- `dexterous_hand/config.py` — `MjxGraspTrainConfig` (line 234), `MjxPegTrainConfig` (line 297)
- `assets/shadow_hand/runpod_sanity_all_tasks.md` — sanity protocol on the 4090

What works well (verified, not findings):

- `mjx_vec_env.py:177-194` correctly distinguishes `truncated` vs `terminated` — the CPU bug (no episode boundary in grasp) is **not** present on GPU. Timeout sets `truncated`, falls/launches set `terminated`.
- `mjx_vec_env.py:183-185` has a NaN guard that zeros rewards / obs and force‑resets divergent envs — good safety net.
- `_apply_dr` at line 23 properly resamples DR per‑env on reset (line 211-221) and preserves multipliers across in‑flight episodes.
- `_obs_noise_key` is split before each use (line 141) — no key reuse.
- VecNormalize stats are saved on both training scripts (`train_grasp.py:111-112`, `train_peg.py:124-125`); the CPU eval env loads them correctly via `make_cpu_eval_env`.

---

## CRITICAL — these will actively destabilize GPU training

### C1. SAC peg uses `norm_reward=True` (the same CPU bug, in the GPU run you're actually executing)

- `config.py:313` — `MjxPegTrainConfig.norm_reward: bool = True`
- `scripts/training/gpu/train_peg.py:53` — `norm_reward=config.norm_reward` passed to `VecNormalize`

`VecNormalize(norm_reward=True)` divides reward by a running std. SAC's value target `r + γ V(s')` and `ent_coef="auto"` (config.py:309) both calibrate against absolute reward magnitudes. Once VecNormalize's running stats lock in (≈ first 1‑5M steps), the per‑step reward distribution is shifted/scaled so that:

- The `complete_bonus=250` (config.py:167) becomes much smaller in the value function's view than it should be relative to dense shaping.
- The auto‑tuned entropy coefficient drifts — the runpod sanity doc (`runpod_sanity_all_tasks.md:153`) explicitly lists *"`train/ent_coef` schedule on auto, not pinned to a low constant"* as a pass criterion, which is exactly the symptom the bug produces.
- Q‑targets become inconsistent across episodes because the running mean changes as the policy improves.

**Fix:** `MjxPegTrainConfig.norm_reward = False`. Verify by re‑running the 1M sanity and checking `train/ent_coef` no longer collapses.

### C2. Grasp `success_bonus=250` is paid per‑step, never truncates, and saturates the return

- `dexterous_hand/rewards/gpu/grasp_reward.py:109-110` — `is_success = new_success_hold >= success_hold_steps`; `success = jnp.where(is_success, success_bonus_value, 0.0)`
- `dexterous_hand/envs/gpu/grasp_env.py:172-209` — the `_step_single` returns `info` from `grasp_reward(...)` but **does not add `info["is_success"]`** to it.
- `mjx_vec_env.py:188-194` — `is_success = reward_info.get("is_success")`; if missing, `is_success = jnp.zeros(num_envs, dtype=bool)` and `truncated_only = (timed_out & ~dones) | False`.

Net effect: a grasp episode that achieves the success condition at, say, step 50 will keep collecting `success=250` every step until the 200‑step timeout. Maximum per‑episode success contribution is `(200 − 20) × 250 = 45,000` reward — orders of magnitude above the dense terms (`reaching ≤ 1`, `holding ≤ 1`, `lifting ≤ 1.5`, weighted to ~20 max per step).

Two compounding consequences:

1. **VecNormalize destabilization** (interacts with C5): once the first successful episode lands, the reward running std jumps by 10–100×, suppressing the dense shaping signal for every other env in the batch — exploration collapses.
2. **Advantage estimation explodes**. PPO's `clip_range=0.2` (config.py:244) is per‑update; it doesn't help when a single trajectory's GAE is dominated by a 45k spike.

The runpod pass bar `ep_rew_mean > 600 by 5M` (`runpod_sanity_all_tasks.md:160`) implicitly assumes a ~1‑2 % success rate clears it — i.e., the bar already accounts for this saturation rather than fixing it.

**Fix (two changes, both needed):**

1. In `dexterous_hand/envs/gpu/grasp_env.py:_step_single` (around line 192, after the reward call), set `info["is_success"] = is_success` so the vec env truncates on success — mirror what peg does at `peg_env.py:331`. Need to plumb `is_success` out of `grasp_reward()` (currently it only emits `reward/success`); easiest is to add `info["is_success"] = is_success` to the dict it returns at `grasp_reward.py:140-156`.
2. Make `success_bonus` one‑shot: change `success = jnp.where(is_success, success_bonus_value, 0.0)` to fire only on the rising edge — e.g., `success = jnp.where(is_success & ~state_is_already_success, success_bonus_value, 0.0)`. Carry the previous flag in `GraspRewardState`.

---

## HIGH — material reward shape and algorithm tuning

### H1. SAC UTD ratio is catastrophically low on GPU

- `config.py:307-308, 299` — `train_freq=1`, `gradient_steps=8`, `num_envs=512` (default), argparse default in `train_peg.py:134` is `--num-envs 2048`.
- SBX SAC's `collect_rollouts` runs `train_freq.frequency` *vectorized* steps then `gradient_steps` updates.

Per‑transition update‑to‑data ratio:

- Sanity (256 envs): `8 / 256 = 0.031`
- Default (512): `8 / 512 = 0.016`
- Argparse default (2048): `8 / 2048 = 0.004`

Standard SAC is UTD≈1; sample‑efficient variants run UTD=10‑20. Current setting wastes 99.6% of collected experience. With 60M timesteps / 2048 envs = ~30K vectorized steps × 8 grad steps = 240K total parameter updates over the whole run. That is roughly the data efficiency of a 250K‑step single‑env SAC.

**Fix:** raise `gradient_steps` to at least `num_envs / 4` (e.g., 128 for 512 envs, 512 for 2048 envs), or set `train_freq=8` to consume more data per update cycle. Watch wall‑clock — JAX gradient steps on a 4090 are cheap; the bottleneck has been the vec env, not the optimizer.

### H2. Reward shaping issues from CPU are 1‑for‑1 on GPU (rewards module is JAX‑ported, same logic)

The GPU reward functions are line‑for‑line ports of the CPU shaping with `np` → `jnp`:

- **Sigmoid steepness on hold gates** — `grasp_reward.py:95-96`, `hold_height_k=50`, `hold_velocity_k=100` (config.py:46-47). At `obj_speed=0.04` the speed gate is ~0.73, at 0.06 ~0.27 — effective transition window ~2 cm/s. **Fix:** drop both to ~20.
- **Hard triple‑AND success gate** — `grasp_reward.py:105`, `(lift_height >= lift_target) & (n_contacts >= 4) & (obj_speed < 0.2)`. Three binary thresholds. **Fix:** use a smooth product like `clip(lift_height/0.1,0,1) * clip(n_contacts/3,0,1) * sigmoid(20*(0.2-obj_speed))` and threshold the product at 0.85 — also softens C2 since `at_target` becomes hysteretic.
- **`contact_scale` reward cancellation** — `grasp_reward.py:88, 90, 92, 97` (and `peg_reward.py:90, 96, 108`). `lifting`/`holding`/`grasping`/`align` all multiplied by `min(n_contacts/k, 1)`. Lose all contacts → all collapse to 0 simultaneously while `reaching` re‑emerges. **Fix:** smooth `contact_scale = tanh(n_contacts / 2)` and add a continuous "loss‑of‑grip" penalty `-w * was_lifted_prev * (1 - contact_scale)`.
- **Drop penalty cliff** — `grasp_reward.py:101-102`, `peg_reward.py:127-128`. `drop = drop_penalty if was_lifted & (lift_height < 0.01)` — fires once at the floor, no shaping during fall.
- **Peg `complete_bonus` temporal cliff** — `peg_reward.py:121`, `complete = jnp.where(new_hold >= peg_hold_steps, complete_bonus, 0.0)`. Cross 70% insertion for 9 frames then bounce back: 0 reward for that effort. **Fix:** `complete_bonus * sigmoid(20*(insertion_fraction - 0.7)) * sigmoid(new_hold/5 - 1)`.
- **Peg lateral 5 cm tunnel** — `peg_reward.py:51, 105, 111`, `lateral_gate_k=10` on lateral distance in meters. At 0.1 m lateral, factor ≈ 0.24. **Fix:** drop `lateral_gate_k` to 5.
- **Peg "aligned but hovering" flat region** — `peg_reward.py:107`, `align_weight = sigmoid(150 * (peg_clearance - 0.02))`. Aligned and hovering above the hole: full align reward, zero depth reward, no derivative pulling down. **Fix:** add a small downward‑velocity term in stage 3, e.g., `align_weight * jax.nn.relu(-peg_linvel[2]) * scale`.

### H3. PPO grasp also has `norm_reward=True` (less critical but interacts with C2)

- `config.py:251` — `MjxGraspTrainConfig.norm_reward = True`
- `train_grasp.py:50` — passed to VecNormalize.

PPO is more tolerant of reward normalization than SAC because advantages are batch‑normalized. But combined with C2 (the 45k bonus saturation), VecNormalize's running std will be dominated by post‑success episodes once a successful policy emerges, *suppressing* the dense shaping for every other parallel env that hasn't succeeded yet. Either fix C2 (so success contribution is bounded), or set `norm_reward=False`. Doing both is safest.

---

## MEDIUM — likely to slow learning or hide problems

### M1. Curriculum changes force a full JIT recompile every stage

- `dexterous_hand/envs/gpu/peg_env.py` — `set_curriculum_params(clearance, p_pre_grasped)` re‑JITs `_batched_reset`/`_batched_step` even when only `p_pre_grasped` changes.

`p_pre_grasped` is a Python float used inside JAX RNG sampling — changing it shouldn't need a re‑trace. Only `clearance` changes the model XML and thus needs a model rebuild. Currently every curriculum stage advance pays the full compile cost (minutes on a 4090).

**Fix:** make `_p_pre_grasped` a `jnp.array` attribute fed to the JITted reset via a closure or an explicit arg, so changing it doesn't invalidate the trace. Only call `_build_batched_step` again when `clearance` actually changed.

### M2. NaN guard hides the symptom — no metric tracks NaN frequency

- `mjx_vec_env.py:183-185` — silently zeros NaN rewards/obs and force‑resets bad envs.

The guard is correct (without it a single divergent env poisons the whole policy). But there's no `metrics/nan_rate` or similar logged to W&B, which means the runpod pass criterion *"NaN guard fires < 1% of steps if at all"* (`runpod_sanity_all_tasks.md:152`) is currently impossible to verify from the dashboards.

**Fix:** add `info["metrics/nan_rate"] = bad.astype(jnp.float32)` (or `jnp.mean(bad)` if you prefer scalar) so `RewardInfoLoggerCallback` picks it up and logs `train/metrics/nan_rate`.

### M3. SAC `policy_kwargs` drops `activation_fn` (CPU/GPU asymmetry)

- `train_peg.py:74-76` — only `net_arch` passed; SBX defaults to ReLU.
- `config.py:216` — CPU `PegTrainConfig` has `activation: str = "elu"` and the CPU script honors it.
- Grasp GPU script also drops activation (`train_grasp.py:67-69`).

CPU and GPU runs use *different network activations* despite being "the same task". When you compare a CPU sanity run to a GPU run, the gradient profiles differ for unrelated reasons.

**Fix:** mirror the CPU policy_kwargs — add an activation arg to the MJX configs and pass it via `policy_kwargs`. At minimum, document the intended activation in the MJX config.

### M4. SAC peg `n_eval_episodes=5` is too few for a stochastic insertion task

- `train_peg.py:103` — `n_eval_episodes=5`.
- Grasp uses 20 (`train_grasp.py:90`).

Peg success is ~0.5–10 % through most of training. Five episodes per eval means the empirical success rate is one of {0, 0.2, 0.4, 0.6, 0.8, 1.0} — too coarse to distinguish "best model" reliably. EvalCallback's "save best" will be driven by the first lucky eval.

**Fix:** raise to 20+ for peg, matching grasp. Eval is on a single CPU env so 20 episodes × 500 steps × ~50 fps ≈ 200s — acceptable cadence.

### M5. `gamma=0.99` discounts late peg insertion to near zero

- `config.py:242` (grasp) and `:306` (peg) — both 0.99.
- Reorient (`MjxReorientTrainConfig.gamma=0.998`, line 267) already gets this right.
- γ^200 ≈ 0.13 (grasp horizon) and γ^500 ≈ 0.0066 (peg horizon).

Peg's success is at the END of a 500‑step trajectory; at γ=0.99 the success bonus is worth <1% of its nominal value when seen from the start of the episode.

**Fix:** `gamma=0.995` for grasp, `gamma=0.997` for peg. Watch `train/value_loss` — higher γ amplifies bootstrap error.

### M6. Default argparse `--num-envs` is more aggressive than the sanity script

- `train_grasp.py:121` — argparse default 2048
- `train_peg.py:134` — argparse default 2048
- `runpod_sanity_all_tasks.md:55, 59` — sanity uses 256 envs

A 4090 has 24 GB VRAM with `MEM_FRACTION=0.7` ≈ 17 GB available. 2048 envs of MJX state for the Shadow Hand model is borderline — and SAC's 1M replay buffer adds another ~1 GB. Worth verifying that `--num-envs 2048` actually fits before running an 8‑hour grasp job.

**Fix:** flip argparse default to 256 to match the sanity protocol; require `--num-envs 2048` to be explicit. Or document that 2048 has only been validated on A100/H100, not 4090.

### M7. SAC replay buffer freshness vs curriculum cadence

- `config.py:303` — `buffer_size = 1_000_000`
- Curriculum at 8M, 16M, 24M, 32M (`config.py:320-327`)
- With 2048 envs, 1M buffer fills in ~500 vectorized steps (a few minutes). Buffer holds the last few minutes of training — far less than one curriculum stage.

When `clearance` drops 0.002 → 0.001 at 24M, the buffer has exclusively 0.002 data, and SAC immediately starts seeing 0.001 transitions which look out‑of‑distribution. The Q‑function may take a long time to recover.

**Fix:** lower priority. Either bump `buffer_size` to 5M (more cross‑stage data), or stagger the curriculum changes (also see M9 below). At minimum, log `metrics/clearance` so you can see whether returns regress on stage transitions.

### M8. Logging gaps: ent_coef, Q‑value, success_rate aggregation

- SBX SAC logs `train/ent_coef` and `train/critic_loss` by default; verify these reach W&B (the `RewardInfoLoggerCallback` only forwards keys starting with `reward/` or `metrics/` from `info`, not SB3's own keys).
- `info["is_success"]` exists for peg but is never aggregated into a scalar `train/success_rate` — VecMonitor logs it on episode close but the rollout‑level mean is what you want for live curves.

**Fix:** add a small callback that maintains a running `success_rate` over the last N episodes and logs it each rollout — useful for both grasp (once C2 is fixed) and peg.

### M9. Curriculum clearance jump 0.002 → 0.001 at 24M is a 50 % step

- `config.py:325-326` — `(24_000_000, 0.002, 0.7), (32_000_000, 0.001, 0.8)`

`p_pre_grasped` increases (easier reset) at the same time clearance halves (harder geometry). Mixed‑direction change confounds the cause of any success‑rate drop.

**Fix:** intermediate stage at 28M with `(0.0015, 0.75)`, or stagger the two parameters by 4M each.

---

## LOW / minor

### L1. `assert obs.shape == ...` inside `_get_obs_single`

- `grasp_env.py:245-247` — Python `assert` inside a JIT‑compiled function. Static asserts are fine under JIT (they're checked at trace time and erased), but if shape ever became dynamic this would fail confusingly. Worth a comment that this is a trace‑time check.

### L2. No top‑level torch / JAX seeding beyond SBX/MJX defaults

- Neither `train_peg.py` nor `train_grasp.py` calls `torch.manual_seed` / `np.random.seed` explicitly. SBX seeds via `seed=config.seed` and `mjx_vec_env.py:57` seeds JAX, but for cross‑pod reproducibility you'd want all three. Low priority on RunPod ephemeral pods.

### L3. Eval determinism

- `make_cpu_eval_env` (`scripts/training/_common.py:39-48`) seeds the env on construction but `EvalCallback` calls `reset()` each evaluation without re‑seeding. Eval episodes sample new initial conditions stochastically; "best model" is selected against a noisy 5‑ or 20‑episode mean (M4 amplifies this for peg).

### L4. CPU↔GPU transfer per step

- `mjx_vec_env.py:241-252` — every step does `np.array(jax_array)` to materialize obs/reward/done on CPU for SB3. Forced device→host sync. Probably necessary for SB3 compatibility, but worth knowing this is the throughput floor (~160 fps according to the runpod doc).

### L5. `policy_kwargs` for SAC re‑uses `net_arch.copy()` for both `pi` and `qf`

- `train_peg.py:74-76`. Fine, but if you ever want a wider Q network than policy network (common for SAC stability), this won't catch it.

---

## Quick‑hit fix list (smallest blast radius first)

1. **C1**: `config.py:313` — `MjxPegTrainConfig.norm_reward = False`. Single bool flip.
2. **C2 (part 1)**: `dexterous_hand/envs/gpu/grasp_env.py:_step_single` — add `info["is_success"] = is_success` so the vec env truncates on success. Need to also return `is_success` from `grasp_reward()` (`rewards/gpu/grasp_reward.py:140-156`).
3. **C2 (part 2)**: in `grasp_reward()`, make `success_bonus` fire on rising edge by tracking `was_success_prev` in `GraspRewardState`.
4. **H1**: `config.py:308` — `gradient_steps = 128` (or scale with `num_envs / 4`).
5. **H2 (sigmoids)**: `config.py:46-47` — `hold_height_smoothness_k = 20`, `hold_velocity_smoothness_k = 20`.
6. **H2 (lateral)**: `config.py:172` — `lateral_gate_k = 5.0`.
7. **H2 (complete cliff)**: `rewards/gpu/peg_reward.py:121` — replace `jnp.where` cliff with `complete_bonus * sigmoid(20*(insertion_fraction - 0.7)) * sigmoid(new_hold/5 - 1)`.
8. **H3**: `config.py:251` — `MjxGraspTrainConfig.norm_reward = False`.
9. **M5**: `config.py:242` → 0.995, `config.py:306` → 0.997.
10. **M2**: `mjx_vec_env.py` — add `info["metrics/nan_rate"] = bad.astype(jnp.float32)` so the runpod pass bar is verifiable.

## Verification plan

After applying fixes, repeat the runpod sanity protocol from `assets/shadow_hand/runpod_sanity_all_tasks.md`:

1. **Smoke**: `pytest tests/gpu` — confirms reward shapes and env step still match.
2. **Peg sanity (1M, ~1.6 hr on 4090)**: `uv run python main.py train-peg-mjx --num-envs 256 --total-timesteps 1000000`. Watch:
   - `train/ent_coef` should follow auto‑schedule, not pin to a low constant (this is the C1 regression test).
   - `metrics/insertion_depth` should rise above noise floor by 500K.
   - New: `train/metrics/nan_rate` < 0.01 (M2 makes this checkable).
   - New: `train/success_rate` non‑zero by end of run (M8).
3. **Grasp sanity (5M, ~8.7 hr on 4090)**: `uv run python main.py train-grasp-mjx --num-envs 256 --total-timesteps 5000000`. Watch:
   - `ep_len_mean` should converge to a value < 200 once the agent learns to succeed (this is the C2 regression test — successful episodes now truncate).
   - `ep_rew_mean` curve should be **bounded and stable** rather than spiking on each success (C2 + H3).
   - `metrics/object_height` peaks ≥ 0.50 m and `success_hold_steps > 0`.
4. **Reward sanity probe**: spawn a random‑policy rollout of 10K steps on each env; print mean/std of each `reward/*` component. Confirm no single component dominates > 80 % of total — that signals tuning is balanced.
5. **Ablation toggles** — keep these on hand from the runpod doc's §9 rollback table:
   - Peg policy collapses → `ent_coef=0.1` (fixed floor).
   - Grasp contacts plateau ≤ 2.5 → strengthen `weights.opposition` or relax `at_target`.
