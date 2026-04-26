from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize


class VecNormSyncEvalCallback(EvalCallback):
    """EvalCallback that keeps eval obs normalization in sync with training env."""

    def _on_step(self) -> bool:
        if isinstance(self.training_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            self.eval_env.obs_rms = self.training_env.obs_rms
        return super()._on_step()


def make_cpu_eval_env(
    env_id: str,
    seed: int,
    scene_config: Any,
    reward_config: Any,
    norm_obs: bool,
    post_make: callable | None = None,
) -> VecNormalize | VecMonitor:
    """Spin up a single-worker CPU mujoco env wrapped as VecNormalize(training=False).

    Used by GPU training scripts so SB3's EvalCallback can save best-of-eval
    checkpoints without needing a second MJX instance. ``post_make`` is an
    optional callable applied to the unwrapped gym env after construction
    (e.g. to pin the peg curriculum to stage 0)."""

    def _init() -> gym.Env:  # type: ignore[type-arg]
        env = gym.make(
            env_id,
            scene_config=deepcopy(scene_config),
            reward_config=deepcopy(reward_config),
        )
        if post_make is not None:
            post_make(env.unwrapped)
        env.reset(seed=seed)
        return env

    eval_env: Any = DummyVecEnv([_init])
    eval_env = VecMonitor(eval_env)
    if norm_obs:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            training=False,
        )
    return eval_env


def compute_eval_freq(total_timesteps: int, num_envs: int, target_evals: int = 25) -> int:
    """Eval-callback frequency that scales sensibly across sanity vs full runs.

    Returns ``eval_freq`` in *vec-steps* (what ``EvalCallback`` actually counts).
    Aims for ~``target_evals`` evals across the whole run, with a minimum spacing
    of 500K timesteps so short sanity runs don't hammer the CPU eval env.
    """
    interval_timesteps = max(total_timesteps // target_evals, 500_000)
    return max(interval_timesteps // num_envs, 1)


class RewardInfoLoggerCallback(BaseCallback):
    """Aggregates per-component reward/metric info dicts from vec env step and
    records their mean onto SB3's logger each rollout.

    Picks up any key starting with ``reward/`` or ``metrics/`` that appears in
    ``infos`` and emits it as ``train/<key>`` at rollout end, alongside SB3's
    built-in ``rollout/`` and ``time/`` sections.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._buf: dict[str, list[float]] = defaultdict(list)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            for k, v in info.items():
                if not (k.startswith("reward/") or k.startswith("metrics/")):
                    continue
                try:
                    self._buf[k].append(float(v))
                except (TypeError, ValueError):
                    continue
        return True

    def _on_rollout_end(self) -> None:
        for k, vals in self._buf.items():
            if not vals:
                continue
            self.logger.record(f"train/{k}", float(np.mean(vals)))
        self._buf.clear()


def setup_sb3_logger(model, run_dir: Path) -> None:
    """Point SB3's logger at ``run_dir/logs`` so stdout stays pretty and a CSV
    is written next to the model. CSV is easy to paste into chat or grep later.
    """
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    new_logger = configure(str(log_dir), ["stdout", "csv"])
    model.set_logger(new_logger)
