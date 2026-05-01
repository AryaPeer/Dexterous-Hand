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
    interval_timesteps = max(total_timesteps // target_evals, 500_000)
    return max(interval_timesteps // num_envs, 1)


class RewardInfoLoggerCallback(BaseCallback):
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
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    new_logger = configure(str(log_dir), ["stdout", "csv"])
    model.set_logger(new_logger)
