import argparse
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

from dexterous_hand.config import ReorientTrainConfig
from dexterous_hand.curriculum.callbacks import (
    ReorientCurriculumCallback,
    scale_stage_starts,
)
import dexterous_hand.envs  # noqa: F401


class VecNormSyncEvalCallback(EvalCallback):
    """EvalCallback that keeps eval obs normalization in sync with training env."""

    def _on_step(self) -> bool:
        if isinstance(self.training_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            self.eval_env.obs_rms = self.training_env.obs_rms
        return super()._on_step()


def make_env(rank: int, seed: int, config: ReorientTrainConfig) -> Callable[[], gym.Env]:  # type: ignore[type-arg]
    def _init() -> gym.Env:  # type: ignore[type-arg]
        import dexterous_hand.envs  # noqa: F401,F811 - register in subprocess
        env = gym.make(
            "ShadowHandReorient-v0",
            scene_config=deepcopy(config.scene_config),
            reward_config=deepcopy(config.reward_config),
        )
        env.reset(seed=seed + rank)
        return env

    return _init


def train(config: ReorientTrainConfig) -> None:

    run_dir = Path("runs") / f"reorient_{config.n_envs}env_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # clamp batch_size for small local runs
    rollout_size = config.n_envs * config.n_steps_per_env
    if config.batch_size > rollout_size:
        config.batch_size = rollout_size
        print(f"Clamped batch_size to {config.batch_size} (n_envs * n_steps_per_env)")

    curriculum_stages = scale_stage_starts(
        stages=config.curriculum_stages,
        total_timesteps=config.total_timesteps,
        reference_total_timesteps=config.curriculum_reference_timesteps,
    )

    wandb_config = asdict(config)
    wandb_config["effective_curriculum_stages"] = curriculum_stages
    wandb.init(
        project="dexterous-hand",
        name=f"reorient-{config.n_envs}env",
        config=wandb_config,
    )

    # environments
    env_fns = [make_env(i, config.seed, config) for i in range(config.n_envs)]
    vec_env = SubprocVecEnv(env_fns) if config.n_envs > 1 else DummyVecEnv(env_fns)

    if config.norm_obs or config.norm_reward:
        vec_env = VecNormalize(  # type: ignore[assignment]
            vec_env,
            norm_obs=config.norm_obs,
            norm_reward=config.norm_reward,
            clip_obs=10.0,
        )

    eval_env = DummyVecEnv([make_env(0, config.seed + 10000, config)])
    if config.norm_obs or config.norm_reward:
        eval_env = VecNormalize(  # type: ignore[assignment]
            eval_env,
            norm_obs=config.norm_obs,
            norm_reward=False,
            clip_obs=10.0,
            training=False,
        )

    activation_fn = {"elu": torch.nn.ELU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}[
        config.activation
    ]

    # PPO
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps_per_env,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs={
            "net_arch": dict(pi=config.net_arch.copy(), vf=config.net_arch.copy()),
            "activation_fn": activation_fn,
        },
        verbose=1,
        seed=config.seed,
        device="auto",
    )

    # callbacks
    callbacks = [
        ReorientCurriculumCallback(stages=curriculum_stages, verbose=1),
        VecNormSyncEvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best"),
            eval_freq=max(50_000 // config.n_envs, 1),
            n_eval_episodes=20,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=max(500_000 // config.n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
        ),
        WandbCallback(
            model_save_path=str(run_dir),
            model_save_freq=max(100_000 // config.n_envs, 1),
            verbose=1,
        ),
    ]

    # train
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # save
    model.save(str(run_dir / "final_model"))
    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(run_dir / "vec_normalize.pkl"))

    print(f"Saved to {run_dir}")
    wandb.finish()

    vec_env.close()
    eval_env.close()


def parse_args() -> ReorientTrainConfig:
    parser = argparse.ArgumentParser(description="Train Shadow Hand reorientation")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=200_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--n-steps-per-env", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return ReorientTrainConfig(
        n_envs=args.n_envs,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps_per_env=args.n_steps_per_env,
        seed=args.seed,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
