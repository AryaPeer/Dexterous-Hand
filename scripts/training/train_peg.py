import argparse
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import torch

from dexterous_hand.config import PegTrainConfig
from dexterous_hand.curriculum.callbacks import AssemblyCurriculumCallback
import dexterous_hand.envs  # noqa: F401


def make_env(rank: int, seed: int) -> Callable[[], gym.Env]:  # type: ignore[type-arg]
    def _init() -> gym.Env:  # type: ignore[type-arg]
        env = gym.make("ShadowHandPeg-v0")
        env.reset(seed=seed + rank)
        return env

    return _init


def train(config: PegTrainConfig) -> None:
    run_dir = Path("runs") / f"peg_{config.n_envs}env_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        import wandb

        wandb.init(
            project="dexterous-hand",
            name=f"peg-{config.n_envs}env",
            config=asdict(config),
        )
        use_wandb = True
    except Exception:
        print("WandB not available or not logged in, continuing without it.")
        use_wandb = False

    print(f"Creating {config.n_envs} parallel environments...")
    env_fns = [make_env(i, config.seed) for i in range(config.n_envs)]
    vec_env = SubprocVecEnv(env_fns) if config.n_envs > 1 else DummyVecEnv(env_fns)

    if config.norm_obs or config.norm_reward:
        vec_env = VecNormalize(  # type: ignore[assignment]
            vec_env,
            norm_obs=config.norm_obs,
            norm_reward=config.norm_reward,
            clip_obs=10.0,
        )

    eval_env = DummyVecEnv([make_env(0, config.seed + 10000)])
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

    print("Initializing SAC...")
    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        tau=config.tau,
        gamma=config.gamma,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        ent_coef=config.ent_coef,
        policy_kwargs={
            "net_arch": dict(pi=config.net_arch.copy(), qf=config.net_arch.copy()),
            "activation_fn": activation_fn,
        },
        verbose=1,
        seed=config.seed,
        device="auto",
    )

    curriculum_callback = AssemblyCurriculumCallback(
        stages=config.curriculum_stages,
        verbose=1,
    )

    callbacks = [
        curriculum_callback,
        EvalCallback(
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
    ]

    if use_wandb:
        try:
            from wandb.integration.sb3 import WandbCallback

            callbacks.append(
                WandbCallback(
                    model_save_path=str(run_dir),
                    model_save_freq=max(100_000 // config.n_envs, 1),
                    verbose=1,
                )
            )
        except ImportError:
            pass

    print(f"Training for {config.total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(str(run_dir / "final_model"))
    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(run_dir / "vec_normalize.pkl"))

    print(f"Training complete. Model saved to {run_dir}")
    if use_wandb:
        wandb.finish()

    vec_env.close()
    eval_env.close()


def parse_args() -> PegTrainConfig:
    parser = argparse.ArgumentParser(description="Train Shadow Hand peg-in-hole (SAC)")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=100_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return PegTrainConfig(
        n_envs=args.n_envs,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
