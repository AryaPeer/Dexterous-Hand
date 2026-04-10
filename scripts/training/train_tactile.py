import argparse
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

from dexterous_hand.config import TactileTrainConfig
from dexterous_hand.curriculum.callbacks import (
    AssemblyCurriculumCallback,
    scale_stage_starts,
)
import dexterous_hand.envs  # noqa: F401
from dexterous_hand.tactile.feature_extractor import TactileFeatureExtractor


def make_env(
    env_id: str,
    rank: int,
    seed: int,
    config: TactileTrainConfig,
) -> Callable[[], gym.Env]:  # type: ignore[type-arg]
    def _init() -> gym.Env:  # type: ignore[type-arg]
        import dexterous_hand.envs  # noqa: F401,F811 - register in subprocess
        kwargs = {
            "scene_config": deepcopy(config.scene_config),
            "reward_config": deepcopy(config.reward_config),
        }
        if env_id == "ShadowHandPegTactile-v0":
            kwargs["tactile_config"] = deepcopy(config.tactile_config)

        env = gym.make(env_id, **kwargs)
        env.reset(seed=seed + rank)
        return env

    return _init


def train_variant(
    variant: str,
    config: TactileTrainConfig,
    run_dir: Path,
) -> None:
    """Train one variant (tactile or baseline)."""

    use_tactile = variant == "tactile"
    env_id = "ShadowHandPegTactile-v0" if use_tactile else "ShadowHandPeg-v0"
    variant_dir = run_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    curriculum_stages = scale_stage_starts(
        stages=config.curriculum_stages,
        total_timesteps=config.total_timesteps,
        reference_total_timesteps=config.curriculum_reference_timesteps,
    )

    wandb_config = asdict(config)
    wandb_config["variant"] = variant
    wandb_config["effective_curriculum_stages"] = curriculum_stages
    wandb.init(
        project="dexterous-hand",
        name=f"tactile-{variant}-{config.n_envs}env",
        config=wandb_config,
        reinit=True,
    )

    env_fns = [make_env(env_id, i, config.seed, config) for i in range(config.n_envs)]
    vec_env = SubprocVecEnv(env_fns) if config.n_envs > 1 else DummyVecEnv(env_fns)

    if config.norm_obs or config.norm_reward:
        vec_env = VecNormalize(  # type: ignore[assignment]
            vec_env,
            norm_obs=config.norm_obs,
            norm_reward=config.norm_reward,
            clip_obs=10.0,
        )

    eval_fns = [make_env(env_id, 0, config.seed + 10000, config)]
    eval_env = DummyVecEnv(eval_fns)
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

    policy_kwargs: dict[str, object] = {
        "net_arch": dict(pi=config.net_arch.copy(), qf=config.net_arch.copy()),
        "activation_fn": activation_fn,
    }
    if use_tactile:
        policy_kwargs["features_extractor_class"] = TactileFeatureExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "proprio_dim": 131,
            "tactile_dim": 240,
        }

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
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=config.seed,
        device="auto",
    )

    callbacks = [
        AssemblyCurriculumCallback(stages=curriculum_stages, verbose=1),
        EvalCallback(
            eval_env,
            best_model_save_path=str(variant_dir / "best"),
            eval_freq=max(50_000 // config.n_envs, 1),
            n_eval_episodes=20,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=max(500_000 // config.n_envs, 1),
            save_path=str(variant_dir / "checkpoints"),
        ),
        WandbCallback(
            model_save_path=str(variant_dir),
            model_save_freq=max(100_000 // config.n_envs, 1),
            verbose=1,
        ),
    ]

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(str(variant_dir / "final_model"))
    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(variant_dir / "vec_normalize.pkl"))

    print(f"Saved {variant} model to {variant_dir}")
    wandb.finish()

    vec_env.close()
    eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tactile ablation study")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=100_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variant",
        choices=["both", "tactile", "baseline"],
        default="both",
        help="Which variant to train (default: both)",
    )
    args = parser.parse_args()

    config = TactileTrainConfig(
        n_envs=args.n_envs,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
    )

    run_dir = Path("runs") / f"tactile_ablation_{config.n_envs}env_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    variants = ["tactile", "baseline"] if args.variant == "both" else [args.variant]
    for variant in variants:
        train_variant(variant, config, run_dir)

    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    main()
