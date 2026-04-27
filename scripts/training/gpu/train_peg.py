
import argparse
from dataclasses import asdict
from pathlib import Path

from sbx import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback

import dexterous_hand.envs  # noqa: F401  - register gym ids for CPU eval env
from dexterous_hand.config import MjxPegTrainConfig
from dexterous_hand.curriculum.callbacks import (
    AssemblyCurriculumCallback,
    scale_stage_starts,
)
from dexterous_hand.envs.gpu.peg_env import ShadowHandPegMjxEnv
from scripts.training._common import (
    RewardInfoLoggerCallback,
    VecNormSyncEvalCallback,
    compute_eval_freq,
    make_cpu_eval_env,
    setup_sb3_logger,
)

def train(config: MjxPegTrainConfig) -> None:

    run_dir = Path("runs") / f"peg_mjx_{config.num_envs}env_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    curriculum_stages = scale_stage_starts(
        stages=config.curriculum_stages,
        total_timesteps=config.total_timesteps,
        reference_total_timesteps=config.curriculum_reference_timesteps,
    )

    wandb_config = asdict(config)
    wandb_config["effective_curriculum_stages"] = curriculum_stages
    wandb.init(
        project="dexterous-hand",
        name=f"peg-mjx-{config.num_envs}env",
        config=wandb_config,
    )

    vec_env = ShadowHandPegMjxEnv.from_config(config)
    vec_env = VecMonitor(vec_env)

    if config.norm_obs or config.norm_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=config.norm_obs,
            norm_reward=config.norm_reward,
            clip_obs=10.0,
        )

    curriculum_callback = AssemblyCurriculumCallback(
        stages=curriculum_stages,
        verbose=1,
    )

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
        },
        verbose=1,
        seed=config.seed,
    )

    setup_sb3_logger(model, run_dir)

    stage0_clearance, stage0_p_pre_grasped = curriculum_stages[0][1], curriculum_stages[0][2]
    eval_env = make_cpu_eval_env(
        env_id="ShadowHandPeg-v0",
        seed=config.seed + 10_000,
        scene_config=config.scene_config,
        reward_config=config.reward_config,
        norm_obs=config.norm_obs,
        post_make=lambda env: env.set_curriculum_params(
            clearance=stage0_clearance,
            p_pre_grasped=stage0_p_pre_grasped,
        ),
    )

    callbacks = [
        curriculum_callback,
        RewardInfoLoggerCallback(),
        VecNormSyncEvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best"),
            eval_freq=compute_eval_freq(config.total_timesteps, config.num_envs),
            n_eval_episodes=5,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=max(500_000 // config.num_envs, 1),
            save_path=str(run_dir / "checkpoints"),
        ),
        WandbCallback(
            model_save_path=str(run_dir),
            model_save_freq=max(100_000 // config.num_envs, 1),
            verbose=1,
        ),
    ]

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(str(run_dir / "final_model"))
    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(run_dir / "vec_normalize.pkl"))

    print(f"Saved to {run_dir}")
    wandb.finish()
    vec_env.close()
    eval_env.close()

def parse_args() -> MjxPegTrainConfig:
    parser = argparse.ArgumentParser(description="Train Shadow Hand peg-in-hole (MJX + SBX SAC)")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--total-timesteps", type=int, default=60_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return MjxPegTrainConfig(
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )

if __name__ == "__main__":
    config = parse_args()
    train(config)
