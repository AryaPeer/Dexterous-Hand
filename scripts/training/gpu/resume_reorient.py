
import argparse
from pathlib import Path
from types import SimpleNamespace

from sbx import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from dexterous_hand.config import MjxReorientTrainConfig
from dexterous_hand.curriculum.callbacks import (
    ReorientCurriculumCallback,
    scale_stage_starts,
)
import dexterous_hand.envs  # noqa: F401
from dexterous_hand.envs.gpu.reorient_env import ShadowHandReorientMjxEnv
from scripts.training._common import (
    RewardInfoLoggerCallback,
    VecNormSyncEvalCallback,
    compute_eval_freq,
    make_cpu_eval_env,
    setup_sb3_logger,
)


def train(args: SimpleNamespace) -> None:

    model_path = Path(args.model_path).expanduser().resolve()
    vec_norm_path = Path(args.vec_normalize_path).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"model not found at {model_path}")
    if not vec_norm_path.exists():
        raise FileNotFoundError(f"vec_normalize not found at {vec_norm_path}")

    if args.output_dir:
        run_dir = Path(args.output_dir).expanduser().resolve()
    else:
        src = model_path.parent
        run_dir = src.with_name(src.name + "_resumed")
    run_dir.mkdir(parents=True, exist_ok=True)

    config = MjxReorientTrainConfig(num_envs=args.num_envs, seed=args.seed)

    curriculum_stages = scale_stage_starts(
        stages=config.curriculum_stages,
        total_timesteps=config.total_timesteps,
        reference_total_timesteps=config.curriculum_reference_timesteps,
    )

    vec_env = ShadowHandReorientMjxEnv.from_config(config)
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
    vec_env.training = True
    vec_env.norm_reward = config.norm_reward

    curriculum_callback = ReorientCurriculumCallback(
        stages=curriculum_stages,
        verbose=1,
    )

    model = PPO.load(str(model_path), env=vec_env)

    setup_sb3_logger(model, run_dir)

    eval_env = make_cpu_eval_env(
        env_id="ShadowHandReorient-v0",
        seed=config.seed + 20_000,
        scene_config=config.scene_config,
        reward_config=config.reward_config,
        norm_obs=config.norm_obs,
    )

    callbacks = [
        curriculum_callback,
        RewardInfoLoggerCallback(),
        VecNormSyncEvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best"),
            eval_freq=compute_eval_freq(args.additional_timesteps, config.num_envs),
            n_eval_episodes=10,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=max(500_000 // config.num_envs, 1),
            save_path=str(run_dir / "checkpoints"),
        ),
    ]

    print(f"Resuming from {model_path} for {args.additional_timesteps:,} additional timesteps.")
    print(f"Output dir: {run_dir}")

    model.learn(
        total_timesteps=args.additional_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    model.save(str(run_dir / "final_model"))
    vec_env.save(str(run_dir / "vec_normalize.pkl"))

    print(f"Saved to {run_dir}")
    vec_env.close()
    eval_env.close()

def parse_args() -> SimpleNamespace:
    parser = argparse.ArgumentParser(description="Resume Shadow Hand reorientation (MJX + SBX PPO)")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to final_model.zip (or any checkpoint .zip)")
    parser.add_argument("--vec-normalize-path", type=str, required=True,
                        help="Path to vec_normalize.pkl saved alongside the model")
    parser.add_argument("--additional-timesteps", type=int, required=True,
                        help="How many MORE timesteps to train (additional, not cumulative)")
    parser.add_argument("--num-envs", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to save resumed run (default: <input_dir>_resumed)")
    return parser.parse_args()

if __name__ == "__main__":
    train(parse_args())
