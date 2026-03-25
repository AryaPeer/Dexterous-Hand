import argparse
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import dexterous_hand.envs  # noqa: F401


def evaluate(
    model_path: str,
    vec_normalize_path: str | None = None,
    n_episodes: int = 100,
    record_video: bool = True,
    video_dir: str = "videos",
) -> None:
    """Run reorientation evaluation, reporting target hit rates and angular accuracy."""

    model = PPO.load(model_path)
    env = gym.make("ShadowHandReorient-v0", render_mode="rgb_array" if record_video else None)

    vec_env: DummyVecEnv | VecNormalize | None = None
    if vec_normalize_path and Path(vec_normalize_path).exists():
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)  # type: ignore[assignment]
        vec_env.training = False  # type: ignore[union-attr]
        vec_env.norm_reward = False  # type: ignore[union-attr]

    total_targets = 0
    total_drops = 0
    total_steps = 0
    angular_distances: list[float] = []
    steps_to_first_target: list[int] = []

    video_frames: list[np.ndarray] = []
    episodes_recorded = 0
    max_video_episodes = 5

    # run episodes
    for _ep in range(n_episodes):
        obs, info = env.reset()
        ep_targets = 0
        ep_steps = 0
        first_target_step: int | None = None
        dropped = False

        for step_i in range(400):
            obs_norm = vec_env.normalize_obs(obs) if vec_env is not None else obs  # type: ignore[union-attr]
            action, _ = model.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_steps += 1

            # track target hits
            if info.get("targets_reached", 0) > ep_targets:
                if first_target_step is None:
                    first_target_step = step_i
                ep_targets = info["targets_reached"]

            # track angular distance
            ang_dist = info.get("metrics/angular_distance", None)
            if ang_dist is not None:
                angular_distances.append(ang_dist)

            if record_video and episodes_recorded < max_video_episodes:
                frame: np.ndarray = env.render()  # type: ignore[assignment]
                if frame is not None:
                    video_frames.append(frame)  # type: ignore[arg-type]

            if terminated:
                dropped = True
                break
            if truncated:
                break

        # accumulate episode stats
        total_targets += ep_targets
        if dropped:
            total_drops += 1
        total_steps += ep_steps
        if first_target_step is not None:
            steps_to_first_target.append(first_target_step)

        if record_video and episodes_recorded < max_video_episodes:
            episodes_recorded += 1

    # results
    success_rate = 100 * len(steps_to_first_target) / max(n_episodes, 1)
    drop_rate = 100 * total_drops / max(n_episodes, 1)
    mean_targets = total_targets / max(n_episodes, 1)
    mean_ang_dist = float(np.mean(angular_distances)) if angular_distances else 0.0
    mean_steps = float(np.mean(steps_to_first_target)) if steps_to_first_target else 400.0

    print(f"\nReorient eval (n={n_episodes})")
    print(f"  success={success_rate:.1f}%  drops={drop_rate:.1f}%  targets/ep={mean_targets:.2f}")
    print(f"  ang_dist={mean_ang_dist:.3f}rad  steps_to_target={mean_steps:.1f}")

    # save video
    if record_video and video_frames:
        vdir = Path(video_dir)
        vdir.mkdir(parents=True, exist_ok=True)
        video_path = vdir / "reorient_evaluation.mp4"
        imageio.mimsave(str(video_path), video_frames, fps=25)  # type: ignore[arg-type]
        print(f"\n  Video saved to {video_path}")

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reorientation model")
    parser.add_argument("--model-path", required=True, help="Path to saved model .zip")
    parser.add_argument("--vec-normalize-path", default=None)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        vec_normalize_path=args.vec_normalize_path,
        n_episodes=args.n_episodes,
        record_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
