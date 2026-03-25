import argparse
from collections import defaultdict
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
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    env = gym.make("ShadowHandGrasp-v0", render_mode="rgb_array" if record_video else None)

    vec_env: DummyVecEnv | VecNormalize | None = None
    if vec_normalize_path and Path(vec_normalize_path).exists():
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)  # type: ignore[assignment]
        vec_env.training = False  # type: ignore[union-attr]
        vec_env.norm_reward = False  # type: ignore[union-attr]

    results: dict[str, dict[str, int]] = defaultdict(lambda: {"grasp": 0, "lift": 0, "total": 0})

    video_frames: list[np.ndarray] = []
    episodes_recorded = 0
    max_video_episodes = 5

    for _ep in range(n_episodes):
        obs, info = env.reset()
        obj_type = info["object_type"]
        results[obj_type]["total"] += 1

        frames: list[np.ndarray] = []
        grasped = False
        lift_steps = 0
        lifted = False

        for _step_i in range(200):
            obs_norm = vec_env.normalize_obs(obs) if vec_env is not None else obs  # type: ignore[union-attr]

            action, _ = model.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            n_contacts = info.get("metrics/num_finger_contacts", 0)
            if n_contacts >= 3:
                grasped = True

            obj_h = info.get("metrics/object_height", 0)
            table_h = env.unwrapped.scene_config.table_height  # type: ignore[attr-defined]
            if obj_h > table_h + 0.1:
                lift_steps += 1
                if lift_steps >= 50:
                    lifted = True
            else:
                lift_steps = 0

            if record_video and episodes_recorded < max_video_episodes:
                frame: np.ndarray = env.render()  # type: ignore[assignment]
                if frame is not None:
                    frames.append(frame)  # type: ignore[arg-type]

            if terminated or truncated:
                break

        if grasped:
            results[obj_type]["grasp"] += 1
        if lifted:
            results[obj_type]["lift"] += 1

        if record_video and frames and episodes_recorded < max_video_episodes:
            video_frames.extend(frames)
            episodes_recorded += 1

    print("\n=== Evaluation Results ===\n")
    print(f"{'Object Type':<18} {'Grasp %':>10} {'Lift %':>10} {'Episodes':>10}")
    print("-" * 52)

    total_grasp = 0
    total_lift = 0
    total_eps = 0
    for obj_type in sorted(results.keys()):
        r = results[obj_type]
        total_eps += r["total"]
        total_grasp += r["grasp"]
        total_lift += r["lift"]
        grasp_pct = 100 * r["grasp"] / max(r["total"], 1)
        lift_pct = 100 * r["lift"] / max(r["total"], 1)
        print(f"{obj_type:<18} {grasp_pct:>9.1f}% {lift_pct:>9.1f}% {r['total']:>10}")

    print("-" * 52)
    overall_grasp = 100 * total_grasp / max(total_eps, 1)
    overall_lift = 100 * total_lift / max(total_eps, 1)
    print(f"{'Overall':<18} {overall_grasp:>9.1f}% {overall_lift:>9.1f}% {total_eps:>10}")

    if record_video and video_frames:
        vdir = Path(video_dir)
        vdir.mkdir(parents=True, exist_ok=True)
        video_path = vdir / "evaluation.mp4"
        imageio.mimsave(str(video_path), video_frames, fps=25)  # type: ignore[arg-type]
        print(f"\nVideo saved to {video_path}")

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained grasping model")
    parser.add_argument("--model-path", required=True, help="Path to saved model .zip")
    parser.add_argument("--vec-normalize-path", default=None, help="Path to VecNormalize stats")
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
