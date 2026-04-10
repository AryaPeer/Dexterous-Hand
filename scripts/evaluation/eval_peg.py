import argparse
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dexterous_hand.config import PegSceneConfig
import dexterous_hand.envs  # noqa: F401


def evaluate(
    model_path: str,
    vec_normalize_path: str | None = None,
    n_episodes: int = 100,
    clearance: float = 0.001,
    record_video: bool = True,
    video_dir: str = "videos",
) -> None:
    """Run peg-in-hole evaluation, reporting success/drop rates and insertion metrics."""

    model = SAC.load(model_path)
    scene_config = PegSceneConfig(clearance=clearance)
    env = gym.make(
        "ShadowHandPeg-v0",
        render_mode="rgb_array" if record_video else None,
        scene_config=scene_config,
    )

    vec_env: DummyVecEnv | VecNormalize | None = None
    if vec_normalize_path and Path(vec_normalize_path).exists():
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)  # type: ignore[assignment]
        vec_env.training = False  # type: ignore[union-attr]
        vec_env.norm_reward = False  # type: ignore[union-attr]

    successes = 0
    drops = 0
    contact_forces: list[float] = []
    insertion_times: list[int] = []

    video_frames: list[np.ndarray] = []
    episodes_recorded = 0
    max_video_episodes = 5

    # run episodes
    for _ep in range(n_episodes):
        obs, info = env.reset()
        ep_steps = 0
        success = False
        dropped = False
        insertion_start: int | None = None

        for step_i in range(500):
            obs_norm = vec_env.normalize_obs(obs) if vec_env is not None else obs  # type: ignore[union-attr]
            action, _ = model.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_steps += 1

            # track insertion stage and contact forces
            stage = info.get("metrics/stage", info.get("stage", 0))
            cf = info.get("metrics/contact_force", 0.0)
            if stage >= 3:
                contact_forces.append(cf)
                if insertion_start is None:
                    insertion_start = step_i

            # check insertion hold
            hold = info.get("metrics/insertion_hold_steps", 0)
            if hold >= 10:
                success = True

            if record_video and episodes_recorded < max_video_episodes:
                frame: np.ndarray = env.render()  # type: ignore[assignment]
                if frame is not None:
                    video_frames.append(frame)  # type: ignore[arg-type]

            if terminated:
                if not success:
                    dropped = True
                break
            if truncated:
                break

        # accumulate episode stats
        if success:
            successes += 1
            if insertion_start is not None:
                insertion_times.append(ep_steps - insertion_start)
        if dropped:
            drops += 1

        if record_video and episodes_recorded < max_video_episodes:
            episodes_recorded += 1

    # results
    success_rate = 100 * successes / max(n_episodes, 1)
    drop_rate = 100 * drops / max(n_episodes, 1)
    mean_force = float(np.mean(contact_forces)) if contact_forces else 0.0
    mean_ins_time = float(np.mean(insertion_times)) if insertion_times else 0.0

    print(f"\nPeg eval (clearance={clearance * 1000:.1f}mm, n={n_episodes})")
    print(f"  success={success_rate:.1f}%  drops={drop_rate:.1f}%  force={mean_force:.2f}N  ins_time={mean_ins_time:.1f}steps")

    # save video
    if record_video and video_frames:
        vdir = Path(video_dir)
        vdir.mkdir(parents=True, exist_ok=True)
        video_path = vdir / f"peg_evaluation_{clearance * 1000:.0f}mm.mp4"
        imageio.mimsave(str(video_path), video_frames, fps=25)  # type: ignore[arg-type]
        print(f"\n  Video saved to {video_path}")

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate peg-in-hole model")
    parser.add_argument("--model-path", required=True, help="Path to saved model .zip")
    parser.add_argument("--vec-normalize-path", default=None)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--clearance", type=float, default=0.001, help="Hole clearance in meters")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        vec_normalize_path=args.vec_normalize_path,
        n_episodes=args.n_episodes,
        clearance=args.clearance,
        record_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
