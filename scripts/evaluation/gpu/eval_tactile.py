
import argparse
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
from sbx import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dexterous_hand.config import PegSceneConfig
import dexterous_hand.envs  # noqa: F401

def run_episodes(
    model: SAC,  # type: ignore[type-arg]
    env_id: str,
    scene_config: PegSceneConfig,
    vec_normalize_path: str | None,
    n_episodes: int,
    record_video: bool = False,
) -> dict:

    env = gym.make(
        env_id, render_mode="rgb_array" if record_video else None, scene_config=scene_config
    )

    vec_env: DummyVecEnv | VecNormalize | None = None
    if vec_normalize_path and Path(vec_normalize_path).exists():
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)  # type: ignore[assignment]
        vec_env.training = False  # type: ignore[union-attr]
        vec_env.norm_reward = False  # type: ignore[union-attr]

    successes = 0
    drops = 0
    slip_events = 0
    contact_forces: list[float] = []
    insertion_times: list[int] = []
    video_frames: list[np.ndarray] = []

    max_steps = 500
    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_steps = 0
        success = False
        insertion_start: int | None = None

        for step_i in range(max_steps):
            obs_norm = vec_env.normalize_obs(obs) if vec_env is not None else obs  # type: ignore[union-attr]
            action, _ = model.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_steps += 1

            stage = info.get("metrics/stage", info.get("stage", 0))
            cf = info.get("metrics/contact_force", 0.0)
            if stage >= 3:
                contact_forces.append(cf)
                if insertion_start is None:
                    insertion_start = step_i

            if info.get("metrics/slip_detected", 0.0) > 0:
                slip_events += 1

            hold = info.get("metrics/insertion_hold_steps", 0)
            if hold >= 10:
                success = True

            if record_video and ep < 3:
                frame: np.ndarray = env.render()  # type: ignore[assignment]
                if frame is not None:
                    video_frames.append(frame)  # type: ignore[arg-type]

            if terminated or truncated:
                break

        if success:
            successes += 1
            if insertion_start is not None:
                insertion_times.append(ep_steps - insertion_start)
        elif terminated:
            drops += 1

    env.close()

    return {
        "success_rate": 100 * successes / max(n_episodes, 1),
        "drop_rate": 100 * drops / max(n_episodes, 1),
        "mean_contact_force": float(np.mean(contact_forces)) if contact_forces else 0.0,
        "slip_events_per_ep": slip_events / max(n_episodes, 1),
        "mean_insertion_time": float(np.mean(insertion_times)) if insertion_times else 0.0,
        "video_frames": video_frames,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Tactile vs baseline comparison (SBX models)")
    parser.add_argument("--tactile-model", required=True, help="Path to SBX tactile model .zip")
    parser.add_argument("--baseline-model", required=True, help="Path to SBX baseline model .zip")
    parser.add_argument("--tactile-vec-normalize", default=None)
    parser.add_argument("--baseline-vec-normalize", default=None)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    tactile_model = SAC.load(args.tactile_model)
    baseline_model = SAC.load(args.baseline_model)

    conditions = [
        ("1mm clearance", PegSceneConfig(clearance=0.001)),
        ("slippery (μ=0.3)", PegSceneConfig(clearance=0.001, peg_friction=(0.3, 0.001, 0.001))),
    ]

    all_results: dict[str, dict[str, dict]] = {}

    for cond_name, scene_config in conditions:
        tactile_results = run_episodes(
            tactile_model,
            "ShadowHandPegTactile-v0",
            scene_config,
            args.tactile_vec_normalize,
            args.n_episodes,
            record_video=not args.no_video,
        )

        baseline_results = run_episodes(
            baseline_model,
            "ShadowHandPeg-v0",
            scene_config,
            args.baseline_vec_normalize,
            args.n_episodes,
        )

        all_results[cond_name] = {"tactile": tactile_results, "baseline": baseline_results}

    for cond_name, variants in all_results.items():
        t = variants["tactile"]
        b = variants["baseline"]

        print(f"\n{cond_name}:")
        print(f"  {'':>26} {'tactile':>10} {'baseline':>10}")
        print(f"  {'success':>26} {t['success_rate']:>9.1f}% {b['success_rate']:>9.1f}%")
        print(f"  {'drops':>26} {t['drop_rate']:>9.1f}% {b['drop_rate']:>9.1f}%")
        print(
            f"  {'contact force (N)':>26} {t['mean_contact_force']:>10.2f} {b['mean_contact_force']:>10.2f}"
        )
        print(
            f"  {'slips/ep':>26} {t['slip_events_per_ep']:>10.2f} {b['slip_events_per_ep']:>10.2f}"
        )
        print(
            f"  {'insertion time':>26} {t['mean_insertion_time']:>10.1f} {b['mean_insertion_time']:>10.1f}"
        )

    if not args.no_video:
        vdir = Path("videos")
        vdir.mkdir(parents=True, exist_ok=True)
        for cond_name, variants in all_results.items():
            frames = variants["tactile"].get("video_frames", [])
            if frames:
                safe_name = (
                    cond_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
                )
                path = vdir / f"tactile_mjx_{safe_name}.mp4"
                imageio.mimsave(str(path), frames, fps=25)  # type: ignore[arg-type]
                print(f"\nVideo saved to {path}")

if __name__ == "__main__":
    main()
