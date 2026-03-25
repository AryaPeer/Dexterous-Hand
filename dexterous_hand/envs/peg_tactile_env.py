from typing import Any

from gymnasium import spaces
import numpy as np

from dexterous_hand.config import PegRewardConfig, PegSceneConfig, TactileConfig
from dexterous_hand.envs.peg_env import ShadowHandPegEnv
from dexterous_hand.tactile.sensor import TactileSensor
from dexterous_hand.utils.mujoco_helpers import get_object_state


class ShadowHandPegTactileEnv(ShadowHandPegEnv):
    """Peg-in-hole but with tactile sensing on the fingertips. Obs is 365-dim (125 proprio + 240 tactile)."""

    def __init__(
        self,
        render_mode: str | None = None,
        scene_config: PegSceneConfig | None = None,
        reward_config: PegRewardConfig | None = None,
        tactile_config: TactileConfig | None = None,
    ) -> None:
        """Sets up the peg env with tactile sensors added on top.

        Args:
            render_mode: 'human' for window, 'rgb_array' for offscreen
            scene_config: scene layout and physics
            reward_config: reward weights/thresholds
            tactile_config: sensor grid size, noise, etc.
        """
        super().__init__(
            render_mode=render_mode, scene_config=scene_config, reward_config=reward_config
        )

        self.tactile_config = tactile_config or TactileConfig()

        n_obs = 365
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float64
        )

        self._tactile_sensor: TactileSensor | None = None

    def _ensure_tactile(self) -> TactileSensor:
        """Create the tactile sensor if it doesn't exist yet (lazy init)."""
        if self._tactile_sensor is None:
            self._tactile_sensor = TactileSensor(
                model=self.model,
                config=self.tactile_config,
                rng=self.np_random,
            )
        return self._tactile_sensor

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset everything including the tactile sensor history.

        Args:
            seed: random seed
            options: extra options (unused)

        Returns:
            obs: (365,) proprio + tactile observations
            info: dict with current stage
        """
        if self._clearance != self.scene_config.clearance:
            self._tactile_sensor = None

        obs, info = super().reset(seed=seed, options=options)

        sensor = self._ensure_tactile()
        sensor.rng = self.np_random
        sensor.reset()

        obs = self._get_obs()

        return obs, info

    def _compute_step_extras(self) -> tuple[float, dict[str, float]]:
        """Extra rewards from tactile feedback — grasp stability bonus and slip penalty."""
        sensor = self._ensure_tactile()
        current, previous, change = sensor.get_readings(self.model, self.data)
        self._last_tactile = (current, previous, change)

        extra_reward = 0.0
        info: dict[str, float] = {}

        per_finger = current.reshape(5, 16)
        finger_forces = per_finger.sum(axis=1)
        thumb_force = finger_forces[4]
        other_force = float(np.mean(finger_forces[:4]))
        denom = thumb_force + other_force + 1e-6
        force_balance = 1.0 - abs(thumb_force - other_force) / denom
        all_active = float(np.min(finger_forces) > 0.1)
        grasp_stability = force_balance * all_active
        grasp_stability_reward = 0.1 * grasp_stability
        extra_reward += grasp_stability_reward

        nm = self.nm
        _, _, peg_linvel, _ = get_object_state(
            self.data, nm.peg_body_id, nm.peg_qpos_start, nm.peg_qvel_start
        )
        object_speed = float(np.linalg.norm(peg_linvel))
        tactile_drop = float(np.mean(change)) < -0.5
        slip_detected = object_speed > 0.05 and tactile_drop
        slip_penalty = -2.0 if slip_detected else 0.0
        extra_reward += slip_penalty

        info["reward/grasp_stability"] = grasp_stability_reward
        info["reward/slip_penalty"] = slip_penalty
        info["metrics/slip_detected"] = float(slip_detected)
        info["metrics/total_tactile_force"] = float(np.sum(current))
        info["metrics/grasp_stability"] = grasp_stability

        return extra_reward, info

    def _get_obs(self) -> np.ndarray:
        """Concatenate proprio obs with tactile readings (current + previous + change)."""
        base_obs = super()._get_obs()  # (125,)

        if hasattr(self, "_last_tactile") and self._last_tactile is not None:
            tactile_current, tactile_previous, tactile_change = self._last_tactile
            self._last_tactile = None  # type: ignore[assignment]
        else:
            sensor = self._ensure_tactile()
            tactile_current, tactile_previous, tactile_change = sensor.get_readings(
                self.model, self.data
            )

        return np.concatenate(
            [
                base_obs,  # [0:125]    125
                tactile_current,  # [125:205]   80
                tactile_previous,  # [205:285]   80
                tactile_change,  # [285:365]   80
            ]
        )
