from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from dexterous_hand.config import ReorientRewardConfig, ReorientSceneConfig
from dexterous_hand.envs.reorient_scene_builder import build_reorient_scene
from dexterous_hand.rewards.reorient_reward import ReorientRewardCalculator
from dexterous_hand.utils.mujoco_helpers import (
    get_cube_face_contacts,
    get_fingertip_positions,
    get_object_state,
    get_palm_position,
)
from dexterous_hand.utils.quaternion import (
    quat_conjugate,
    quat_multiply,
    random_quaternion_within_angle,
)


class ShadowHandReorientEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 25,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        scene_config: ReorientSceneConfig | None = None,
        reward_config: ReorientRewardConfig | None = None,
    ) -> None:
        """Sets up the in-hand reorientation environment.

        Args:
            render_mode: 'human' opens a window, 'rgb_array' for offscreen
            scene_config: scene and physics settings
            reward_config: reward weights and thresholds
        """
        super().__init__()

        self.scene_config = scene_config or ReorientSceneConfig()
        self.reward_config = reward_config or ReorientRewardConfig()
        self.render_mode = render_mode

        self.model, self.data, self.nm = build_reorient_scene(self.scene_config)

        # Obs: 24+24+3+4+3+3+4+4+15+5+6+20 = 115
        n_obs = 115
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nm.n_actuators,), dtype=np.float32
        )

        mujoco.mj_forward(self.model, self.data)
        init_cube_pos = self.data.xpos[self.nm.cube_body_id].copy()

        self.reward_calculator = ReorientRewardCalculator(
            config=self.reward_config,
            initial_cube_pos=init_cube_pos,
        )

        self._previous_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)
        self._target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._max_target_angle = 0.5236  # start at 30 degrees, curriculum ramps this up
        self._targets_reached = 0
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()
        self._palm_z: float = 0.0
        self.current_timestep: int = 0  # for curriculum via set_attr

        self._renderer: mujoco.Renderer | None = None
        if render_mode == "human":
            self._viewer: mujoco.viewer.Handle | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the env — puts the cube back in the palm and picks a new target.

        Args:
            seed: random seed
            options: extra options (unused)

        Returns:
            obs: observation vector (115,)
            info: has targets_reached count
        """
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        hand_qpos = self._init_qpos[self.nm.hand_qpos_start : self.nm.hand_qpos_end]
        noise = self.np_random.uniform(-0.01, 0.01, size=hand_qpos.shape)
        self.data.qpos[self.nm.hand_qpos_start : self.nm.hand_qpos_end] = hand_qpos + noise
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        palm_pos = get_palm_position(self.data, self.nm.palm_body_id)
        self._palm_z = float(palm_pos[2])

        s = self.nm.cube_qpos_start
        self.data.qpos[s : s + 3] = palm_pos + [0.0, 0.0, 0.03]
        init_quat = random_quaternion_within_angle(self.np_random, 0.1)
        self.data.qpos[s + 3 : s + 7] = init_quat

        mujoco.mj_forward(self.model, self.data)

        self._target_quat = random_quaternion_within_angle(self.np_random, self._max_target_angle)
        self._targets_reached = 0
        init_cube_pos = self.data.xpos[self.nm.cube_body_id].copy()
        self.reward_calculator.reset(initial_cube_pos=init_cube_pos)

        self._previous_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)

        obs = self._get_obs()
        info = {"targets_reached": 0}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step the env: apply action, run physics, compute reward.

        Args:
            action: (20,) normalized joint commands [-1, 1]

        Returns:
            obs: observation vector (115,)
            reward: scalar reward
            terminated: True if the cube was dropped
            truncated: always False (gym handles time limits)
            info: targets_reached + reward breakdown
        """
        action = np.clip(action, -1.0, 1.0)
        low = self.nm.ctrl_ranges[:, 0]
        high = self.nm.ctrl_ranges[:, 1]
        ctrl = low + (action + 1.0) / 2.0 * (high - low)
        self.data.ctrl[: self.nm.n_actuators] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=self.scene_config.frame_skip)

        fingertip_pos = get_fingertip_positions(self.data, self.nm.fingertip_site_ids)
        cube_pos, cube_quat, cube_linvel, cube_angvel = get_object_state(
            self.data,
            self.nm.cube_body_id,
            self.nm.cube_qpos_start,
            self.nm.cube_qvel_start,
        )

        dropped = cube_pos[2] < self._palm_z - self.reward_config.drop_height_offset

        reward, reward_info, target_reached = self.reward_calculator.compute(
            cube_quat=cube_quat,
            target_quat=self._target_quat,
            cube_pos=cube_pos,
            cube_linvel=cube_linvel,
            cube_angvel=cube_angvel,
            fingertip_positions=fingertip_pos,
            actions=action.astype(np.float64),
            previous_actions=self._previous_actions,
            dropped=dropped,
        )

        self._previous_actions = action.astype(np.float64).copy()

        if target_reached:
            self._targets_reached += 1
            self._target_quat = random_quaternion_within_angle(
                self.np_random, self._max_target_angle
            )
            self.reward_calculator.reset()

        terminated = dropped
        obs = self._get_obs()
        info = {
            "targets_reached": self._targets_reached,
            **reward_info,
        }

        return obs, float(reward), terminated, False, info

    def _get_obs(self) -> np.ndarray:
        """Build the flat observation from the current sim state.

        Returns:
            obs: (115,) joints + cube state + target quat + error quat + fingertips + actions
        """
        nm = self.nm
        joint_pos = self.data.qpos[nm.hand_qpos_start : nm.hand_qpos_end]  # [0:24]   24
        joint_vel = self.data.qvel[nm.hand_qvel_start : nm.hand_qvel_end]  # [24:48]  24
        cube_pos, cube_quat, cube_linvel, cube_angvel = get_object_state(
            self.data, nm.cube_body_id, nm.cube_qpos_start, nm.cube_qvel_start
        )
        fingertip_pos = get_fingertip_positions(self.data, nm.fingertip_site_ids)

        fingertip_cube_dists = np.linalg.norm(fingertip_pos - cube_pos, axis=1)

        err_quat = quat_multiply(quat_conjugate(cube_quat), self._target_quat)

        face_contacts = get_cube_face_contacts(
            self.model, self.data, nm.cube_geom_id, nm.fingertip_geom_ids
        )

        obs = np.concatenate(
            [
                joint_pos,  # [0:24]    24
                joint_vel,  # [24:48]   24
                cube_pos,  # [48:51]    3
                cube_quat,  # [51:55]    4
                cube_linvel,  # [55:58]    3
                cube_angvel,  # [58:61]    3
                self._target_quat,  # [61:65]    4
                err_quat,  # [65:69]    4
                fingertip_pos.flatten(),  # [69:84]   15
                fingertip_cube_dists,  # [84:89]    5
                face_contacts,  # [89:95]    6
                self._previous_actions,  # [95:115]  20
            ]
        )
        return obs

    def set_curriculum_stage(self, max_angle: float) -> None:
        """Update how far we rotate the target (called by the curriculum callback).

        Args:
            max_angle: max rotation in radians for sampling new targets
        """
        self._max_target_angle = max_angle

    def render(self) -> np.ndarray | None:  # type: ignore[override]
        """Render the scene. Returns RGB frame for rgb_array mode, None otherwise."""
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="track_cam")
            return np.asarray(self._renderer.render())
        elif self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        return None

    def close(self) -> None:
        """Clean up renderer and viewer."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if hasattr(self, "_viewer") and self._viewer is not None:
            self._viewer.close()
            self._viewer = None
