from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from dexterous_hand.config import RewardConfig, SceneConfig
from dexterous_hand.envs.scene_builder import (
    OBJECT_TYPES,
    build_scene,
    get_object_half_height,
)
from dexterous_hand.rewards.grasp_reward import GraspRewardCalculator
from dexterous_hand.utils.mujoco_helpers import (
    get_fingertip_contacts,
    get_fingertip_positions,
    get_object_state,
    get_palm_position,
)


class ShadowHandGraspEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 25,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        scene_config: SceneConfig | None = None,
        reward_config: RewardConfig | None = None,
        object_types: list[str] | None = None,
    ) -> None:
        """Sets up the grasp environment with the Shadow Hand.

        Args:
            render_mode: 'human' opens a viewer window, 'rgb_array' for offscreen rendering
            scene_config: physics and scene layout settings
            reward_config: reward shaping weights and thresholds
            object_types: which objects to train on (defaults to all of them)
        """
        super().__init__()

        self.scene_config = scene_config or SceneConfig()
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode
        self._object_type_names = object_types or list(OBJECT_TYPES.keys())

        self.model, self.data, self.nm = build_scene(self.scene_config)

        n_obs = 99  # 24+24+3+4+3+3+3+15+20
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nm.n_actuators,), dtype=np.float32
        )

        self.reward_calculator = GraspRewardCalculator(
            config=self.reward_config,
            table_height=self.scene_config.table_height,
        )

        self._previous_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)
        self._current_object_type: str = self._object_type_names[0]
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

        self._renderer: mujoco.Renderer | None = None
        if render_mode == "human":
            self._viewer: mujoco.viewer.Handle | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to a fresh episode with a random object on the table.

        Args:
            seed: random seed for reproducibility
            options: extra reset options (unused for now)

        Returns:
            obs: flat observation vector (99,)
            info: has the object_type we sampled
        """
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self._current_object_type = self.np_random.choice(self._object_type_names)
        geom_type, geom_size = OBJECT_TYPES[self._current_object_type]
        self.model.geom_type[self.nm.object_geom_id] = geom_type
        self.model.geom_size[self.nm.object_geom_id] = geom_size + [0.0] * (3 - len(geom_size))

        half_h = get_object_half_height(geom_type, geom_size)
        obj_x = self.np_random.uniform(-0.05, 0.05)
        obj_y = self.np_random.uniform(-0.05, 0.05)
        obj_z = self.scene_config.table_height + half_h + 0.001
        s = self.nm.obj_qpos_start
        self.data.qpos[s : s + 3] = [obj_x, obj_y, obj_z]
        self.data.qpos[s + 3 : s + 7] = [1.0, 0.0, 0.0, 0.0]

        hand_qpos = self._init_qpos[self.nm.hand_qpos_start : self.nm.hand_qpos_end]
        noise = self.np_random.uniform(-0.01, 0.01, size=hand_qpos.shape)
        self.data.qpos[self.nm.hand_qpos_start : self.nm.hand_qpos_end] = hand_qpos + noise
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._previous_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)
        self.reward_calculator.reset()

        obs = self._get_obs()
        info = {"object_type": self._current_object_type}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take one step: apply the action, step physics, get reward.

        Args:
            action: (20,) normalized joint commands in [-1, 1]

        Returns:
            obs: observation vector (99,)
            reward: scalar reward
            terminated: True if the object fell off the table or flew away
            truncated: always False (gym wrapper handles time limits)
            info: reward breakdown + object type
        """
        action = np.clip(action, -1.0, 1.0)
        low = self.nm.ctrl_ranges[:, 0]
        high = self.nm.ctrl_ranges[:, 1]
        ctrl = low + (action + 1.0) / 2.0 * (high - low)
        self.data.ctrl[: self.nm.n_actuators] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=self.scene_config.frame_skip)

        fingertip_pos = get_fingertip_positions(self.data, self.nm.fingertip_site_ids)
        obj_pos, obj_quat, obj_linvel, obj_angvel = get_object_state(
            self.data,
            self.nm.object_body_id,
            self.nm.obj_qpos_start,
            self.nm.obj_qvel_start,
        )
        num_contacts, _ = get_fingertip_contacts(
            self.model, self.data, self.nm.fingertip_geom_ids, self.nm.object_geom_id
        )

        reward, reward_info = self.reward_calculator.compute(
            fingertip_positions=fingertip_pos,
            object_position=obj_pos,
            object_linear_velocity=obj_linvel,
            num_fingers_in_contact=num_contacts,
            actions=action.astype(np.float64),
            previous_actions=self._previous_actions,
        )

        self._previous_actions = action.astype(np.float64).copy()
        terminated = False
        if obj_pos[2] < self.scene_config.table_height - 0.05:  # fell off
            terminated = True
        if np.linalg.norm(obj_pos) > 1.5:  # something went wrong, object launched
            terminated = True

        obs = self._get_obs()
        info = {
            "object_type": self._current_object_type,
            **reward_info,
        }
        return obs, float(reward), terminated, False, info

    def _get_obs(self) -> np.ndarray:
        """Build the flat obs vector from current sim state.

        Returns:
            obs: (99,) everything concatenated — joints, object, fingertips, prev actions
        """
        nm = self.nm
        joint_pos = self.data.qpos[nm.hand_qpos_start : nm.hand_qpos_end]
        joint_vel = self.data.qvel[nm.hand_qvel_start : nm.hand_qvel_end]
        obj_pos, obj_quat, obj_linvel, obj_angvel = get_object_state(
            self.data, nm.object_body_id, nm.obj_qpos_start, nm.obj_qvel_start
        )
        palm_pos = get_palm_position(self.data, nm.palm_body_id)
        rel_pos = obj_pos - palm_pos
        fingertip_pos = get_fingertip_positions(self.data, nm.fingertip_site_ids).flatten()

        obs = np.concatenate(
            [
                joint_pos,  # 24
                joint_vel,  # 24
                obj_pos,  # 3
                obj_quat,  # 4
                obj_linvel,  # 3
                obj_angvel,  # 3
                rel_pos,  # 3
                fingertip_pos,  # 15
                self._previous_actions,  # 20
            ]
        )
        return obs

    def render(self) -> np.ndarray | None:  # type: ignore[override]
        """Render the scene. Returns an RGB frame if using rgb_array mode, otherwise None."""
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
        """Clean up the renderer and viewer."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if hasattr(self, "_viewer") and self._viewer is not None:
            self._viewer.close()
            self._viewer = None
