from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from dexterous_hand.config import RewardConfig, SceneConfig
from dexterous_hand.envs.scene_builder import (
    OBJECT_TYPES,
    TABLE_TASK_FLEXION_BIAS,
    build_scene,
    get_object_half_height,
)
from dexterous_hand.rewards.grasp_reward import GraspRewardCalculator
from dexterous_hand.utils.mujoco_helpers import (
    get_finger_contacts,
    get_finger_positions,
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
        """Shadow Hand grasp env.

        @param render_mode: 'human' for viewer, 'rgb_array' for offscreen
        @type render_mode: str | None
        @param scene_config: physics and scene layout
        @type scene_config: SceneConfig | None
        @param reward_config: reward weights and thresholds
        @type reward_config: RewardConfig | None
        @param object_types: which objects to train on (defaults to all)
        @type object_types: list[str] | None
        """

        super().__init__()

        self.scene_config = scene_config or SceneConfig()
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode
        self._object_type_names = object_types or list(OBJECT_TYPES.keys())

        # build scene and spaces
        self.model, self.data, self.nm = build_scene(self.scene_config)

        n_obs = 105  # 26+26+3+4+3+3+3+15+22
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nm.n_actuators,), dtype=np.float32
        )

        # reward
        self.reward_calculator = GraspRewardCalculator(
            config=self.reward_config,
            table_height=self.scene_config.table_height,
        )

        # state tracking
        self._previous_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)
        self._smoothed_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)
        self._current_object_type: str = self._object_type_names[0]
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

        for jname, bias in TABLE_TASK_FLEXION_BIAS.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                adr = self.model.jnt_qposadr[jid]
                self._init_qpos[adr] = float(np.clip(
                    bias, self.model.jnt_range[jid][0], self.model.jnt_range[jid][1],
                ))

        # rendering
        self._renderer: mujoco.Renderer | None = None
        if render_mode == "human":
            self._viewer: mujoco.viewer.Handle | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset with a random object on the table.

        @param seed: random seed
        @type seed: int | None
        @param options: unused
        @type options: dict[str, Any] | None
        @return: (obs (105,), info with sampled object_type)
        @rtype: tuple[np.ndarray, dict[str, Any]]
        """

        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # pick a random object and set its geom
        self._current_object_type = self.np_random.choice(self._object_type_names)
        geom_type, geom_size = OBJECT_TYPES[self._current_object_type]
        self.model.geom_type[self.nm.object_geom_id] = geom_type
        self.model.geom_size[self.nm.object_geom_id] = geom_size + [0.0] * (3 - len(geom_size))

        # place object on table with slight random XY offset
        half_h = get_object_half_height(geom_type, geom_size)
        obj_x = self.np_random.uniform(-0.05, 0.05)
        obj_y = self.np_random.uniform(-0.05, 0.05)
        obj_z = self.scene_config.table_height + half_h + 0.001

        s = self.nm.obj_qpos_start
        self.data.qpos[s : s + 3] = [obj_x, obj_y, obj_z]
        self.data.qpos[s + 3 : s + 7] = [1.0, 0.0, 0.0, 0.0]

        # randomize hand joint positions slightly
        hand_qpos = self._init_qpos[self.nm.hand_qpos_start : self.nm.hand_qpos_end]
        noise = self.np_random.uniform(-0.01, 0.01, size=hand_qpos.shape)
        self.data.qpos[self.nm.hand_qpos_start : self.nm.hand_qpos_end] = hand_qpos + noise

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._previous_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)
        self._smoothed_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)
        self.reward_calculator.reset(initial_object_height=obj_z)

        obs = self._get_obs()
        info = {"object_type": self._current_object_type}

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply action, step physics, compute reward.

        @param action: (22,) normalized joint commands in [-1, 1]
        @type action: np.ndarray
        @return: standard gym tuple (obs, reward, terminated, truncated, info)
        @rtype: tuple[np.ndarray, float, bool, bool, dict[str, Any]]
        """

        action = np.clip(action, -1.0, 1.0)
        alpha = float(np.clip(self.scene_config.action_smoothing_alpha, 0.0, 1.0))
        if alpha > 0.0:
            action = (1.0 - alpha) * self._smoothed_actions + alpha * action
        self._smoothed_actions = action.astype(np.float64).copy()

        low = self.nm.ctrl_ranges[:, 0]
        high = self.nm.ctrl_ranges[:, 1]
        ctrl = low + (action + 1.0) / 2.0 * (high - low)
        self.data.ctrl[: self.nm.n_actuators] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=self.scene_config.frame_skip)

        # read state
        finger_pos = get_finger_positions(self.data, self.nm.finger_geom_ids_per_finger)

        obj_pos, obj_quat, obj_linvel, obj_angvel = get_object_state(
            self.data,
            self.nm.object_body_id,
            self.nm.obj_qpos_start,
            self.nm.obj_qvel_start,
        )

        num_contacts, _ = get_finger_contacts(
            self.model,
            self.data,
            self.nm.finger_geom_ids_per_finger,
            self.nm.object_geom_id,
        )

        reward, reward_info = self.reward_calculator.compute(
            finger_positions=finger_pos,
            object_position=obj_pos,
            object_linear_velocity=obj_linvel,
            num_fingers_in_contact=num_contacts,
            actions=action.astype(np.float64),
            previous_actions=self._previous_actions,
        )

        # termination checks
        self._previous_actions = action.astype(np.float64).copy()
        terminated = False

        if obj_pos[2] < self.scene_config.table_height - 0.05:  # fell off
            terminated = True

        if np.linalg.norm(obj_pos) > 1.5:  # object launched
            terminated = True

        obs = self._get_obs()
        info = {
            "object_type": self._current_object_type,
            **reward_info,
        }
        info["reward/total"] = float(reward)

        return obs, float(reward), terminated, False, info

    def _get_obs(self) -> np.ndarray:
        """Flat obs vector (105,) — joints, object state, fingertips, prev actions."""

        nm = self.nm

        joint_pos = self.data.qpos[nm.hand_qpos_start : nm.hand_qpos_end]
        joint_vel = self.data.qvel[nm.hand_qvel_start : nm.hand_qvel_end]

        obj_pos, obj_quat, obj_linvel, obj_angvel = get_object_state(
            self.data, nm.object_body_id, nm.obj_qpos_start, nm.obj_qvel_start
        )

        palm_pos = get_palm_position(self.data, nm.palm_body_id)
        rel_pos = obj_pos - palm_pos
        fingertip_pos = get_finger_positions(self.data, nm.finger_geom_ids_per_finger).flatten()

        obs = np.concatenate(
            [
                joint_pos,       # 24
                joint_vel,       # 24
                obj_pos,         # 3
                obj_quat,        # 4
                obj_linvel,      # 3
                obj_angvel,      # 3
                rel_pos,         # 3
                fingertip_pos,   # 15
                self._previous_actions,  # 20
            ]
        )

        return obs

    def render(self) -> np.ndarray | None:  # type: ignore[override]
        """RGB frame if rgb_array mode, otherwise syncs viewer."""

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
