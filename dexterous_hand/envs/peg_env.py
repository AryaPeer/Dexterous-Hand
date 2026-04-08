from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from dexterous_hand.config import PegRewardConfig, PegSceneConfig
from dexterous_hand.envs.peg_scene_builder import build_peg_scene
from dexterous_hand.envs.scene_builder import TABLE_TASK_FLEXION_BIAS
from dexterous_hand.rewards.peg_reward import PegRewardCalculator
from dexterous_hand.utils.mujoco_helpers import (
    get_body_axis,
    get_contact_forces,
    get_contact_forces_on_body,
    get_finger_contacts,
    get_fingertip_positions,
    get_insertion_depth,
    get_object_state,
    get_palm_position,
    get_peg_hole_relative,
)


class ShadowHandPegEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 25,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        scene_config: PegSceneConfig | None = None,
        reward_config: PegRewardConfig | None = None,
    ) -> None:
        """Peg-in-hole assembly env.

        @param render_mode: 'human' for viewer, 'rgb_array' for offscreen
        @type render_mode: str | None
        @param scene_config: scene layout, hole clearance, physics
        @type scene_config: PegSceneConfig | None
        @param reward_config: reward weights and thresholds
        @type reward_config: PegRewardConfig | None
        """

        super().__init__()

        self.scene_config = scene_config or PegSceneConfig()
        self.reward_config = reward_config or PegRewardConfig()
        self.render_mode = render_mode

        # build scene and spaces
        self.model, self.data, self.nm = build_peg_scene(self.scene_config)

        n_obs = 131  # 26+26+3+4+3+3+3+4+3+3+15+5+3+1+6+1+22
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nm.n_actuators,), dtype=np.float32
        )

        # reward
        self.reward_calculator = PegRewardCalculator(
            config=self.reward_config,
            table_height=self.scene_config.table_height,
        )

        # state tracking
        self._previous_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)
        self._smoothed_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)
        self._stage = 0
        self._peg_pre_grasped = False
        self._clearance = self.scene_config.clearance
        self._init_qpos = self.data.qpos.copy()
        self._wall_geom_set: set[int] = set(self.nm.hole_wall_geom_ids)
        self._initial_peg_height = self.scene_config.table_height

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
        """Reset — peg on table (or in hand if pre-grasped).

        @param seed: random seed
        @type seed: int | None
        @param options: unused
        @type options: dict[str, Any] | None
        @return: (obs (131,), info with current stage)
        @rtype: tuple[np.ndarray, dict[str, Any]]
        """

        super().reset(seed=seed)

        # rebuild scene if clearance changed (curriculum)
        if self._clearance != self.scene_config.clearance:
            self.scene_config.clearance = self._clearance
            self.model, self.data, self.nm = build_peg_scene(self.scene_config)
            self._init_qpos = self.data.qpos.copy()
            self._wall_geom_set = set(self.nm.hole_wall_geom_ids)

        mujoco.mj_resetData(self.model, self.data)

        # randomize hand joints
        hand_qpos = self._init_qpos[self.nm.hand_qpos_start : self.nm.hand_qpos_end]
        noise = self.np_random.uniform(-0.01, 0.01, size=hand_qpos.shape)
        self.data.qpos[self.nm.hand_qpos_start : self.nm.hand_qpos_end] = hand_qpos + noise
        mujoco.mj_forward(self.model, self.data)

        # place peg — either in hand or on table
        s = self.nm.peg_qpos_start

        if self._peg_pre_grasped:
            palm_pos = get_palm_position(self.data, self.nm.palm_body_id)
            self.data.qpos[s : s + 3] = palm_pos + np.array([0.0, 0.0, -0.03], dtype=np.float64)
            self.data.qpos[s + 3 : s + 7] = [1.0, 0.0, 0.0, 0.0]
        else:
            hole_xy = np.array(self.scene_config.hole_offset[:2], dtype=np.float64)
            min_r = self.scene_config.spawn_min_radius
            peg_x, peg_y = 0.0, 0.0
            for _ in range(100):
                peg_x = float(self.np_random.uniform(-0.05, 0.05))
                peg_y = float(self.np_random.uniform(-0.05, 0.05))
                if np.linalg.norm([peg_x - hole_xy[0], peg_y - hole_xy[1]]) >= min_r:
                    break
            peg_z = self.scene_config.table_height + self.reward_config.peg_half_length + 0.001
            self.data.qpos[s : s + 3] = [peg_x, peg_y, peg_z]
            self.data.qpos[s + 3 : s + 7] = [1.0, 0.0, 0.0, 0.0]

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._previous_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)
        self._smoothed_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)
        self._stage = 0
        initial_peg_height = float(self.data.xpos[self.nm.peg_body_id][2])
        self._initial_peg_height = initial_peg_height
        self.reward_calculator.reset(initial_peg_height=initial_peg_height)

        obs = self._get_obs()
        info = {"stage": self._stage}

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply action, step physics, compute reward and task stage.

        @param action: (22,) normalized joint commands [-1, 1]
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

        nm = self.nm

        finger_pos = get_fingertip_positions(self.data, nm.fingertip_site_ids)

        peg_pos, peg_quat, peg_linvel, peg_angvel = get_object_state(
            self.data,
            nm.peg_body_id,
            nm.peg_qpos_start,
            nm.peg_qvel_start,
        )

        num_contacts, _ = get_finger_contacts(
            self.model,
            self.data,
            nm.finger_geom_ids_per_finger,
            nm.peg_geom_id,
        )

        palm_pos = get_palm_position(self.data, nm.palm_body_id)
        peg_axis = get_body_axis(self.data, nm.peg_body_id)
        hole_axis = get_body_axis(self.data, nm.hole_body_id)
        hole_pos = self.data.xpos[nm.hole_body_id].copy()

        # insertion + contact forces
        peg_half_length = self.reward_config.peg_half_length

        insertion_depth = get_insertion_depth(
            self.data, nm.peg_body_id, nm.hole_body_id, peg_half_length
        )

        per_wall_forces, contact_force_mag = get_contact_forces(
            self.model, self.data, nm.peg_geom_id, nm.hole_wall_geom_ids
        )

        peg_geom_set = {nm.peg_geom_id}
        wrench = get_contact_forces_on_body(
            self.model, self.data, peg_geom_set, self._wall_geom_set
        )
        reward_force_mag = float(np.linalg.norm(wrench[:3]))

        # determine task stage
        fingers_on_peg = num_contacts >= 3
        peg_lifted = peg_pos[2] > self._initial_peg_height + 0.02
        peg_near_hole = float(np.linalg.norm(peg_pos[:2] - hole_pos[:2])) < 0.03
        peg_aligned = abs(float(np.dot(peg_axis, hole_axis))) > 0.95

        if not fingers_on_peg:
            self._stage = 0
        elif not peg_lifted:
            self._stage = 1
        elif not (peg_near_hole and peg_aligned):
            self._stage = 2
        else:
            self._stage = 3

        # reward
        peg_height = float(peg_pos[2])

        reward, reward_info = self.reward_calculator.compute(
            stage=self._stage,
            finger_positions=finger_pos,
            peg_position=peg_pos,
            peg_axis=peg_axis,
            hole_position=hole_pos,
            hole_axis=hole_axis,
            insertion_depth=insertion_depth,
            contact_force_magnitude=reward_force_mag,
            num_fingers_in_contact=num_contacts,
            peg_height=peg_height,
            actions=action.astype(np.float64),
            previous_actions=self._previous_actions,
        )

        extra_reward, extra_info = self._compute_step_extras()
        reward += extra_reward
        self._previous_actions = action.astype(np.float64).copy()

        # cache obs data to avoid recomputing in _get_obs
        rel_pos, ang_error = get_peg_hole_relative(self.data, nm.peg_body_id, nm.hole_body_id)

        hole_quat = np.zeros(4)
        mujoco.mju_mat2Quat(hole_quat, self.data.xmat[nm.hole_body_id].flatten())

        fingertip_peg_dist = np.linalg.norm(finger_pos - peg_pos, axis=1)
        rel_peg_to_palm = peg_pos - palm_pos

        self._cached_obs = {
            "peg_pos": peg_pos,
            "peg_quat": peg_quat,
            "peg_linvel": peg_linvel,
            "peg_angvel": peg_angvel,
            "hole_pos": hole_pos,
            "hole_quat": hole_quat,
            "rel_pos": rel_pos,
            "ang_error": ang_error,
            "fingertip_pos": finger_pos,
            "fingertip_peg_dist": fingertip_peg_dist,
            "rel_peg_to_palm": rel_peg_to_palm,
            "insertion_depth": insertion_depth,
            "per_wall_forces": per_wall_forces,
            "contact_force_mag": contact_force_mag,
        }

        # termination
        terminated = False
        peg_length = peg_half_length * 2.0

        if (
            insertion_depth > 0.9 * peg_length
            and self.reward_calculator._insertion_hold_steps >= 10
        ):
            terminated = True

        if peg_pos[2] < self.scene_config.table_height - 0.1:
            terminated = True

        obs = self._get_obs()
        info = {
            "stage": self._stage,
            **extra_info,
            **reward_info,
        }
        info["reward/total"] = float(reward)

        return obs, float(reward), terminated, False, info

    def _compute_step_extras(self) -> tuple[float, dict[str, float]]:
        """Hook for subclasses to add extra reward terms (e.g. tactile)."""

        return 0.0, {}

    def _get_obs(self) -> np.ndarray:
        """Flat obs (131,) — joints, peg/hole state, contacts, prev actions."""

        nm = self.nm

        joint_pos = self.data.qpos[nm.hand_qpos_start : nm.hand_qpos_end]  # 26
        joint_vel = self.data.qvel[nm.hand_qvel_start : nm.hand_qvel_end]  # 26

        # use cached values from step() if available, otherwise recompute
        if hasattr(self, "_cached_obs") and self._cached_obs is not None:
            c = self._cached_obs
            peg_pos = c["peg_pos"]
            peg_quat = c["peg_quat"]
            peg_linvel = c["peg_linvel"]
            peg_angvel = c["peg_angvel"]
            hole_pos = c["hole_pos"]
            hole_quat = c["hole_quat"]
            rel_pos = c["rel_pos"]
            ang_error = c["ang_error"]
            fingertip_pos = c["fingertip_pos"]
            fingertip_peg_dist = c["fingertip_peg_dist"]
            rel_peg_to_palm = c["rel_peg_to_palm"]
            insertion_depth = c["insertion_depth"]
            per_wall_forces = c["per_wall_forces"]
            contact_force_mag = c["contact_force_mag"]
            self._cached_obs = None  # type: ignore[assignment]

        else:
            peg_pos, peg_quat, peg_linvel, peg_angvel = get_object_state(
                self.data, nm.peg_body_id, nm.peg_qpos_start, nm.peg_qvel_start
            )

            hole_pos = self.data.xpos[nm.hole_body_id].copy()
            hole_quat = np.zeros(4)
            mujoco.mju_mat2Quat(hole_quat, self.data.xmat[nm.hole_body_id].flatten())

            rel_pos, ang_error = get_peg_hole_relative(self.data, nm.peg_body_id, nm.hole_body_id)

            fingertip_pos = get_fingertip_positions(self.data, nm.fingertip_site_ids)
            fingertip_peg_dist = np.linalg.norm(fingertip_pos - peg_pos, axis=1)

            palm_pos = get_palm_position(self.data, nm.palm_body_id)
            rel_peg_to_palm = peg_pos - palm_pos

            insertion_depth = get_insertion_depth(
                self.data,
                nm.peg_body_id,
                nm.hole_body_id,
                self.reward_config.peg_half_length,
            )

            per_wall_forces, contact_force_mag = get_contact_forces(
                self.model, self.data, nm.peg_geom_id, nm.hole_wall_geom_ids
            )

        contact_forces = np.append(per_wall_forces, contact_force_mag)

        obs = np.concatenate(
            [
                joint_pos,                # 24
                joint_vel,                # 24
                peg_pos,                  # 3
                peg_quat,                 # 4
                peg_linvel,               # 3
                peg_angvel,               # 3
                hole_pos,                 # 3
                hole_quat,                # 4
                rel_pos,                  # 3
                ang_error,                # 3
                fingertip_pos.flatten(),  # 15
                fingertip_peg_dist,       # 5
                rel_peg_to_palm,          # 3
                [insertion_depth],        # 1
                contact_forces,           # 6
                [float(self._stage)],     # 1
                self._previous_actions,   # 20
            ]
        )

        return obs

    def set_clearance(self, clearance: float) -> None:
        """Set hole clearance (rebuilds scene on next reset)."""

        self._clearance = clearance

    def set_curriculum_params(self, clearance: float, pre_grasped: bool) -> None:
        """Update curriculum settings (called by curriculum callback)."""

        self._clearance = clearance
        self._peg_pre_grasped = pre_grasped

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
