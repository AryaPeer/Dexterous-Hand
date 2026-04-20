from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from dexterous_hand.config import ReorientRewardConfig, ReorientSceneConfig
from dexterous_hand.envs.reorient_scene_builder import build_reorient_scene
from dexterous_hand.rewards.cpu.reorient_reward import ReorientRewardCalculator
from dexterous_hand.utils.cpu.mujoco_helpers import (
    get_cube_face_contacts,
    get_finger_contacts,
    get_fingertip_positions,
    get_object_state,
    get_palm_position,
)
from dexterous_hand.utils.cpu.quaternion import (
    quat_angular_distance,
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

        super().__init__()

        self.scene_config = scene_config or ReorientSceneConfig()
        self.reward_config = reward_config or ReorientRewardConfig()
        self.render_mode = render_mode

                                
        self.model, self.data, self.nm = build_reorient_scene(self.scene_config)

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
        self._smoothed_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)
        self._target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._max_target_angle = 0.5236                                    
        self._targets_reached = 0
        self._init_qpos = self.data.qpos.copy()
        self._palm_z: float = 0.0
        self._grasp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")

                   
        self._renderer: mujoco.Renderer | None = None
        if render_mode == "human":
            self._viewer: mujoco.viewer.Handle | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:

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
        cube_pos = self.data.site_xpos[self._grasp_site_id].copy()
        cube_pos += self.np_random.uniform(-0.004, 0.004, size=3)
        self.data.qpos[s : s + 3] = cube_pos
        init_quat = random_quaternion_within_angle(self.np_random, 0.1)
        self.data.qpos[s + 3 : s + 7] = init_quat

        mujoco.mj_forward(self.model, self.data)

                                            
        self._target_quat = self._sample_target_quat(cube_quat=init_quat)
        self._targets_reached = 0
        init_cube_pos = self.data.xpos[self.nm.cube_body_id].copy()
        self.reward_calculator.reset(initial_cube_pos=init_cube_pos)

        self._previous_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)
        self._smoothed_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)

        obs = self._get_obs()
        info = {"targets_reached": 0}

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:

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

                    
        fingertip_pos = get_fingertip_positions(self.data, self.nm.fingertip_site_ids)

        cube_pos, cube_quat, cube_linvel, cube_angvel = get_object_state(
            self.data,
            self.nm.cube_body_id,
            self.nm.cube_qpos_start,
            self.nm.cube_qvel_start,
        )

        num_contacts, _ = get_finger_contacts(
            self.model,
            self.data,
            self.nm.finger_geom_ids_per_finger,
            self.nm.cube_geom_id,
        )

        dropped = cube_pos[2] < self._palm_z - self.reward_config.drop_height_offset

        reward, reward_info, target_reached = self.reward_calculator.compute(
            cube_quat=cube_quat,
            target_quat=self._target_quat,
            cube_pos=cube_pos,
            cube_linvel=cube_linvel,
            finger_positions=fingertip_pos,
            num_fingers_in_contact=num_contacts,
            actions=action.astype(np.float64),
            previous_actions=self._previous_actions,
            dropped=dropped,
        )

        self._previous_actions = action.astype(np.float64).copy()

                                             
        if target_reached:
            self._targets_reached += 1
            self._target_quat = self._sample_target_quat(cube_quat=cube_quat)
            self.reward_calculator.reset()

        terminated = dropped

        obs = self._get_obs()
        info = {
            "targets_reached": self._targets_reached,
            **reward_info,
        }
        info["reward/total"] = float(reward)

        return obs, float(reward), terminated, False, info

    def _sample_target_quat(self, cube_quat: np.ndarray) -> np.ndarray:

        min_angle = min(self.scene_config.target_min_angle, 0.8 * self._max_target_angle)
        best_quat: np.ndarray | None = None
        best_dist = -1.0

        for _ in range(32):
            candidate = random_quaternion_within_angle(self.np_random, self._max_target_angle)
            dist = float(quat_angular_distance(cube_quat, candidate))
            if dist >= min_angle:
                return candidate
            if dist > best_dist:
                best_dist = dist
                best_quat = candidate

        if best_quat is not None:
            return best_quat
        return random_quaternion_within_angle(self.np_random, self._max_target_angle)

    def _get_obs(self) -> np.ndarray:

        nm = self.nm

        joint_pos = self.data.qpos[nm.hand_qpos_start : nm.hand_qpos_end]      
        joint_vel = self.data.qvel[nm.hand_qvel_start : nm.hand_qvel_end]      

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
                joint_pos,      
                joint_vel,      
                cube_pos,     
                cube_quat,     
                cube_linvel,     
                cube_angvel,     
                self._target_quat,     
                err_quat,     
                fingertip_pos.flatten(),      
                fingertip_cube_dists,     
                face_contacts,     
                self._previous_actions,      
            ]
        )

        return obs

    def set_curriculum_stage(self, max_angle: float) -> None:

        self._max_target_angle = max_angle

    def render(self) -> np.ndarray | None:  # type: ignore[override]

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

        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        if hasattr(self, "_viewer") and self._viewer is not None:
            self._viewer.close()
            self._viewer = None
