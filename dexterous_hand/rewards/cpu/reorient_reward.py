
import numpy as np

from dexterous_hand.config import ReorientRewardConfig
from dexterous_hand.utils.cpu.quaternion import quat_angular_distance

class ReorientRewardCalculator:

    def __init__(self, config: ReorientRewardConfig, initial_cube_pos: np.ndarray) -> None:

        self.weights = config.weights
        self.success_threshold = config.success_threshold
        self.success_hold_steps = config.success_hold_steps
        self.drop_penalty_value = config.drop_penalty
        self.contact_bonus_value = config.contact_bonus
        self.no_contact_penalty_value = config.no_contact_penalty
        self.min_contacts_for_rotation = config.min_contacts_for_rotation
        self.angular_progress_clip = config.angular_progress_clip
        self.tracking_k = config.tracking_k
        self.orientation_contact_alpha = config.orientation_contact_alpha
        del initial_cube_pos
        self._success_steps = 0
        self._prev_ang_dist: float | None = None

    def reset(self, initial_cube_pos: np.ndarray | None = None) -> None:

        self._success_steps = 0
        self._prev_ang_dist = None
        del initial_cube_pos

    def compute(
        self,
        cube_quat: np.ndarray,
        target_quat: np.ndarray,
        cube_pos: np.ndarray,
        cube_linvel: np.ndarray,
        finger_positions: np.ndarray,
        num_fingers_in_contact: int,
        actions: np.ndarray,
        previous_actions: np.ndarray,
        dropped: bool,
    ) -> tuple[float, dict[str, float], bool]:

        info: dict[str, float] = {}

        min_contacts = self.min_contacts_for_rotation

        ang_dist = quat_angular_distance(cube_quat, target_quat)

        if self._prev_ang_dist is None:
            angular_progress = 0.0
        else:
            angular_progress = float(self._prev_ang_dist - ang_dist)
        clip = self.angular_progress_clip
        angular_progress = float(np.clip(angular_progress, -clip, clip))
        self._prev_ang_dist = float(ang_dist)
        info["reward/angular_progress"] = angular_progress

        soft_contact_scale = min(num_fingers_in_contact / float(min_contacts), 1.0)
        alpha = self.orientation_contact_alpha
        orientation_gate = alpha + (1.0 - alpha) * soft_contact_scale
        orientation = float(np.exp(-self.tracking_k * ang_dist)) * orientation_gate
        info["reward/orientation"] = orientation

        at_target = ang_dist < self.success_threshold
        if at_target and num_fingers_in_contact >= min_contacts:
            self._success_steps += 1
        else:
            self._success_steps = 0
        target_reached = self._success_steps >= self.success_hold_steps

        cube_drop = self.drop_penalty_value if dropped else 0.0
        info["reward/cube_drop"] = cube_drop

        action_penalty = -0.0002 * float(np.sum(actions**2))
        info["reward/action_penalty"] = action_penalty
        del previous_actions

        contact_raw = self.contact_bonus_value * min(num_fingers_in_contact / 3.0, 1.0)
        finger_contact_bonus = self.weights.contact_bonus * contact_raw
        info["reward/finger_contact_bonus"] = finger_contact_bonus

        no_contact_ramp = float(np.exp(-2.0 * num_fingers_in_contact))
        no_contact_raw = self.no_contact_penalty_value * no_contact_ramp
        no_contact_penalty = self.weights.no_contact * no_contact_raw
        info["reward/no_contact_penalty"] = no_contact_penalty

        total = (
            self.weights.angular_progress * angular_progress
            + self.weights.orientation * orientation
            + self.weights.cube_drop * cube_drop
            + self.weights.action_penalty * action_penalty
            + finger_contact_bonus
            + no_contact_penalty
        )

        info["reward/total"] = total
        info["metrics/angular_distance"] = ang_dist
        info["metrics/num_finger_contacts"] = float(num_fingers_in_contact)
        info["metrics/success_steps"] = float(self._success_steps)
        del cube_pos, cube_linvel, finger_positions

        return total, info, target_reached
