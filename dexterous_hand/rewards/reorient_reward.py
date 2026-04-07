import numpy as np

from dexterous_hand.config import ReorientRewardConfig
from dexterous_hand.utils.quaternion import quat_angular_distance


class ReorientRewardCalculator:
    def __init__(self, config: ReorientRewardConfig, initial_cube_pos: np.ndarray) -> None:
        """Reorientation reward calculator.

        @param config: reward weights and thresholds
        @type config: ReorientRewardConfig
        @param initial_cube_pos: (3,) starting cube position (for displacement penalty)
        @type initial_cube_pos: np.ndarray
        """

        self.weights = config.weights
        self.success_threshold = config.success_threshold
        self.success_hold_steps = config.success_hold_steps
        self.drop_penalty_value = config.drop_penalty
        self.contact_bonus_value = config.contact_bonus
        self.no_contact_penalty_value = config.no_contact_penalty
        self.min_contacts_for_rotation = config.min_contacts_for_rotation
        self._initial_cube_pos = initial_cube_pos.copy()
        self._success_steps = 0

    def reset(self, initial_cube_pos: np.ndarray | None = None) -> None:
        """Reset for a new episode or new target."""

        self._success_steps = 0
        if initial_cube_pos is not None:
            self._initial_cube_pos = initial_cube_pos.copy()

    def compute(
        self,
        cube_quat: np.ndarray,
        target_quat: np.ndarray,
        cube_pos: np.ndarray,
        cube_linvel: np.ndarray,
        cube_angvel: np.ndarray,
        finger_positions: np.ndarray,
        num_fingers_in_contact: int,
        actions: np.ndarray,
        previous_actions: np.ndarray,
        dropped: bool,
    ) -> tuple[float, dict[str, float], bool]:
        """Reorientation reward: orientation tracking + contact-gated penalties.

        @param cube_quat: (4,) cube orientation [w,x,y,z]
        @type cube_quat: np.ndarray
        @param target_quat: (4,) target orientation
        @type target_quat: np.ndarray
        @param cube_pos: (3,) cube position
        @type cube_pos: np.ndarray
        @param cube_linvel: (3,) cube linear velocity
        @type cube_linvel: np.ndarray
        @param cube_angvel: (3,) cube angular velocity
        @type cube_angvel: np.ndarray
        @param finger_positions: (5, 3) per-finger representative positions
        @type finger_positions: np.ndarray
        @param num_fingers_in_contact: fingers touching the cube
        @type num_fingers_in_contact: int
        @param actions: (20,) current actions
        @type actions: np.ndarray
        @param previous_actions: (20,) last step's actions
        @type previous_actions: np.ndarray
        @param dropped: True if cube fell
        @type dropped: bool
        @return: (total, info, target_reached) reward, breakdown, and success flag
        @rtype: tuple[float, dict[str, float], bool]
        """

        info: dict[str, float] = {}

        min_contacts = self.min_contacts_for_rotation
        contact_factor = min(num_fingers_in_contact / max(float(min_contacts), 1.0), 1.0)

        ang_dist = quat_angular_distance(cube_quat, target_quat)
        orientation_tracking = float(np.exp(-5.0 * ang_dist)) * contact_factor
        info["reward/orientation_tracking"] = orientation_tracking

        at_target = ang_dist < self.success_threshold

        if at_target:
            self._success_steps += 1
        else:
            self._success_steps = 0

        orientation_success = (5.0 if at_target else 0.0) if num_fingers_in_contact >= min_contacts else 0.0
        info["reward/orientation_success"] = orientation_success
        target_reached = self._success_steps >= self.success_hold_steps

        cube_drop = self.drop_penalty_value if dropped else 0.0
        info["reward/cube_drop"] = cube_drop

        velocity_penalty = -0.1 * float(
            np.linalg.norm(cube_linvel) ** 2 + 0.5 * np.linalg.norm(cube_angvel) ** 2
        )
        info["reward/velocity_penalty"] = velocity_penalty

        dists = np.linalg.norm(finger_positions - cube_pos, axis=1)
        fingertip_distance = float(np.exp(-5.0 * np.mean(dists))) * (0.5 + 0.5 * contact_factor)
        info["reward/fingertip_distance"] = fingertip_distance

        pos_error_sq = float(np.linalg.norm(cube_pos - self._initial_cube_pos) ** 2)
        position_penalty = -5.0 * pos_error_sq
        info["reward/position_penalty"] = position_penalty

        action_penalty = -0.005 * float(np.sum(actions**2))
        info["reward/action_penalty"] = action_penalty

        action_rate_penalty = -0.002 * float(np.sum((actions - previous_actions) ** 2))
        info["reward/action_rate_penalty"] = action_rate_penalty

        finger_contact_bonus = self.contact_bonus_value * min(num_fingers_in_contact / 3.0, 1.0)
        info["reward/finger_contact_bonus"] = finger_contact_bonus

        no_contact_penalty = self.no_contact_penalty_value if num_fingers_in_contact == 0 else 0.0
        info["reward/no_contact_penalty"] = no_contact_penalty

        total = (
            self.weights.orientation_tracking * orientation_tracking
            + self.weights.orientation_success * orientation_success
            + self.weights.cube_drop * cube_drop
            + self.weights.velocity_penalty * velocity_penalty
            + self.weights.fingertip_distance * fingertip_distance
            + self.weights.position_penalty * position_penalty
            + self.weights.action_penalty * action_penalty
            + self.weights.action_rate_penalty * action_rate_penalty
            + finger_contact_bonus
            + no_contact_penalty
        )

        info["reward/total"] = total
        info["metrics/angular_distance"] = ang_dist
        info["metrics/num_finger_contacts"] = float(num_fingers_in_contact)
        info["metrics/mean_fingertip_dist"] = float(np.mean(dists))
        info["metrics/cube_displacement"] = float(np.sqrt(pos_error_sq))
        info["metrics/success_steps"] = float(self._success_steps)

        return total, info, target_reached
