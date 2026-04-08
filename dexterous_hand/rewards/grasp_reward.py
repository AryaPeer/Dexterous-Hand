import numpy as np

from dexterous_hand.config import RewardConfig


class GraspRewardCalculator:
    def __init__(self, config: RewardConfig, table_height: float) -> None:
        """Grasp reward calculator.

        @param config: reward weights and thresholds
        @type config: RewardConfig
        @param table_height: table surface height for lift calculations
        @type table_height: float
        """

        self.weights = config.weights
        self.lift_target = config.lift_target
        self.hold_velocity_threshold = config.hold_velocity_threshold
        self.drop_penalty_value = config.drop_penalty
        self.no_contact_idle_penalty = config.no_contact_idle_penalty
        self.table_height = table_height
        self._was_lifted = False
        self._initial_height_above_table = 0.0

    def reset(self, initial_object_height: float | None = None) -> None:
        """Reset for a new episode."""

        self._was_lifted = False
        if initial_object_height is None:
            self._initial_height_above_table = 0.0
        else:
            self._initial_height_above_table = max(
                float(initial_object_height - self.table_height),
                0.0,
            )

    def compute(
        self,
        finger_positions: np.ndarray,
        object_position: np.ndarray,
        object_linear_velocity: np.ndarray,
        num_fingers_in_contact: int,
        contact_finger_indices: set[int],
        actions: np.ndarray,
        previous_actions: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Total grasp reward: reaching + grasping + lifting + holding - penalties.

        @param finger_positions: (5, 3) per-finger representative positions
        @type finger_positions: np.ndarray
        @param object_position: (3,) object center
        @type object_position: np.ndarray
        @param object_linear_velocity: (3,) object velocity
        @type object_linear_velocity: np.ndarray
        @param num_fingers_in_contact: fingers touching the object
        @type num_fingers_in_contact: int
        @param contact_finger_indices: set of finger indices currently in contact
        @type contact_finger_indices: set[int]
        @param actions: (20,) current actions
        @type actions: np.ndarray
        @param previous_actions: (20,) last step's actions
        @type previous_actions: np.ndarray
        @return: (total, info) weighted reward sum and per-component breakdown
        @rtype: tuple[float, dict[str, float]]
        """

        info: dict[str, float] = {}

        obj_height = object_position[2]
        height_above_table = obj_height - self.table_height
        lift_height = max(height_above_table - self._initial_height_above_table, 0.0)

        dists = np.linalg.norm(finger_positions - object_position, axis=1)
        contact_factor = min(num_fingers_in_contact / 2.0, 1.0)
        reaching = float(np.exp(-10.0 * np.mean(dists))) * (1.0 - 0.5 * contact_factor)
        info["reward/reaching"] = reaching

        side_contacts = 0
        for idx in contact_finger_indices:
            if finger_positions[idx, 2] <= object_position[2] + 0.015:
                side_contacts += 1
        side_ratio = side_contacts / max(num_fingers_in_contact, 1)
        info["reward/grasp_quality"] = side_ratio

        grasping = (num_fingers_in_contact / 5.0) * (0.3 + 0.7 * side_ratio)
        info["reward/grasping"] = grasping

        lift_hold_gate = 1.0 if num_fingers_in_contact >= 2 else 0.0
        lifting = float(np.clip(lift_height, 0.0, self.lift_target) / self.lift_target) * lift_hold_gate
        info["reward/lifting"] = lifting

        obj_speed = float(np.linalg.norm(object_linear_velocity))
        is_above = lift_height >= self.lift_target
        is_stable = obj_speed < self.hold_velocity_threshold
        holding = (1.0 if (is_above and is_stable) else 0.0) * lift_hold_gate
        info["reward/holding"] = holding

        if is_above:
            self._was_lifted = True

        dropped = self._was_lifted and lift_height < 0.01
        drop = self.drop_penalty_value if dropped else 0.0
        info["reward/drop"] = drop

        idle_raw = self.no_contact_idle_penalty if (num_fingers_in_contact == 0 and lifting < 0.01) else 0.0
        idle_penalty = self.weights.idle * idle_raw
        info["reward/idle_penalty"] = idle_penalty

        action_pen = -0.01 * float(np.sum(actions**2))
        info["reward/action_penalty"] = action_pen

        action_rate_pen = -0.005 * float(np.sum((actions - previous_actions) ** 2))
        info["reward/action_rate_penalty"] = action_rate_pen

        total = (
            self.weights.reaching * reaching
            + self.weights.grasping * grasping
            + self.weights.lifting * lifting
            + self.weights.holding * holding
            + self.weights.drop * drop
            + self.weights.action * action_pen
            + self.weights.action_rate * action_rate_pen
            + idle_penalty
        )

        info["reward/total"] = total
        info["metrics/num_finger_contacts"] = float(num_fingers_in_contact)
        info["metrics/object_height"] = obj_height
        info["metrics/object_speed"] = obj_speed
        info["metrics/mean_fingertip_dist"] = float(np.mean(dists))

        return total, info
