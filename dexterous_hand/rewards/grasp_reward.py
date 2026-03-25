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
        self.table_height = table_height
        self._was_lifted = False

    def reset(self) -> None:
        """Reset for a new episode."""

        self._was_lifted = False

    def compute(
        self,
        fingertip_positions: np.ndarray,  # (5, 3)
        object_position: np.ndarray,  # (3,)
        object_linear_velocity: np.ndarray,  # (3,)
        num_fingers_in_contact: int,
        actions: np.ndarray,  # (20,)
        previous_actions: np.ndarray,  # (20,)
    ) -> tuple[float, dict[str, float]]:
        """Total grasp reward: reaching + grasping + lifting + holding - penalties.

        @param fingertip_positions: (5, 3) fingertip positions
        @type fingertip_positions: np.ndarray
        @param object_position: (3,) object center
        @type object_position: np.ndarray
        @param object_linear_velocity: (3,) object velocity
        @type object_linear_velocity: np.ndarray
        @param num_fingers_in_contact: fingers touching the object
        @type num_fingers_in_contact: int
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

        # reaching — exponential reward for fingertip proximity
        dists = np.linalg.norm(fingertip_positions - object_position, axis=1)
        reaching = float(np.exp(-10.0 * np.mean(dists)))
        info["reward/reaching"] = reaching

        # grasping — fraction of fingers in contact
        grasping = num_fingers_in_contact / 5.0
        info["reward/grasping"] = grasping

        # lifting — clipped progress toward lift target
        lifting = float(np.clip(height_above_table, 0.0, self.lift_target) / self.lift_target)
        info["reward/lifting"] = lifting

        # holding — bonus for stable hold above target
        obj_speed = float(np.linalg.norm(object_linear_velocity))
        is_above = height_above_table >= self.lift_target
        is_stable = obj_speed < self.hold_velocity_threshold
        holding = 1.0 if (is_above and is_stable) else 0.0
        info["reward/holding"] = holding

        # drop detection
        if is_above:
            self._was_lifted = True

        dropped = self._was_lifted and height_above_table < 0.01
        drop = self.drop_penalty_value if dropped else 0.0
        info["reward/drop"] = drop

        # action penalties
        action_pen = -0.01 * float(np.sum(actions**2))
        info["reward/action_penalty"] = action_pen

        action_rate_pen = -0.005 * float(np.sum((actions - previous_actions) ** 2))
        info["reward/action_rate_penalty"] = action_rate_pen

        # weighted sum
        total = (
            self.weights.reaching * reaching
            + self.weights.grasping * grasping
            + self.weights.lifting * lifting
            + self.weights.holding * holding
            + self.weights.drop * drop
            + self.weights.action * action_pen
            + self.weights.action_rate * action_rate_pen
        )

        info["reward/total"] = total
        info["metrics/num_finger_contacts"] = float(num_fingers_in_contact)
        info["metrics/object_height"] = obj_height
        info["metrics/object_speed"] = obj_speed
        info["metrics/mean_fingertip_dist"] = float(np.mean(dists))

        return total, info
