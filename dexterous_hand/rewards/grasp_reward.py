import numpy as np

from dexterous_hand.config import RewardConfig


FINGERTIP_WEIGHTS = np.array([2.5, 1.0, 1.0, 1.0, 1.0])  # thumb weighted heavier


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))


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
        self._is_sphere = False
        self._initial_height_above_table = 0.0

    def reset(self, initial_object_height: float | None = None, is_sphere: bool = False) -> None:
        """Reset for a new episode."""

        self._was_lifted = False
        self._is_sphere = is_sphere
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
        """Dense shaped grasp reward: reach -> grasp -> lift -> hold.

        Potentials are smooth (1 - tanh), stage weighting is smooth (contact_scale),
        and the drop flag is recoverable so a bounce-then-regrasp doesn't keep paying
        the penalty forever.

        @param finger_positions: (5, 3) per-finger positions; row 0 is thumb
        @type finger_positions: np.ndarray
        @param object_position: (3,) object center
        @type object_position: np.ndarray
        @param object_linear_velocity: (3,) object velocity
        @type object_linear_velocity: np.ndarray
        @param num_fingers_in_contact: fingers touching the object
        @type num_fingers_in_contact: int
        @param contact_finger_indices: set of finger indices currently in contact
        @type contact_finger_indices: set[int]
        @param actions: (22,) current actions
        @type actions: np.ndarray
        @param previous_actions: (22,) last step's actions
        @type previous_actions: np.ndarray
        @return: (total, info) weighted reward sum and per-component breakdown
        @rtype: tuple[float, dict[str, float]]
        """

        info: dict[str, float] = {}
        n_contacts = num_fingers_in_contact

        obj_height = object_position[2]
        height_above_table = obj_height - self.table_height
        lift_height = max(height_above_table - self._initial_height_above_table, 0.0)

        # reach: thumb-weighted distance through a smooth tanh potential
        dists = np.linalg.norm(finger_positions - object_position, axis=1)
        weighted_dist = float(np.sum(FINGERTIP_WEIGHTS * dists) / FINGERTIP_WEIGHTS.sum())
        reaching = 1.0 - float(np.tanh(5.0 * weighted_dist))
        info["reward/reaching"] = reaching

        # antipodal opposition — thumb vs. mean of other contacting fingers
        thumb_contact = 0 in contact_finger_indices
        others = contact_finger_indices - {0}
        if self._is_sphere and n_contacts > 0:
            opposition = 1.0
        elif thumb_contact and len(others) >= 1:
            thumb_vec = finger_positions[0] - object_position
            other_stack = np.stack([finger_positions[i] - object_position for i in others])
            mean_other_vec = other_stack.mean(axis=0)
            thumb_n = float(np.linalg.norm(thumb_vec)) + 1e-6
            other_n = float(np.linalg.norm(mean_other_vec)) + 1e-6
            opposition = float(-np.dot(thumb_vec / thumb_n, mean_other_vec / other_n))
            opposition = max(opposition, 0.0)
        else:
            opposition = 0.0
        info["reward/grasp_quality"] = opposition

        grasping = (n_contacts / 5.0) * (0.3 + 0.7 * opposition)
        info["reward/grasping"] = grasping

        # lift + hold: contact-scaled, not binary-gated
        contact_scale = min(n_contacts / 3.0, 1.0)
        lifting = float(np.clip(lift_height, 0.0, self.lift_target) / self.lift_target) * contact_scale
        info["reward/lifting"] = lifting

        # upward-velocity bonus — only fires under an actual grasp
        upward_bonus = max(float(object_linear_velocity[2]), 0.0) * contact_scale
        info["reward/upward"] = upward_bonus

        obj_speed = float(np.linalg.norm(object_linear_velocity))
        is_above = lift_height >= self.lift_target
        is_stable = obj_speed < self.hold_velocity_threshold
        holding = (1.0 if (is_above and is_stable) else 0.0) * contact_scale
        info["reward/holding"] = holding

        if is_above:
            self._was_lifted = True

        # drop: latches on first real lift, clears when the agent regrasps
        dropped_now = self._was_lifted and lift_height < 0.01
        if dropped_now and n_contacts >= 2:
            self._was_lifted = False
            dropped_now = False
        drop = self.drop_penalty_value if dropped_now else 0.0
        info["reward/drop"] = drop

        idle_raw = self.no_contact_idle_penalty if (n_contacts == 0 and lifting < 0.01) else 0.0
        idle_penalty = self.weights.idle * idle_raw
        info["reward/idle_penalty"] = idle_penalty

        # action_penalty dropped (its 1.5x weight was pushing the agent to freeze)
        info["reward/action_penalty"] = 0.0

        action_rate_pen = -5e-5 * float(np.sum((actions - previous_actions) ** 2))
        info["reward/action_rate_penalty"] = action_rate_pen

        total = (
            self.weights.reaching * reaching
            + self.weights.grasping * grasping
            + self.weights.opposition * opposition
            + self.weights.lifting * lifting
            + self.weights.upward * upward_bonus
            + self.weights.holding * holding
            + self.weights.drop * drop
            + self.weights.action_rate * action_rate_pen
            + idle_penalty
        )

        info["reward/total"] = total
        info["metrics/num_finger_contacts"] = float(n_contacts)
        info["metrics/object_height"] = obj_height
        info["metrics/object_speed"] = obj_speed
        info["metrics/mean_fingertip_dist"] = float(np.mean(dists))

        return total, info
