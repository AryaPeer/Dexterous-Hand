import numpy as np

from dexterous_hand.config import ReorientRewardConfig
from dexterous_hand.utils.quaternion import quat_angular_distance


class ReorientRewardCalculator:
    def __init__(self, config: ReorientRewardConfig, initial_cube_pos: np.ndarray) -> None:
        """Set up the reorientation reward calculator.

        Args:
            config: reward weights and thresholds
            initial_cube_pos: (3,) where the cube starts (so we can penalize displacement)
        """
        self.weights = config.weights
        self.success_threshold = config.success_threshold
        self.success_hold_steps = config.success_hold_steps
        self.drop_penalty_value = config.drop_penalty
        self._initial_cube_pos = initial_cube_pos.copy()
        self._success_steps = 0

    def reset(self, initial_cube_pos: np.ndarray | None = None) -> None:
        """Reset for a new episode (or new target within the same episode).

        Args:
            initial_cube_pos: new starting pos if provided, otherwise keeps the old one
        """
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
        fingertip_positions: np.ndarray,
        actions: np.ndarray,
        previous_actions: np.ndarray,
        dropped: bool,
    ) -> tuple[float, dict[str, float], bool]:
        """Compute the reorientation reward.

        Rewards tracking the target orientation, penalizes dropping the cube,
        moving it too fast, or displacing it from its starting position.

        Args:
            cube_quat: (4,) current cube quaternion (w, x, y, z)
            target_quat: (4,) where we want the cube to be oriented
            cube_pos: (3,) cube position
            cube_linvel: (3,) cube linear velocity
            cube_angvel: (3,) cube angular velocity
            fingertip_positions: (5, 3) fingertip positions
            actions: (20,) current actions
            previous_actions: (20,) last step's actions
            dropped: True if the cube fell

        Returns:
            total: weighted reward sum
            info: reward/metric breakdown
            target_reached: True if we held the target orientation long enough
        """
        info: dict[str, float] = {}

        ang_dist = quat_angular_distance(cube_quat, target_quat)

        orientation_tracking = float(np.exp(-5.0 * ang_dist))
        info["reward/orientation_tracking"] = orientation_tracking

        at_target = ang_dist < self.success_threshold
        if at_target:
            self._success_steps += 1
        else:
            self._success_steps = 0
        orientation_success = 5.0 if at_target else 0.0
        info["reward/orientation_success"] = orientation_success

        target_reached = self._success_steps >= self.success_hold_steps

        cube_drop = self.drop_penalty_value if dropped else 0.0
        info["reward/cube_drop"] = cube_drop

        velocity_penalty = -0.1 * float(
            np.linalg.norm(cube_linvel) ** 2 + 0.5 * np.linalg.norm(cube_angvel) ** 2
        )
        info["reward/velocity_penalty"] = velocity_penalty

        dists = np.linalg.norm(fingertip_positions - cube_pos, axis=1)
        fingertip_distance = float(np.exp(-5.0 * np.mean(dists)))
        info["reward/fingertip_distance"] = fingertip_distance

        pos_error_sq = float(np.linalg.norm(cube_pos - self._initial_cube_pos) ** 2)
        position_penalty = -5.0 * pos_error_sq
        info["reward/position_penalty"] = position_penalty

        action_penalty = -0.005 * float(np.sum(actions**2))
        info["reward/action_penalty"] = action_penalty

        action_rate_penalty = -0.002 * float(np.sum((actions - previous_actions) ** 2))
        info["reward/action_rate_penalty"] = action_rate_penalty

        total = (
            self.weights.orientation_tracking * orientation_tracking
            + self.weights.orientation_success * orientation_success
            + self.weights.cube_drop * cube_drop
            + self.weights.velocity_penalty * velocity_penalty
            + self.weights.fingertip_distance * fingertip_distance
            + self.weights.position_penalty * position_penalty
            + self.weights.action_penalty * action_penalty
            + self.weights.action_rate_penalty * action_rate_penalty
        )

        info["reward/total"] = total
        info["metrics/angular_distance"] = ang_dist
        info["metrics/mean_fingertip_dist"] = float(np.mean(dists))
        info["metrics/cube_displacement"] = float(np.sqrt(pos_error_sq))
        info["metrics/success_steps"] = float(self._success_steps)

        return total, info, target_reached
