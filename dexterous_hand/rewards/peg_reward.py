import numpy as np

from dexterous_hand.config import PegRewardConfig


class PegRewardCalculator:
    def __init__(
        self,
        config: PegRewardConfig,
        table_height: float,
    ) -> None:
        """Set up the peg insertion reward calculator.

        Args:
            config: reward weights and thresholds
            table_height: table surface height for lift calculations
        """
        self.weights = config.weights
        self.peg_length = config.peg_half_length * 2.0
        self.drop_penalty_value = config.drop_penalty
        self.complete_bonus = config.complete_bonus
        self.force_threshold = config.force_threshold
        self.table_height = table_height
        self._was_grasped = False
        self._insertion_hold_steps = 0

    def reset(self) -> None:
        """Reset tracking state for a new episode."""
        self._was_grasped = False
        self._insertion_hold_steps = 0

    def compute(
        self,
        stage: int,
        fingertip_positions: np.ndarray,  # (5, 3)
        peg_position: np.ndarray,  # (3,)
        peg_axis: np.ndarray,  # (3,) — peg's local Z in world frame
        hole_position: np.ndarray,  # (3,)
        hole_axis: np.ndarray,  # (3,) — hole's local Z in world frame
        insertion_depth: float,
        contact_force_magnitude: float,
        num_fingers_in_contact: int,
        peg_height: float,
        actions: np.ndarray,
        previous_actions: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Compute the peg insertion reward.

        Covers reaching the peg, grasping it, lifting, aligning with
        the hole, inserting, and a big bonus for completing the task.

        Args:
            stage: current task stage (0=reach, 1=grasp, 2=align, 3=insert)
            fingertip_positions: (5, 3) fingertip positions
            peg_position: (3,) peg center
            peg_axis: (3,) which way the peg is pointing (world frame)
            hole_position: (3,) hole center
            hole_axis: (3,) hole direction (world frame)
            insertion_depth: how far in the peg is
            contact_force_magnitude: force on the peg from hole walls
            num_fingers_in_contact: fingers touching the peg
            peg_height: peg z-position
            actions: current actions
            previous_actions: last step's actions

        Returns:
            total: weighted reward sum
            info: reward/metric breakdown
        """
        info: dict[str, float] = {}

        if num_fingers_in_contact >= 3:
            self._was_grasped = True

        if stage == 0:
            dists = np.linalg.norm(fingertip_positions - peg_position, axis=1)
            reach = float(np.exp(-10.0 * np.mean(dists)))
        else:
            reach = 0.0
        info["reward/reach"] = reach

        grasp = min(num_fingers_in_contact / 3.0, 1.0)
        info["reward/grasp"] = grasp

        lift = float(np.clip(peg_height - self.table_height, 0.0, 0.1) / 0.1)
        info["reward/lift"] = lift

        lateral_dist = float(np.linalg.norm(peg_position[:2] - hole_position[:2]))
        align = float(np.dot(peg_axis, hole_axis)) * float(np.exp(-20.0 * lateral_dist))
        info["reward/align"] = align

        depth_reward = 10.0 * (insertion_depth / self.peg_length) if lateral_dist < 0.005 else 0.0
        info["reward/depth"] = depth_reward

        insertion_fraction = insertion_depth / self.peg_length
        if insertion_fraction > 0.9:
            self._insertion_hold_steps += 1
            complete = self.complete_bonus if self._insertion_hold_steps >= 10 else 0.0
        else:
            self._insertion_hold_steps = 0
            complete = 0.0
        info["reward/complete"] = complete

        force_excess = max(0.0, contact_force_magnitude - self.force_threshold)
        force_penalty = -0.01 * force_excess**2
        info["reward/force_penalty"] = force_penalty

        dropped = self._was_grasped and peg_height < self.table_height - 0.02
        drop = self.drop_penalty_value if dropped else 0.0
        info["reward/drop"] = drop

        smoothness = -0.002 * float(np.sum((actions - previous_actions) ** 2))
        info["reward/smoothness"] = smoothness

        total = (
            self.weights.reach * reach
            + self.weights.grasp * grasp
            + self.weights.lift * lift
            + self.weights.align * align
            + self.weights.depth * depth_reward
            + self.weights.complete * complete
            + self.weights.force * force_penalty
            + self.weights.drop * drop
            + self.weights.smoothness * smoothness
        )

        info["reward/total"] = total
        info["metrics/stage"] = float(stage)
        info["metrics/num_finger_contacts"] = float(num_fingers_in_contact)
        info["metrics/peg_height"] = peg_height
        info["metrics/insertion_depth"] = insertion_depth
        info["metrics/contact_force"] = contact_force_magnitude
        info["metrics/lateral_distance"] = lateral_dist
        info["metrics/insertion_hold_steps"] = float(self._insertion_hold_steps)

        return total, info
