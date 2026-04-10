import numpy as np

from dexterous_hand.config import PegRewardConfig


class PegRewardCalculator:
    def __init__(
        self,
        config: PegRewardConfig,
        table_height: float,
        peg_half_length: float,
    ) -> None:
        """Peg insertion reward calculator.

        @param config: reward weights and thresholds
        @type config: PegRewardConfig
        @param table_height: table surface height for lift calculations
        @type table_height: float
        @param peg_half_length: half-length of the peg along its long axis
        @type peg_half_length: float
        """

        self.weights = config.weights
        self.peg_length = peg_half_length * 2.0
        self.lift_target = 0.1
        self.drop_penalty_value = config.drop_penalty
        self.complete_bonus = config.complete_bonus
        self.force_threshold = config.force_threshold
        self.idle_stage0_penalty = config.idle_stage0_penalty
        self.min_contacts_for_align = config.min_contacts_for_align
        self.table_height = table_height
        self._was_lifted = False
        self._insertion_hold_steps = 0
        self._initial_peg_height = table_height

    def reset(self, initial_peg_height: float | None = None) -> None:
        """Reset for a new episode."""

        self._was_lifted = False
        self._insertion_hold_steps = 0
        if initial_peg_height is None:
            self._initial_peg_height = self.table_height
        else:
            self._initial_peg_height = float(initial_peg_height)

    def compute(
        self,
        stage: int,
        finger_positions: np.ndarray,
        peg_position: np.ndarray,
        peg_axis: np.ndarray,
        hole_position: np.ndarray,
        hole_axis: np.ndarray,
        insertion_depth: float,
        contact_force_magnitude: float,
        num_fingers_in_contact: int,
        contact_finger_indices: set[int],
        peg_height: float,
        actions: np.ndarray,
        previous_actions: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Peg insertion reward: reach -> grasp -> lift -> align -> insert -> complete.

        @param stage: task stage (0=reach, 1=grasp, 2=align, 3=insert)
        @type stage: int
        @param finger_positions: (5, 3) per-finger representative positions
        @type finger_positions: np.ndarray
        @param peg_position: (3,) peg center
        @type peg_position: np.ndarray
        @param peg_axis: (3,) peg direction (world frame)
        @type peg_axis: np.ndarray
        @param hole_position: (3,) hole center
        @type hole_position: np.ndarray
        @param hole_axis: (3,) hole direction (world frame)
        @type hole_axis: np.ndarray
        @param insertion_depth: how far in the peg is
        @type insertion_depth: float
        @param contact_force_magnitude: force from hole walls
        @type contact_force_magnitude: float
        @param num_fingers_in_contact: fingers touching peg
        @type num_fingers_in_contact: int
        @param contact_finger_indices: set of finger indices currently in contact
        @type contact_finger_indices: set[int]
        @param peg_height: peg z-position
        @type peg_height: float
        @param actions: current actions
        @type actions: np.ndarray
        @param previous_actions: last step's actions
        @type previous_actions: np.ndarray
        @return: (total, info) weighted reward sum and per-component breakdown
        @rtype: tuple[float, dict[str, float]]
        """

        info: dict[str, float] = {}

        dists = np.linalg.norm(finger_positions - peg_position, axis=1)
        contact_factor = min(num_fingers_in_contact / 2.0, 1.0)
        reach = float(np.exp(-10.0 * np.mean(dists))) * (1.0 - 0.5 * contact_factor)
        info["reward/reach"] = reach

        side_contacts = 0
        for idx in contact_finger_indices:
            if finger_positions[idx, 2] <= peg_position[2] + 0.015:
                side_contacts += 1
        side_ratio = side_contacts / max(num_fingers_in_contact, 1)
        info["reward/grasp_quality"] = side_ratio

        grasp = (num_fingers_in_contact / 5.0) * (0.3 + 0.7 * side_ratio)
        info["reward/grasp"] = grasp

        lift_hold_gate = 1.0 if num_fingers_in_contact >= 2 else 0.0
        lift_height = max(peg_height - self._initial_peg_height, 0.0)
        lift = float(np.clip(lift_height, 0.0, self.lift_target) / self.lift_target) * lift_hold_gate
        info["reward/lift"] = lift

        if lift_height >= self.lift_target:
            self._was_lifted = True

        lateral_dist = float(np.linalg.norm(peg_position[:2] - hole_position[:2]))
        raw_align = float(np.dot(peg_axis, hole_axis)) * float(np.exp(-20.0 * lateral_dist))

        if num_fingers_in_contact < self.min_contacts_for_align or stage < 2:
            align = 0.0
            depth_reward = 0.0
        elif stage == 2:
            align = max(raw_align, 0.0)
            depth_reward = 0.0
        else:
            align = max(raw_align, 0.0)
            depth_reward = 10.0 * (insertion_depth / self.peg_length) if lateral_dist < 0.005 else 0.0
        info["reward/align"] = align
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

        dropped = self._was_lifted and lift_height < 0.01
        drop = self.drop_penalty_value if dropped else 0.0
        info["reward/drop"] = drop

        smoothness = -0.005 * float(np.sum((actions - previous_actions) ** 2))
        info["reward/smoothness"] = smoothness

        action_mag_raw = -0.01 * float(np.sum(actions**2))
        action_magnitude_penalty = self.weights.action_magnitude * action_mag_raw
        info["reward/action_magnitude_penalty"] = action_magnitude_penalty

        idle_raw = self.idle_stage0_penalty if (stage == 0 and num_fingers_in_contact == 0) else 0.0
        idle_stage0_penalty = self.weights.idle_stage0 * idle_raw
        info["reward/idle_stage0_penalty"] = idle_stage0_penalty

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
            + action_magnitude_penalty
            + idle_stage0_penalty
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
