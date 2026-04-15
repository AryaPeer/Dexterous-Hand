import numpy as np

from dexterous_hand.config import PegRewardConfig


FINGERTIP_WEIGHTS = np.array([2.5, 1.0, 1.0, 1.0, 1.0])  # thumb weighted heavier


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))


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
        self.lift_target = config.lift_target
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
        peg_linvel: np.ndarray,
        actions: np.ndarray,
        previous_actions: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Dense shaped reward for peg-in-hole assembly.

        Potentials are smooth (1 - tanh) so the gradient doesn't vanish at distance.
        Stage-specific rewards are weighted by soft sigmoids over progress signals
        (contact count, lift height) instead of binary gates, so the agent can't
        fall off a cliff by momentarily losing contact.

        @param stage: task stage tag (logged only; reward doesn't branch on it)
        @type stage: int
        @param finger_positions: (5, 3) per-finger representative positions; row 0 is thumb
        @type finger_positions: np.ndarray
        @param peg_position: (3,) peg center
        @type peg_position: np.ndarray
        @param peg_axis: (3,) peg direction (world frame)
        @type peg_axis: np.ndarray
        @param hole_position: (3,) hole center
        @type hole_position: np.ndarray
        @param hole_axis: (3,) hole direction (world frame)
        @type hole_axis: np.ndarray
        @param insertion_depth: how far the peg has entered the hole
        @type insertion_depth: float
        @param contact_force_magnitude: combined force on peg from hole walls
        @type contact_force_magnitude: float
        @param num_fingers_in_contact: fingers touching the peg
        @type num_fingers_in_contact: int
        @param contact_finger_indices: set of finger indices currently in contact
        @type contact_finger_indices: set[int]
        @param peg_height: peg z-position
        @type peg_height: float
        @param peg_linvel: (3,) peg linear velocity — positive z means lifting
        @type peg_linvel: np.ndarray
        @param actions: current actions
        @type actions: np.ndarray
        @param previous_actions: last step's actions
        @type previous_actions: np.ndarray
        @return: (total, info) weighted reward sum and per-component breakdown
        @rtype: tuple[float, dict[str, float]]
        """

        info: dict[str, float] = {}
        n_contacts = num_fingers_in_contact

        # reach: thumb-weighted fingertip distance through a smooth tanh potential
        dists = np.linalg.norm(finger_positions - peg_position, axis=1)
        weighted_dist = float(np.sum(FINGERTIP_WEIGHTS * dists) / FINGERTIP_WEIGHTS.sum())
        reach = 1.0 - float(np.tanh(5.0 * weighted_dist))
        info["reward/reach"] = reach

        # antipodal opposition: thumb on the opposite side of the peg from the other contacts
        thumb_contact = 0 in contact_finger_indices
        others = contact_finger_indices - {0}
        if thumb_contact and len(others) >= 1:
            thumb_vec = finger_positions[0] - peg_position
            other_stack = np.stack([finger_positions[i] - peg_position for i in others])
            mean_other_vec = other_stack.mean(axis=0)
            thumb_n = float(np.linalg.norm(thumb_vec)) + 1e-6
            other_n = float(np.linalg.norm(mean_other_vec)) + 1e-6
            opposition = float(-np.dot(thumb_vec / thumb_n, mean_other_vec / other_n))
            opposition = max(opposition, 0.0)
        else:
            opposition = 0.0
        info["reward/grasp_quality"] = opposition

        grasp = (n_contacts / 5.0) * (0.3 + 0.7 * opposition)
        info["reward/grasp"] = grasp

        # lift: contact-scaled ramp, no binary gate
        contact_scale = min(n_contacts / 3.0, 1.0)
        lift_height = max(peg_height - self._initial_peg_height, 0.0)
        lift = float(np.clip(lift_height, 0.0, self.lift_target) / self.lift_target) * contact_scale
        info["reward/lift"] = lift

        if lift_height >= self.lift_target:
            self._was_lifted = True

        # upward velocity bonus: only fires under a real grasp, not a nudge
        upward_bonus = max(float(peg_linvel[2]), 0.0) * contact_scale
        info["reward/upward"] = upward_bonus

        # align + depth: smooth everywhere, align_weight softly unlocks once the peg is
        # clear of the table (works for both free-start and pre-grasped episodes)
        lateral_dist = float(np.linalg.norm(peg_position[:2] - hole_position[:2]))
        axis_align = float(np.dot(peg_axis, hole_axis))
        lateral_factor_align = 1.0 - float(np.tanh(10.0 * lateral_dist))
        peg_clearance = max(peg_height - self.table_height - self.peg_length * 0.5, 0.0)
        align_weight = _sigmoid((peg_clearance - 0.02) * 50.0)
        raw_align = axis_align * lateral_factor_align
        align = max(raw_align, 0.0) * align_weight * contact_scale
        info["reward/align"] = align

        lateral_factor_depth = 1.0 - float(np.tanh(40.0 * lateral_dist))
        depth_reward = 10.0 * (insertion_depth / self.peg_length) * lateral_factor_depth
        info["reward/depth"] = depth_reward

        # completion: still requires holding the insertion for 10 steps
        insertion_fraction = insertion_depth / self.peg_length
        if insertion_fraction > 0.9:
            self._insertion_hold_steps += 1
            complete = self.complete_bonus if self._insertion_hold_steps >= 10 else 0.0
        else:
            self._insertion_hold_steps = 0
            complete = 0.0
        info["reward/complete"] = complete

        force_excess = max(0.0, contact_force_magnitude - self.force_threshold)
        force_penalty = -0.01 * force_excess ** 2
        info["reward/force_penalty"] = force_penalty

        # drop: latches on first lift, clears when the agent successfully regrasps
        dropped_now = self._was_lifted and lift_height < 0.01
        if dropped_now and n_contacts >= 2:
            self._was_lifted = False
            dropped_now = False
        drop = self.drop_penalty_value if dropped_now else 0.0
        info["reward/drop"] = drop

        # action shaping: rate only (magnitude penalty dropped — it rewarded stillness)
        smoothness = -5e-5 * float(np.sum((actions - previous_actions) ** 2))
        info["reward/smoothness"] = smoothness
        info["reward/action_magnitude_penalty"] = 0.0

        # idle penalty: fire whenever the hand isn't touching the peg, not only stage 0
        idle_raw = self.idle_stage0_penalty if n_contacts == 0 else 0.0
        idle_penalty = self.weights.idle_stage0 * idle_raw
        info["reward/idle_stage0_penalty"] = idle_penalty

        total = (
            self.weights.reach * reach
            + self.weights.grasp * grasp
            + self.weights.opposition * opposition
            + self.weights.lift * lift
            + self.weights.upward * upward_bonus
            + self.weights.align * align
            + self.weights.depth * depth_reward
            + self.weights.complete * complete
            + self.weights.force * force_penalty
            + self.weights.drop * drop
            + self.weights.smoothness * smoothness
            + idle_penalty
        )

        info["reward/total"] = total
        info["metrics/stage"] = float(stage)
        info["metrics/num_finger_contacts"] = float(n_contacts)
        info["metrics/peg_height"] = peg_height
        info["metrics/insertion_depth"] = insertion_depth
        info["metrics/contact_force"] = contact_force_magnitude
        info["metrics/lateral_distance"] = lateral_dist
        info["metrics/insertion_hold_steps"] = float(self._insertion_hold_steps)

        return total, info
