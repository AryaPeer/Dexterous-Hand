import numpy as np

from dexterous_hand.config import PegRewardConfig


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))


class PegRewardCalculator:
    def __init__(
        self,
        config: PegRewardConfig,
        table_height: float,
        peg_half_length: float,
        peg_radius: float = 0.0,
    ) -> None:
        """Peg reward calculator.

        @param config: peg reward weights and thresholds
        @type config: PegRewardConfig
        @param table_height: table surface height
        @type table_height: float
        @param peg_half_length: half-length of the peg cylinder body
        @type peg_half_length: float
        @param peg_radius: peg cylinder radius
        @type peg_radius: float
        """

        self.weights = config.weights
        self.peg_length = peg_half_length * 2.0 + peg_radius * 2.0
        self.lift_target = config.lift_target
        self.drop_penalty_value = config.drop_penalty
        self.complete_bonus = config.complete_bonus
        self.depth_reward_scale = config.depth_reward_scale
        self.force_threshold = config.force_threshold
        self.idle_stage0_penalty = config.idle_stage0_penalty
        self.lateral_gate_k = config.lateral_gate_k
        self.idle_stage_cutoff = config.idle_stage_cutoff
        self.idle_grace_steps = config.idle_grace_steps
        self.success_threshold = config.success_threshold
        self.peg_hold_steps = config.peg_hold_steps
        self.reach_tanh_k = config.reach_tanh_k
        self.fingertip_weights = np.asarray(config.fingertip_weights, dtype=np.float64)
        self.table_height = table_height
        self._was_lifted = False
        self._insertion_hold_steps = 0
        self._initial_peg_height = table_height
        self._idle_steps = 0

    def reset(self, initial_peg_height: float | None = None) -> None:
        """Reset for a new episode.

        @param initial_peg_height: peg height at episode start; None defaults to table.
        @type initial_peg_height: float | None
        """

        self._was_lifted = False
        self._insertion_hold_steps = 0
        self._idle_steps = 0
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
        """Total peg-in-hole reward across the four curriculum stages.

        @param stage: current curriculum stage (0=reach, 1=grasp, 2=lift, 3=insert)
        @type stage: int
        @param finger_positions: (5, 3) per-finger representative positions
        @type finger_positions: np.ndarray
        @param peg_position: (3,) peg center
        @type peg_position: np.ndarray
        @param peg_axis: (3,) peg long axis (world frame)
        @type peg_axis: np.ndarray
        @param hole_position: (3,) hole center
        @type hole_position: np.ndarray
        @param hole_axis: (3,) hole insertion axis (world frame)
        @type hole_axis: np.ndarray
        @param insertion_depth: current peg-tip insertion depth (m)
        @type insertion_depth: float
        @param contact_force_magnitude: total normal force on hole walls (N)
        @type contact_force_magnitude: float
        @param num_fingers_in_contact: fingers touching the peg
        @type num_fingers_in_contact: int
        @param contact_finger_indices: set of finger indices in contact
        @type contact_finger_indices: set[int]
        @param peg_height: peg z-coordinate (world frame)
        @type peg_height: float
        @param peg_linvel: (3,) peg linear velocity
        @type peg_linvel: np.ndarray
        @param actions: (22,) current actions
        @type actions: np.ndarray
        @param previous_actions: (22,) last step's actions
        @type previous_actions: np.ndarray
        @return: (total, info) weighted reward sum and per-component breakdown
        @rtype: tuple[float, dict[str, float]]
        """

        info: dict[str, float] = {}
        n_contacts = num_fingers_in_contact

        # reach: weighted fingertip-to-peg distance
        dists = np.linalg.norm(finger_positions - peg_position, axis=1)
        weighted_dist = float(np.sum(self.fingertip_weights * dists) / self.fingertip_weights.sum())
        reach = 1.0 - float(np.tanh(self.reach_tanh_k * weighted_dist))
        info["reward/reach"] = reach

        # grasp quality: thumb opposing the rest
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

        contact_scale = min(n_contacts / 3.0, 1.0)
        tripod_bonus = 0.5 if (thumb_contact and len(others) >= 2) else 0.0
        grasp = contact_scale * (0.3 + 0.7 * opposition) + tripod_bonus
        info["reward/grasp"] = grasp

        lift_height = max(peg_height - self._initial_peg_height, 0.0)
        lift = float(min(lift_height / self.lift_target, 1.5)) * contact_scale
        info["reward/lift"] = lift

        was_lifted_prev = self._was_lifted
        if lift_height >= self.lift_target:
            self._was_lifted = True

        # align + insertion drive: gated on the peg actually being above the table
        lateral_dist = float(np.linalg.norm(peg_position[:2] - hole_position[:2]))
        axis_align = abs(float(np.dot(peg_axis, hole_axis)))
        lateral_factor_align = 1.0 - float(np.tanh(self.lateral_gate_k * lateral_dist))
        peg_clearance = max(peg_height - self.table_height - self.peg_length * 0.5, 0.0)
        align_weight = _sigmoid((peg_clearance - 0.02) * 150.0)
        align = axis_align * lateral_factor_align * align_weight * contact_scale
        info["reward/align"] = align

        insertion_drive = (
            align_weight
            * lateral_factor_align
            * axis_align
            * contact_scale
            * max(float(-peg_linvel[2]), 0.0)
            * 5.0
        )
        info["reward/insertion_drive"] = insertion_drive

        lateral_factor_depth = 1.0 - float(np.tanh(self.lateral_gate_k * lateral_dist))
        insertion_fraction = float(np.clip(insertion_depth / self.peg_length, 0.0, 1.0))
        depth_reward = self.depth_reward_scale * insertion_fraction * lateral_factor_depth
        info["reward/depth"] = depth_reward

        if insertion_fraction > self.success_threshold:
            self._insertion_hold_steps += 1
        else:
            self._insertion_hold_steps = 0
        complete = (
            self.complete_bonus
            * axis_align
            * lateral_factor_align
            * contact_scale
            * _sigmoid(20.0 * (insertion_fraction - self.success_threshold))
            * _sigmoid((self._insertion_hold_steps - self.peg_hold_steps) / 2.0)
        )
        info["reward/complete"] = complete

        force_excess = max(0.0, contact_force_magnitude - self.force_threshold)
        force_penalty = -0.01 * force_excess**2
        info["reward/force_penalty"] = force_penalty

        just_dropped = was_lifted_prev and lift_height < 0.01
        drop = self.drop_penalty_value if just_dropped else 0.0
        if just_dropped:
            self._was_lifted = False
        info["reward/drop"] = drop

        action_penalty = -0.0002 * float(np.sum(actions**2))
        info["reward/action_penalty"] = action_penalty
        del previous_actions

        idle_active = n_contacts == 0 and stage < self.idle_stage_cutoff
        if idle_active:
            self._idle_steps += 1
        else:
            self._idle_steps = 0
        idle_raw = (
            self.idle_stage0_penalty if self._idle_steps >= self.idle_grace_steps else 0.0
        )
        idle_penalty = self.weights.idle_stage0 * idle_raw
        info["reward/idle_stage0_penalty"] = idle_penalty

        total = (
            self.weights.reach * reach
            + self.weights.grasp * grasp
            + self.weights.opposition * opposition
            + self.weights.lift * lift
            + self.weights.align * align
            + self.weights.depth * depth_reward
            + self.weights.complete * complete
            + self.weights.force * force_penalty
            + self.weights.drop * drop
            + self.weights.action_penalty * action_penalty
            + self.weights.insertion_drive * insertion_drive
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
