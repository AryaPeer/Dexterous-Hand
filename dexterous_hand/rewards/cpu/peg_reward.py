
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

        info: dict[str, float] = {}
        n_contacts = num_fingers_in_contact

                                                                                  
        dists = np.linalg.norm(finger_positions - peg_position, axis=1)
        weighted_dist = float(np.sum(self.fingertip_weights * dists) / self.fingertip_weights.sum())
        reach = 1.0 - float(np.tanh(self.reach_tanh_k * weighted_dist))
        info["reward/reach"] = reach

                                                                                              
                                                                                             
                                                                                           
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
        grasp = contact_scale * (0.3 + 0.7 * opposition)
        info["reward/grasp"] = grasp

                                                                             
                                                             
        lift_height = max(peg_height - self._initial_peg_height, 0.0)
        lift = float(min(lift_height / self.lift_target, 1.5)) * contact_scale
        info["reward/lift"] = lift

        was_lifted_prev = self._was_lifted
        if lift_height >= self.lift_target:
            self._was_lifted = True

        lateral_dist = float(np.linalg.norm(peg_position[:2] - hole_position[:2]))
        axis_align = float(np.dot(peg_axis, hole_axis))
        lateral_factor_align = 1.0 - float(np.tanh(self.lateral_gate_k * lateral_dist))
        peg_clearance = max(peg_height - self.table_height - self.peg_length * 0.5, 0.0)
        align_weight = _sigmoid((peg_clearance - 0.02) * 150.0)
        raw_align = axis_align * lateral_factor_align
        align = max(raw_align, 0.0) * align_weight * contact_scale
        info["reward/align"] = align

                                                                                   
                                                                                      
        lateral_factor_depth = 1.0 - float(np.tanh(self.lateral_gate_k * lateral_dist))
        insertion_fraction = float(np.clip(insertion_depth / self.peg_length, 0.0, 1.0))
        depth_reward = self.depth_reward_scale * insertion_fraction * lateral_factor_depth
        info["reward/depth"] = depth_reward

                                                                                   
                                                                                         
        if insertion_fraction > self.success_threshold:
            self._insertion_hold_steps += 1
            complete = (
                self.complete_bonus if self._insertion_hold_steps >= self.peg_hold_steps else 0.0
            )
        else:
            self._insertion_hold_steps = 0
            complete = 0.0
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
