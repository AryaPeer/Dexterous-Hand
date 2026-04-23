
import numpy as np

from dexterous_hand.config import RewardConfig

def _sigmoid(x: float) -> float:

    return 1.0 / (1.0 + float(np.exp(-x)))

class GraspRewardCalculator:

    def __init__(self, config: RewardConfig, table_height: float) -> None:

        self.weights = config.weights
        self.reach_tanh_k = config.reach_tanh_k
        self.lift_target = config.lift_target
        self.hold_velocity_threshold = config.hold_velocity_threshold
        self.hold_height_k = config.hold_height_smoothness_k
        self.hold_velocity_k = config.hold_velocity_smoothness_k
        self.drop_penalty_value = config.drop_penalty
        self.no_contact_idle_penalty = config.no_contact_idle_penalty
        self.idle_grace_steps = config.idle_grace_steps
        self.fingertip_weights = np.asarray(config.fingertip_weights, dtype=np.float64)
        self.table_height = table_height
        self._was_lifted = False
        self._initial_height_above_table = 0.0
        self._idle_steps = 0

    def reset(self, initial_object_height: float | None = None) -> None:

        self._was_lifted = False
        self._idle_steps = 0
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

        info: dict[str, float] = {}
        n_contacts = num_fingers_in_contact

        obj_height = object_position[2]
        height_above_table = obj_height - self.table_height
        lift_height = max(height_above_table - self._initial_height_above_table, 0.0)

                                                                        
        dists = np.linalg.norm(finger_positions - object_position, axis=1)
        weighted_dist = float(np.sum(self.fingertip_weights * dists) / self.fingertip_weights.sum())
        reaching = 1.0 - float(np.tanh(self.reach_tanh_k * weighted_dist))
        info["reward/reaching"] = reaching

                                                                           
        thumb_contact = 0 in contact_finger_indices
        others = contact_finger_indices - {0}
        if thumb_contact and len(others) >= 1:
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

                                                                              
                                                                               
                                                                            
                                                            
        contact_scale = min(n_contacts / 3.0, 1.0)
        grasping = contact_scale * (0.3 + 0.7 * opposition)
        info["reward/grasping"] = grasping

                                                                       
                                                                          
                                       
        lifting = float(min(lift_height / self.lift_target, 1.5)) * contact_scale
        info["reward/lifting"] = lifting

        obj_speed = float(np.linalg.norm(object_linear_velocity))
        height_gate = _sigmoid(self.hold_height_k * (lift_height - self.lift_target + 0.04))
        speed_gate = _sigmoid(self.hold_velocity_k * (self.hold_velocity_threshold - obj_speed))
        holding = height_gate * speed_gate * contact_scale
        info["reward/holding"] = holding

        was_lifted_prev = self._was_lifted
        if lift_height >= self.lift_target:
            self._was_lifted = True

        # charge the penalty on every drop; clear was_lifted so a re-lift
        # can be credited again, but do NOT zero the penalty on regrasp
        just_dropped = was_lifted_prev and lift_height < 0.01
        drop = self.drop_penalty_value if just_dropped else 0.0
        if just_dropped:
            self._was_lifted = False
        info["reward/drop"] = drop

                                                                                 
                                                          
        idle_active = n_contacts == 0
        if idle_active:
            self._idle_steps += 1
        else:
            self._idle_steps = 0
        idle_raw = (
            self.no_contact_idle_penalty if self._idle_steps >= self.idle_grace_steps else 0.0
        )
        idle_penalty = self.weights.idle * idle_raw
        info["reward/idle_penalty"] = idle_penalty

                                                                                  
        action_rate_pen = -5e-3 * float(np.sum((actions - previous_actions) ** 2))
        info["reward/action_rate_penalty"] = action_rate_pen

        total = (
            self.weights.reaching * reaching
            + self.weights.grasping * grasping
            + self.weights.opposition * opposition
            + self.weights.lifting * lifting
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
