
from typing import NamedTuple

import jax.numpy as jnp

from dexterous_hand.config import ReorientRewardWeights
from dexterous_hand.utils.gpu.quaternion import quat_angular_distance

class ReorientRewardState(NamedTuple):

    success_steps: jnp.ndarray              
    prev_ang_dist: jnp.ndarray                
    initial_cube_pos: jnp.ndarray        
    has_prev: jnp.ndarray                                       

def init_reorient_reward_state(initial_cube_pos: jnp.ndarray) -> ReorientRewardState:

    return ReorientRewardState(
        success_steps=jnp.array(0, dtype=jnp.int32),
        prev_ang_dist=jnp.array(0.0),
        initial_cube_pos=initial_cube_pos,
        has_prev=jnp.array(False),
    )

def reorient_reward(
    state: ReorientRewardState,
    cube_quat: jnp.ndarray,
    target_quat: jnp.ndarray,
    cube_pos: jnp.ndarray,
    cube_linvel: jnp.ndarray,
    finger_positions: jnp.ndarray,
    finger_contact_mask: jnp.ndarray,
    actions: jnp.ndarray,
    previous_actions: jnp.ndarray,
    dropped: jnp.ndarray,
    weights: ReorientRewardWeights,
    success_threshold: float,
    success_hold_steps: int,
    drop_penalty_value: float,
    contact_bonus_value: float,
    no_contact_penalty_value: float,
    min_contacts_for_rotation: int,
    angular_progress_clip: float = 0.2,
    orientation_success_k: float = 5.0,
    tracking_k: float = 5.0,
) -> tuple[jnp.ndarray, ReorientRewardState, dict[str, jnp.ndarray], jnp.ndarray]:

    n_contacts = jnp.sum(finger_contact_mask).astype(jnp.float32)

    ang_dist = quat_angular_distance(cube_quat, target_quat)

    angular_progress = jnp.where(state.has_prev, state.prev_ang_dist - ang_dist, 0.0)
    angular_progress = jnp.clip(angular_progress, -angular_progress_clip, angular_progress_clip)

    orientation_tracking = jnp.exp(-tracking_k * ang_dist)

                                                                      
    soft_contact_scale = jnp.minimum(n_contacts / float(min_contacts_for_rotation), 1.0)
    orientation_success = jnp.exp(-orientation_success_k * ang_dist) * soft_contact_scale

                                                                             
    at_target = ang_dist < success_threshold
    enough_contacts = n_contacts >= min_contacts_for_rotation
    new_success_steps = jnp.where(
        at_target & enough_contacts,
        state.success_steps + 1,
        jnp.array(0, dtype=jnp.int32),
    )
    target_reached = new_success_steps >= success_hold_steps

    cube_drop = jnp.where(dropped, drop_penalty_value, 0.0)

    velocity_penalty = -0.1 * jnp.sum(cube_linvel**2)

    dists = jnp.linalg.norm(finger_positions - cube_pos, axis=1)
    fingertip_distance = jnp.exp(-5.0 * jnp.mean(dists))

    action_penalty = -0.002 * jnp.sum(actions**2)
    action_rate_penalty = -0.001 * jnp.sum((actions - previous_actions) ** 2)

    contact_raw = contact_bonus_value * jnp.minimum(n_contacts / 3.0, 1.0)
    finger_contact_bonus = weights.contact_bonus * contact_raw

                                                               
    no_contact_ramp = jnp.maximum(1.0 - n_contacts, 0.0)
    no_contact_raw = no_contact_penalty_value * no_contact_ramp
    no_contact_penalty = weights.no_contact * no_contact_raw

    pos_error = jnp.linalg.norm(cube_pos - state.initial_cube_pos)
    position_stability_penalty = -pos_error

    total = (
        weights.angular_progress * angular_progress
        + weights.orientation_tracking * orientation_tracking
        + weights.orientation_success * orientation_success
        + weights.cube_drop * cube_drop
        + weights.velocity_penalty * velocity_penalty
        + weights.fingertip_distance * fingertip_distance
        + weights.action_penalty * action_penalty
        + weights.action_rate_penalty * action_rate_penalty
        + finger_contact_bonus
        + no_contact_penalty
        + weights.position_stability * position_stability_penalty
    )

    new_state = ReorientRewardState(
        success_steps=new_success_steps,
        prev_ang_dist=ang_dist,
        initial_cube_pos=state.initial_cube_pos,
        has_prev=jnp.array(True),
    )

    info = {
        "reward/angular_progress": angular_progress,
        "reward/orientation_tracking": orientation_tracking,
        "reward/orientation_success": orientation_success,
        "reward/cube_drop": cube_drop,
        "reward/velocity_penalty": velocity_penalty,
        "reward/fingertip_distance": fingertip_distance,
        "reward/action_penalty": action_penalty,
        "reward/action_rate_penalty": action_rate_penalty,
        "reward/finger_contact_bonus": finger_contact_bonus,
        "reward/no_contact_penalty": no_contact_penalty,
        "reward/position_stability": position_stability_penalty,
        "reward/total": total,
        "metrics/angular_distance": ang_dist,
        "metrics/num_finger_contacts": n_contacts,
        "metrics/mean_fingertip_dist": jnp.mean(dists),
        "metrics/cube_displacement": pos_error,
        "metrics/success_steps": new_success_steps.astype(jnp.float32),
    }

    return total, new_state, info, target_reached
