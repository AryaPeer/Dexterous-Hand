
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from dexterous_hand.config import PegRewardConfig, ReorientRewardConfig, RewardConfig  # noqa: E402
from dexterous_hand.rewards.cpu.grasp_reward import GraspRewardCalculator  # noqa: E402
from dexterous_hand.rewards.cpu.peg_reward import PegRewardCalculator  # noqa: E402
from dexterous_hand.rewards.cpu.reorient_reward import ReorientRewardCalculator  # noqa: E402
from dexterous_hand.rewards.gpu.grasp_reward import (  # noqa: E402
    grasp_reward,
    init_grasp_reward_state,
)
from dexterous_hand.rewards.gpu.peg_reward import (  # noqa: E402
    init_peg_reward_state,
    peg_reward,
)
from dexterous_hand.rewards.gpu.reorient_reward import (  # noqa: E402
    init_reorient_reward_state,
    reorient_reward,
)

N_SAMPLES = 20
ATOL = 1e-5
RTOL = 1e-4

def _rand_unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(3)
    return v / (np.linalg.norm(v) + 1e-9)

def _rand_quat(rng: np.random.Generator) -> np.ndarray:
    q = rng.standard_normal(4)
    return q / (np.linalg.norm(q) + 1e-9)

def _mask_from_set(indices: set[int]) -> np.ndarray:
    mask = np.zeros(5, dtype=bool)
    for i in indices:
        mask[i] = True
    return mask

def _keys_intersect(a: dict, b: dict) -> list[str]:
    return sorted(set(a.keys()) & set(b.keys()))

class TestGraspParity:
    def test_parity_random_inputs(self):
        rng = np.random.default_rng(12345)
        cfg = RewardConfig()
        table_height = 0.4

        for _ in range(N_SAMPLES):
            obj_pos = np.array(
                [rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(0.4, 0.6)]
            )
            finger_positions = obj_pos + rng.uniform(-0.05, 0.05, size=(5, 3))
            obj_vel = rng.uniform(-0.2, 0.2, size=3)
            n_contacts = int(rng.integers(0, 6))
            contact_idx = (
                set(rng.choice(5, size=n_contacts, replace=False).tolist()) if n_contacts else set()
            )
            mask = _mask_from_set(contact_idx)
            actions = rng.uniform(-1, 1, size=22)
            prev_actions = rng.uniform(-1, 1, size=22)

            initial_obj_h = float(rng.uniform(0.4, 0.45))
            np_calc = GraspRewardCalculator(cfg, table_height=table_height)
            np_calc.reset(initial_object_height=initial_obj_h, is_sphere=False)
            np_total, np_info = np_calc.compute(
                finger_positions=finger_positions,
                object_position=obj_pos,
                object_linear_velocity=obj_vel,
                num_fingers_in_contact=n_contacts,
                contact_finger_indices=contact_idx,
                actions=actions,
                previous_actions=prev_actions,
            )

            jax_state = init_grasp_reward_state(initial_obj_h, table_height)
            jax_total, _, jax_info = grasp_reward(
                state=jax_state,
                finger_positions=jnp.asarray(finger_positions),
                object_position=jnp.asarray(obj_pos),
                object_linear_velocity=jnp.asarray(obj_vel),
                finger_contact_mask=jnp.asarray(mask),
                actions=jnp.asarray(actions),
                previous_actions=jnp.asarray(prev_actions),
                table_height=table_height,
                lift_target=cfg.lift_target,
                hold_velocity_threshold=cfg.hold_velocity_threshold,
                drop_penalty_value=cfg.drop_penalty,
                no_contact_idle_penalty=cfg.no_contact_idle_penalty,
                weights=cfg.weights,
                reach_tanh_k=cfg.reach_tanh_k,
                hold_height_k=cfg.hold_height_smoothness_k,
                hold_velocity_k=cfg.hold_velocity_smoothness_k,
                fingertip_weights=cfg.fingertip_weights,
            )

            np.testing.assert_allclose(float(jax_total), np_total, atol=ATOL, rtol=RTOL)
            for k in _keys_intersect(np_info, jax_info):
                np.testing.assert_allclose(
                    float(jax_info[k]),
                    np_info[k],
                    atol=ATOL,
                    rtol=RTOL,
                    err_msg=f"grasp parity mismatch at key={k}",
                )

class TestPegParity:
    def test_parity_random_inputs(self):
        rng = np.random.default_rng(54321)
        cfg = PegRewardConfig()
        scene_cfg_half_len = 0.03
        peg_length = scene_cfg_half_len * 2.0
        table_height = 0.82

        for _ in range(N_SAMPLES):
            peg_pos = np.array(
                [rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), rng.uniform(0.82, 0.95)]
            )
            peg_axis = _rand_unit_vec(rng)
            hole_pos = np.array([rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), 0.88])
            hole_axis = _rand_unit_vec(rng)
            finger_positions = peg_pos + rng.uniform(-0.05, 0.05, size=(5, 3))

            insertion_depth = float(rng.uniform(-0.01, peg_length * 1.1))
            contact_force = float(rng.uniform(0.0, 20.0))
            n_contacts = int(rng.integers(0, 6))
            contact_idx = (
                set(rng.choice(5, size=n_contacts, replace=False).tolist()) if n_contacts else set()
            )
            mask = _mask_from_set(contact_idx)
            peg_height = float(peg_pos[2])
            peg_linvel = rng.uniform(-0.2, 0.2, size=3)
            actions = rng.uniform(-1, 1, size=22)
            prev_actions = rng.uniform(-1, 1, size=22)
            stage = int(rng.integers(0, 5))

            initial_peg_h = float(rng.uniform(0.82, 0.84))
            np_calc = PegRewardCalculator(
                cfg, table_height=table_height, peg_half_length=scene_cfg_half_len
            )
            np_calc.reset(initial_peg_height=initial_peg_h)
            np_total, np_info = np_calc.compute(
                stage=stage,
                finger_positions=finger_positions,
                peg_position=peg_pos,
                peg_axis=peg_axis,
                hole_position=hole_pos,
                hole_axis=hole_axis,
                insertion_depth=insertion_depth,
                contact_force_magnitude=contact_force,
                num_fingers_in_contact=n_contacts,
                contact_finger_indices=contact_idx,
                peg_height=peg_height,
                peg_linvel=peg_linvel,
                actions=actions,
                previous_actions=prev_actions,
            )

            jax_state = init_peg_reward_state(initial_peg_h)
            jax_total, _, jax_info = peg_reward(
                state=jax_state,
                stage=jnp.asarray(stage),
                finger_positions=jnp.asarray(finger_positions),
                peg_position=jnp.asarray(peg_pos),
                peg_axis=jnp.asarray(peg_axis),
                hole_position=jnp.asarray(hole_pos),
                hole_axis=jnp.asarray(hole_axis),
                insertion_depth=jnp.asarray(insertion_depth),
                contact_force_magnitude=jnp.asarray(contact_force),
                finger_contact_mask=jnp.asarray(mask),
                peg_height=jnp.asarray(peg_height),
                peg_linvel=jnp.asarray(peg_linvel),
                actions=jnp.asarray(actions),
                previous_actions=jnp.asarray(prev_actions),
                weights=cfg.weights,
                peg_length=peg_length,
                lift_target=cfg.lift_target,
                table_height=table_height,
                drop_penalty_value=cfg.drop_penalty,
                complete_bonus=cfg.complete_bonus,
                force_threshold=cfg.force_threshold,
                idle_stage0_penalty=cfg.idle_stage0_penalty,
                lateral_gate_k=cfg.lateral_gate_k,
                idle_stage_cutoff=cfg.idle_stage_cutoff,
                success_threshold=cfg.success_threshold,
                peg_hold_steps=cfg.peg_hold_steps,
                reach_tanh_k=cfg.reach_tanh_k,
                fingertip_weights=cfg.fingertip_weights,
            )

            np.testing.assert_allclose(float(jax_total), np_total, atol=ATOL, rtol=RTOL)
            for k in _keys_intersect(np_info, jax_info):
                np.testing.assert_allclose(
                    float(jax_info[k]),
                    np_info[k],
                    atol=ATOL,
                    rtol=RTOL,
                    err_msg=f"peg parity mismatch at key={k}",
                )

class TestReorientParity:
    def test_parity_random_inputs(self):
        rng = np.random.default_rng(99999)
        cfg = ReorientRewardConfig()

        for _ in range(N_SAMPLES):
            cube_quat = _rand_quat(rng)
            target_quat = _rand_quat(rng)
            cube_pos = rng.uniform(-0.05, 0.05, size=3)
            cube_linvel = rng.uniform(-0.2, 0.2, size=3)
            finger_positions = cube_pos + rng.uniform(-0.05, 0.05, size=(5, 3))
            n_contacts = int(rng.integers(0, 6))
            mask = np.zeros(5, dtype=bool)
            if n_contacts:
                mask[rng.choice(5, size=n_contacts, replace=False)] = True
            actions = rng.uniform(-1, 1, size=22)
            prev_actions = rng.uniform(-1, 1, size=22)
            dropped = bool(rng.integers(0, 2))
            initial_cube_pos = rng.uniform(-0.05, 0.05, size=3)

            np_calc = ReorientRewardCalculator(cfg, initial_cube_pos=initial_cube_pos)
            np_total, np_info, np_reached = np_calc.compute(
                cube_quat=cube_quat,
                target_quat=target_quat,
                cube_pos=cube_pos,
                cube_linvel=cube_linvel,
                finger_positions=finger_positions,
                num_fingers_in_contact=n_contacts,
                actions=actions,
                previous_actions=prev_actions,
                dropped=dropped,
            )

            jax_state = init_reorient_reward_state(jnp.asarray(initial_cube_pos))
            jax_total, _, jax_info, jax_reached = reorient_reward(
                state=jax_state,
                cube_quat=jnp.asarray(cube_quat),
                target_quat=jnp.asarray(target_quat),
                cube_pos=jnp.asarray(cube_pos),
                cube_linvel=jnp.asarray(cube_linvel),
                finger_positions=jnp.asarray(finger_positions),
                finger_contact_mask=jnp.asarray(mask),
                actions=jnp.asarray(actions),
                previous_actions=jnp.asarray(prev_actions),
                dropped=jnp.asarray(dropped),
                weights=cfg.weights,
                success_threshold=cfg.success_threshold,
                success_hold_steps=cfg.success_hold_steps,
                drop_penalty_value=cfg.drop_penalty,
                contact_bonus_value=cfg.contact_bonus,
                no_contact_penalty_value=cfg.no_contact_penalty,
                min_contacts_for_rotation=cfg.min_contacts_for_rotation,
                angular_progress_clip=cfg.angular_progress_clip,
                orientation_success_k=cfg.orientation_success_k,
                tracking_k=cfg.tracking_k,
            )

            np.testing.assert_allclose(float(jax_total), np_total, atol=ATOL, rtol=RTOL)
            assert bool(jax_reached) == np_reached
            for k in _keys_intersect(np_info, jax_info):
                np.testing.assert_allclose(
                    float(jax_info[k]),
                    np_info[k],
                    atol=ATOL,
                    rtol=RTOL,
                    err_msg=f"reorient parity mismatch at key={k}",
                )
