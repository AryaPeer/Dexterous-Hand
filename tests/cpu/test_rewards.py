import numpy as np
from numpy.testing import assert_allclose

from dexterous_hand.config import (
    PegRewardConfig,
    PegSceneConfig,
    ReorientRewardConfig,
    RewardConfig,
)
from dexterous_hand.rewards.cpu.grasp_reward import GraspRewardCalculator
from dexterous_hand.rewards.cpu.peg_reward import PegRewardCalculator
from dexterous_hand.rewards.cpu.reorient_reward import ReorientRewardCalculator

                 
def _fingertips_at(pos: np.ndarray) -> np.ndarray:
    return np.tile(pos, (5, 1))

ZERO_ACTIONS = np.zeros(22)
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
ZERO3 = np.zeros(3)

                            
class TestGraspReward:
    def make_calc(self) -> GraspRewardCalculator:
        return GraspRewardCalculator(RewardConfig(), table_height=0.4)

    def test_all_keys_present(self):
        calc = self.make_calc()
        _, info = calc.compute(
            finger_positions=_fingertips_at(np.array([0.0, 0.0, 0.5])),
            object_position=np.array([0.0, 0.0, 0.5]),
            object_linear_velocity=ZERO3,
            num_fingers_in_contact=0,
            contact_finger_indices=set(),
            actions=ZERO_ACTIONS,
            previous_actions=ZERO_ACTIONS,
        )

        expected_keys = {
            "reward/reaching",
            "reward/grasping",
            "reward/lifting",
            "reward/holding",
            "reward/drop",
            "reward/success",
            "reward/idle_penalty",
            "reward/action_penalty",
            "reward/grasp_quality",
            "reward/total",
            "metrics/num_finger_contacts",
            "metrics/object_height",
            "metrics/object_speed",
            "metrics/mean_fingertip_dist",
            "metrics/success_hold_steps",
            "is_success",
        }
        assert expected_keys == set(info.keys())

    def test_reaching_decreases_with_distance(self):
        calc = self.make_calc()
        obj = np.array([0.0, 0.0, 0.5])
        _, info_close = calc.compute(
            _fingertips_at(obj + 0.01),
            obj,
            ZERO3,
            0,
            set(),
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        _, info_far = calc.compute(
            _fingertips_at(obj + 0.2),
            obj,
            ZERO3,
            0,
            set(),
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        assert info_close["reward/reaching"] > info_far["reward/reaching"]

    def test_grasping_proportional_to_contacts(self):
        calc = self.make_calc()
        obj = np.array([0.0, 0.0, 0.5])
                                                                                       
        fingers = np.array(
            [
                [+0.05, 0.0, 0.5],         
                [-0.05, 0.0, 0.5],
                [-0.05, 0.0, 0.5],
                [-0.05, 0.0, 0.5],
                [-0.05, 0.0, 0.5],
            ]
        )
        _, info5 = calc.compute(
            fingers,
            obj,
            ZERO3,
            5,
            {0, 1, 2, 3, 4},
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        _, info0 = calc.compute(
            _fingertips_at(obj),
            obj,
            ZERO3,
            0,
            set(),
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        # contact_scale = tanh(5/2) ≈ 0.987, raw_opposition = 1.0,
        # tripod_bonus = 0.5 → 0.987 * 1.0 + 0.5 ≈ 1.487
        assert_allclose(info5["reward/grasping"], np.tanh(2.5) + 0.5, atol=1e-3)
        assert_allclose(info0["reward/grasping"], 0.0)

    def test_lifting_scales_with_height(self):
        calc = self.make_calc()
        obj_at_table = np.array([0.0, 0.0, 0.4])
        obj_lifted = np.array([0.0, 0.0, 0.5])
        _, info_low = calc.compute(
            _fingertips_at(obj_at_table),
            obj_at_table,
            ZERO3,
            4,
            {0, 1, 2, 3},
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        _, info_high = calc.compute(
            _fingertips_at(obj_lifted),
            obj_lifted,
            ZERO3,
            4,
            {0, 1, 2, 3},
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        assert_allclose(info_low["reward/lifting"], 0.0)
        # lift_height=0.1, lift_target=0.04 → ratio 2.5 clamped to 1.5;
        # contact_scale = tanh(4/2) ≈ 0.964 → 1.5 * 0.964 ≈ 1.446
        expected_high = 1.5 * float(np.tanh(2.0))
        assert_allclose(info_high["reward/lifting"], expected_high, rtol=1e-3)

    def test_holding_requires_height_and_stability(self):
                                                                                    
                                                                                 
        calc = self.make_calc()

        obj_high = np.array([0.0, 0.0, 0.6])                                    
        _, info_hold = calc.compute(
            _fingertips_at(obj_high),
            obj_high,
            np.array([0.0, 0.0, 0.0]),                    
            5,
            {0, 1, 2, 3, 4},
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )

        _, info_fast = calc.compute(
            _fingertips_at(obj_high),
            obj_high,
            np.array([1.0, 0.0, 0.0]),                               
            5,
            {0, 1, 2, 3, 4},
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )

        obj_low = np.array([0.0, 0.0, 0.405])
        _, info_low = calc.compute(
            _fingertips_at(obj_low),
            obj_low,
            ZERO3,
            5,
            {0, 1, 2, 3, 4},
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )

        assert info_hold["reward/holding"] > info_fast["reward/holding"] * 50
        assert info_hold["reward/holding"] > info_low["reward/holding"] * 1.5
        assert info_fast["reward/holding"] < 0.01
        assert info_low["reward/holding"] < 0.5

    def test_drop_penalty_triggers(self):
        calc = self.make_calc()
        obj_high = np.array([0.0, 0.0, 0.51])

        calc.compute(
            _fingertips_at(obj_high),
            obj_high,
            ZERO3,
            5,
            {0, 1, 2, 3, 4},
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )

        obj_dropped = np.array([0.0, 0.0, 0.405])
        _, info = calc.compute(
            _fingertips_at(obj_dropped),
            obj_dropped,
            ZERO3,
            0,
            set(),
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        assert info["reward/drop"] == -20.0

    def test_drop_penalty_not_triggered_without_lift(self):
        calc = self.make_calc()
        obj_low = np.array([0.0, 0.0, 0.4])
        _, info = calc.compute(
            _fingertips_at(obj_low),
            obj_low,
            ZERO3,
            0,
            set(),
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        assert_allclose(info["reward/drop"], 0.0)

    def test_action_penalty_scales_with_action_norm(self):
        # IsaacGymEnvs ShadowHand scale: -0.0002·||a||². zero actions give 0,
        # max-magnitude actions give -0.0002·22 = -0.0044.
        calc = self.make_calc()
        obj = np.array([0.0, 0.0, 0.5])

        _, info_zero = calc.compute(
            _fingertips_at(obj), obj, ZERO3, 0, set(), np.zeros(22), np.zeros(22),
        )
        _, info_big = calc.compute(
            _fingertips_at(obj), obj, ZERO3, 0, set(), np.ones(22), np.ones(22),
        )
        assert_allclose(info_zero["reward/action_penalty"], 0.0)
        assert_allclose(info_big["reward/action_penalty"], -0.0002 * 22, rtol=1e-6)

    def test_reward_is_finite(self):
        calc = self.make_calc()
        rng = np.random.default_rng(42)
        for _ in range(20):
            n_contacts = int(rng.integers(0, 6))
            contact_indices = (
                set(rng.choice(5, size=n_contacts, replace=False)) if n_contacts > 0 else set()
            )
            total, info = calc.compute(
                finger_positions=rng.uniform(-1, 1, (5, 3)),
                object_position=rng.uniform(-1, 1, 3),
                object_linear_velocity=rng.uniform(-5, 5, 3),
                num_fingers_in_contact=n_contacts,
                contact_finger_indices=contact_indices,
                actions=rng.uniform(-1, 1, 22),
                previous_actions=rng.uniform(-1, 1, 22),
            )
            assert np.isfinite(total)
            for v in info.values():
                assert np.isfinite(v)

                               
class TestReorientReward:
    def make_calc(self) -> ReorientRewardCalculator:
        return ReorientRewardCalculator(
            ReorientRewardConfig(),
            np.array([0.0, 0.0, 0.5]),
        )

    def _default_kwargs(self, **overrides) -> dict:
        defaults = dict(
            cube_quat=IDENTITY_QUAT,
            target_quat=IDENTITY_QUAT,
            cube_pos=np.array([0.0, 0.0, 0.5]),
            cube_linvel=ZERO3,
            finger_positions=_fingertips_at(np.array([0.0, 0.0, 0.5])),
            num_fingers_in_contact=3,
            actions=ZERO_ACTIONS,
            previous_actions=ZERO_ACTIONS,
            dropped=False,
        )
        defaults.update(overrides)
        return defaults

    def test_all_keys_present(self):
        calc = self.make_calc()
        _, info, _ = calc.compute(**self._default_kwargs())

        expected_keys = {
            "reward/angular_progress",
            "reward/orientation",
            "reward/cube_drop",
            "reward/action_penalty",
            "reward/finger_contact_bonus",
            "reward/no_contact_penalty",
            "reward/total",
            "metrics/angular_distance",
            "metrics/num_finger_contacts",
            "metrics/success_steps",
        }
        assert expected_keys == set(info.keys())

    def test_perfect_orientation_max_tracking(self):
        # with 3 contacts (>= min_contacts_for_rotation=2), orientation term
        # peaks at 1.0·(alpha + (1-alpha)·1) = 1.0.
        calc = self.make_calc()
        _, info, _ = calc.compute(**self._default_kwargs())
        assert_allclose(info["reward/orientation"], 1.0, rtol=1e-6)

    def test_90_degree_error_lower_tracking(self):
        calc = self.make_calc()
        from dexterous_hand.utils.cpu.quaternion import quat_from_axis_angle

        q90 = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        _, info, _ = calc.compute(**self._default_kwargs(target_quat=q90))
        # 3 contacts → soft_contact_scale=1, gate=1, orientation=exp(-k·d).
        # config default tracking_k=2.0.
        expected = float(np.exp(-2.0 * np.pi / 2))
        assert_allclose(info["reward/orientation"], expected, rtol=1e-5)

    def test_orientation_contact_gate_lowers_no_contact(self):
        # at 0 contacts, orientation = exp(-k·d) · alpha. with 7-weight and
        # alpha=3/7, the contribution is 3.0·exp(-k·d), matching the
        # pre-collapse orientation_tracking (weight 3) unconditional term.
        calc = self.make_calc()
        _, info_full, _ = calc.compute(**self._default_kwargs(num_fingers_in_contact=3))
        _, info_none, _ = calc.compute(**self._default_kwargs(num_fingers_in_contact=0))
        alpha = ReorientRewardConfig().orientation_contact_alpha
        assert_allclose(info_none["reward/orientation"], alpha, rtol=1e-6)
        assert info_full["reward/orientation"] > info_none["reward/orientation"]

    def test_success_detection_after_hold_steps(self):
        calc = self.make_calc()
        target_reached = False
        for _ in range(25):
            _, _, target_reached = calc.compute(**self._default_kwargs())
        assert target_reached is True

    def test_success_resets_on_failure(self):
        calc = self.make_calc()
        from dexterous_hand.utils.cpu.quaternion import quat_from_axis_angle

        far_quat = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi / 2)

        for _ in range(20):
            calc.compute(**self._default_kwargs())

        calc.compute(**self._default_kwargs(target_quat=far_quat))

        target_reached = False
        for _ in range(5):
            _, _, target_reached = calc.compute(**self._default_kwargs())
        assert target_reached is False

    def test_drop_penalty(self):
        calc = self.make_calc()
        _, info, _ = calc.compute(**self._default_kwargs(dropped=True))
        assert_allclose(info["reward/cube_drop"], -20.0)

    def test_angular_progress_zero_on_first_step(self):
        calc = self.make_calc()
        from dexterous_hand.utils.cpu.quaternion import quat_from_axis_angle

        q_off = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 4)
        _, info, _ = calc.compute(**self._default_kwargs(cube_quat=q_off))
        assert_allclose(info["reward/angular_progress"], 0.0)

    def test_angular_progress_positive_when_approaching_target(self):
        calc = self.make_calc()
        from dexterous_hand.utils.cpu.quaternion import quat_from_axis_angle

        q_far = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        q_near = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 4)

        calc.compute(**self._default_kwargs(cube_quat=q_far))
        _, info, _ = calc.compute(**self._default_kwargs(cube_quat=q_near))
        assert info["reward/angular_progress"] > 0.0

                          
class TestPegReward:
    TABLE_HEIGHT = 0.4

    def make_calc(self) -> PegRewardCalculator:
        return PegRewardCalculator(
            PegRewardConfig(),
            table_height=self.TABLE_HEIGHT,
            peg_half_length=PegSceneConfig().peg_half_length,
        )

    def _default_kwargs(self, **overrides) -> dict:
        defaults = dict(
            stage=0,
            finger_positions=_fingertips_at(np.array([0.0, 0.0, 0.45])),
            peg_position=np.array([0.0, 0.0, 0.45]),
            peg_axis=np.array([0.0, 0.0, 1.0]),
            hole_position=np.array([0.1, 0.0, 0.45]),
            hole_axis=np.array([0.0, 0.0, 1.0]),
            insertion_depth=0.0,
            contact_force_magnitude=0.0,
            num_fingers_in_contact=0,
            contact_finger_indices=set(),
            peg_height=0.45,
            peg_linvel=ZERO3,
            actions=ZERO_ACTIONS,
            previous_actions=ZERO_ACTIONS,
        )
        defaults.update(overrides)
        return defaults

    def test_all_keys_present(self):
        calc = self.make_calc()
        _, info = calc.compute(**self._default_kwargs())

        expected_keys = {
            "reward/reach",
            "reward/grasp",
            "reward/grasp_quality",
            "reward/lift",
            "reward/align",
            "reward/depth",
            "reward/complete",
            "reward/force_penalty",
            "reward/drop",
            "reward/action_penalty",
            "reward/idle_stage0_penalty",
            "reward/insertion_drive",
            "reward/total",
            "metrics/stage",
            "metrics/num_finger_contacts",
            "metrics/peg_height",
            "metrics/insertion_depth",
            "metrics/contact_force",
            "metrics/lateral_distance",
            "metrics/insertion_hold_steps",
        }
        assert expected_keys == set(info.keys())

    def test_reach_decreases_with_distance(self):
        calc = self.make_calc()
        peg = np.array([0.0, 0.0, 0.45])
        _, info_close = calc.compute(
            **self._default_kwargs(
                peg_position=peg,
                finger_positions=_fingertips_at(peg + 0.01),
            )
        )
        _, info_far = calc.compute(
            **self._default_kwargs(
                peg_position=peg,
                finger_positions=_fingertips_at(peg + 0.3),
            )
        )
        assert info_close["reward/reach"] > info_far["reward/reach"]

    def test_grasp_proportional(self):
        calc = self.make_calc()
        peg = np.array([0.0, 0.0, 0.45])
                                                                             
        antipodal_fingers = np.array(
            [
                [+0.02, 0.0, 0.45],         
                [-0.02, 0.0, 0.45],
                [-0.02, 0.0, 0.45],
                [-0.02, 0.0, 0.45],
                [-0.02, 0.0, 0.45],
            ]
        )
        _, info5 = calc.compute(
            **self._default_kwargs(
                peg_position=peg,
                finger_positions=antipodal_fingers,
                num_fingers_in_contact=5,
                contact_finger_indices={0, 1, 2, 3, 4},
            )
        )
        _, info0 = calc.compute(**self._default_kwargs(num_fingers_in_contact=0))
        assert_allclose(info5["reward/grasp"], 1.5, atol=1e-3)
        assert_allclose(info0["reward/grasp"], 0.0)

    def test_opposition_zero_without_thumb_or_others(self):
        calc = self.make_calc()
                                                                  
        _, info_thumb_only = calc.compute(
            **self._default_kwargs(num_fingers_in_contact=1, contact_finger_indices={0})
        )
        assert_allclose(info_thumb_only["reward/grasp_quality"], 0.0)

                                                     
        _, info_no_thumb = calc.compute(
            **self._default_kwargs(num_fingers_in_contact=3, contact_finger_indices={1, 2, 3})
        )
        assert_allclose(info_no_thumb["reward/grasp_quality"], 0.0)

    def test_lift_scales_with_height(self):
        calc = self.make_calc()
                                                                                  
        calc.reset(initial_peg_height=self.TABLE_HEIGHT)
        _, info_low = calc.compute(
            **self._default_kwargs(
                peg_height=self.TABLE_HEIGHT,
                num_fingers_in_contact=3,
                contact_finger_indices={0, 1, 2},
            )
        )
        _, info_high = calc.compute(
            **self._default_kwargs(
                peg_height=self.TABLE_HEIGHT + 0.1,
                num_fingers_in_contact=3,
                contact_finger_indices={0, 1, 2},
            )
        )
        assert_allclose(info_low["reward/lift"], 0.0)
        assert_allclose(info_high["reward/lift"], 1.0)

    def test_lift_scales_with_contact_count(self):
        calc = self.make_calc()
        calc.reset(initial_peg_height=self.TABLE_HEIGHT)

        _, info_two = calc.compute(
            **self._default_kwargs(
                peg_height=self.TABLE_HEIGHT + 0.1,
                num_fingers_in_contact=2,
                contact_finger_indices={0, 1},
            )
        )
        assert_allclose(info_two["reward/lift"], 2.0 / 3.0, rtol=1e-6)

    def test_lift_zero_with_no_contact(self):
        calc = self.make_calc()
        _, info = calc.compute(**self._default_kwargs(peg_height=0.5, num_fingers_in_contact=0))
        assert_allclose(info["reward/lift"], 0.0, rtol=1e-6)

    def test_insertion_depth_reward(self):
        calc = self.make_calc()
        peg_length = PegSceneConfig().peg_half_length * 2.0        
        hole_pos = np.array([0.0, 0.0, 0.45])
        peg_pos = np.array([0.0, 0.0, 0.45])                    
        _, info = calc.compute(
            **self._default_kwargs(
                stage=3,
                num_fingers_in_contact=3,
                peg_position=peg_pos,
                hole_position=hole_pos,
                insertion_depth=peg_length * 0.5,
            )
        )
        expected_depth = 10.0 * 0.5
        assert_allclose(info["reward/depth"], expected_depth, rtol=1e-6)

    def test_insertion_depth_decays_with_lateral(self):
        calc = self.make_calc()
        peg_length = PegSceneConfig().peg_half_length * 2.0
        peg_pos = np.array([0.0, 0.0, 0.45])
        _, info_aligned = calc.compute(
            **self._default_kwargs(
                peg_position=peg_pos,
                hole_position=np.array([0.0, 0.0, 0.45]),
                insertion_depth=peg_length * 0.5,
            )
        )
        _, info_off = calc.compute(
            **self._default_kwargs(
                peg_position=peg_pos,
                hole_position=np.array([0.01, 0.0, 0.45]),                
                insertion_depth=peg_length * 0.5,
            )
        )
        _, info_far = calc.compute(
            **self._default_kwargs(
                peg_position=peg_pos,
                hole_position=np.array([0.1, 0.0, 0.45]),                 
                insertion_depth=peg_length * 0.5,
            )
        )
                                       
        assert info_aligned["reward/depth"] > info_off["reward/depth"] > info_far["reward/depth"]
        assert info_far["reward/depth"] >= 0.0

    def test_completion_bonus_after_hold(self):
        # smooth bonus: 250 * sigmoid(20*(frac-0.7)) * sigmoid(hold/5 - 1).
        # asymptote at long hold is 250 * sigmoid(20*0.25) ≈ 250 * 0.9933.
        calc = self.make_calc()
        peg_length = PegSceneConfig().peg_half_length * 2.0
        kwargs = self._default_kwargs(
            insertion_depth=peg_length * 0.95,
            peg_position=np.array([0.0, 0.0, 0.45]),
            hole_position=np.array([0.0, 0.0, 0.45]),
        )
        for _ in range(50):
            _, info = calc.compute(**kwargs)

        asymptote = 250.0 * 1.0 / (1.0 + float(np.exp(-20.0 * 0.25)))
        assert_allclose(info["reward/complete"], asymptote, rtol=1e-3)

    def test_force_penalty_above_threshold(self):
        calc = self.make_calc()
        # threshold bumped from 5.0 → 15.0 in phase 3.8; 20 N gives
        # excess 5 N → penalty -0.01 * 25 = -0.25
        _, info = calc.compute(**self._default_kwargs(contact_force_magnitude=20.0))
        assert_allclose(info["reward/force_penalty"], -0.25, rtol=1e-6)

    def test_force_penalty_below_threshold(self):
        calc = self.make_calc()
        # 10 N is now below the 15 N threshold
        _, info = calc.compute(**self._default_kwargs(contact_force_magnitude=10.0))
        assert_allclose(info["reward/force_penalty"], 0.0)

    def test_drop_penalty(self):
        calc = self.make_calc()

                                                       
        calc.compute(
            **self._default_kwargs(
                peg_height=self.TABLE_HEIGHT + 0.15,
                num_fingers_in_contact=3,
                contact_finger_indices={0, 1, 2},
            )
        )

        _, info = calc.compute(
            **self._default_kwargs(
                peg_height=self.TABLE_HEIGHT,
                num_fingers_in_contact=0,
            )
        )
        assert_allclose(info["reward/drop"], -20.0)

    def test_drop_penalty_requires_real_lift(self):
        calc = self.make_calc()

                                                        
        calc.compute(
            **self._default_kwargs(
                peg_height=self.TABLE_HEIGHT + 0.02,
                num_fingers_in_contact=3,
                contact_finger_indices={0, 1, 2},
            )
        )

        _, info = calc.compute(
            **self._default_kwargs(
                peg_height=self.TABLE_HEIGHT - 0.05,
                num_fingers_in_contact=0,
            )
        )
        assert_allclose(info["reward/drop"], 0.0)

    def test_action_penalty_zero_for_zero_actions(self):
        # IsaacGymEnvs scale: -0.0002·||a||². zero actions → 0 penalty.
        calc = self.make_calc()
        _, info = calc.compute(
            **self._default_kwargs(
                actions=np.zeros(22),
                previous_actions=np.zeros(22),
            )
        )
        assert_allclose(info["reward/action_penalty"], 0.0)
