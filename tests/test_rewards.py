import numpy as np
from numpy.testing import assert_allclose

from dexterous_hand.config import PegRewardConfig, ReorientRewardConfig, RewardConfig
from dexterous_hand.rewards.grasp_reward import GraspRewardCalculator
from dexterous_hand.rewards.peg_reward import PegRewardCalculator
from dexterous_hand.rewards.reorient_reward import ReorientRewardCalculator


# --- helpers ---
def _fingertips_at(pos: np.ndarray) -> np.ndarray:
    """Return a (5, 3) array with all fingertips at the same position."""
    return np.tile(pos, (5, 1))


ZERO_ACTIONS = np.zeros(20)
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
ZERO3 = np.zeros(3)


# --- grasp reward tests ---
class TestGraspReward:
    def make_calc(self) -> GraspRewardCalculator:
        return GraspRewardCalculator(RewardConfig(), table_height=0.4)

    # -- key presence -------------------------------------------------------
    def test_all_keys_present(self):
        calc = self.make_calc()
        _, info = calc.compute(
            fingertip_positions=_fingertips_at(np.array([0.0, 0.0, 0.5])),
            object_position=np.array([0.0, 0.0, 0.5]),
            object_linear_velocity=ZERO3,
            num_fingers_in_contact=0,
            actions=ZERO_ACTIONS,
            previous_actions=ZERO_ACTIONS,
        )

        expected_keys = {
            "reward/reaching",
            "reward/grasping",
            "reward/lifting",
            "reward/holding",
            "reward/drop",
            "reward/action_penalty",
            "reward/action_rate_penalty",
            "reward/total",
            "metrics/num_finger_contacts",
            "metrics/object_height",
            "metrics/object_speed",
            "metrics/mean_fingertip_dist",
        }
        assert expected_keys == set(info.keys())

    # -- reaching -----------------------------------------------------------
    def test_reaching_decreases_with_distance(self):
        calc = self.make_calc()
        obj = np.array([0.0, 0.0, 0.5])
        _, info_close = calc.compute(
            _fingertips_at(obj + 0.01),
            obj,
            ZERO3,
            0,
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        _, info_far = calc.compute(
            _fingertips_at(obj + 0.2),
            obj,
            ZERO3,
            0,
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        assert info_close["reward/reaching"] > info_far["reward/reaching"]

    # -- grasping -----------------------------------------------------------
    def test_grasping_proportional_to_contacts(self):
        calc = self.make_calc()
        obj = np.array([0.0, 0.0, 0.5])
        _, info5 = calc.compute(
            _fingertips_at(obj),
            obj,
            ZERO3,
            5,
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        _, info0 = calc.compute(
            _fingertips_at(obj),
            obj,
            ZERO3,
            0,
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        assert_allclose(info5["reward/grasping"], 1.0)
        assert_allclose(info0["reward/grasping"], 0.0)

    # -- lifting ------------------------------------------------------------
    def test_lifting_scales_with_height(self):
        calc = self.make_calc()
        obj_at_table = np.array([0.0, 0.0, 0.4])  # table_height
        obj_lifted = np.array([0.0, 0.0, 0.5])  # table + lift_target
        _, info_low = calc.compute(
            _fingertips_at(obj_at_table),
            obj_at_table,
            ZERO3,
            0,
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        _, info_high = calc.compute(
            _fingertips_at(obj_lifted),
            obj_lifted,
            ZERO3,
            0,
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        assert_allclose(info_low["reward/lifting"], 0.0)
        assert_allclose(info_high["reward/lifting"], 1.0)

    # -- holding ------------------------------------------------------------
    def test_holding_requires_height_and_stability(self):
        calc = self.make_calc()
        # Above target height and slow
        obj_high = np.array([0.0, 0.0, 0.51])
        _, info_hold = calc.compute(
            _fingertips_at(obj_high),
            obj_high,
            np.array([0.0, 0.0, 0.01]),
            5,
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        assert_allclose(info_hold["reward/holding"], 1.0)

        # Above target but fast
        _, info_fast = calc.compute(
            _fingertips_at(obj_high),
            obj_high,
            np.array([1.0, 0.0, 0.0]),
            5,
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        assert_allclose(info_fast["reward/holding"], 0.0)

        # Below target and slow
        obj_low = np.array([0.0, 0.0, 0.45])
        _, info_low = calc.compute(
            _fingertips_at(obj_low),
            obj_low,
            ZERO3,
            5,
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        assert_allclose(info_low["reward/holding"], 0.0)

    # -- drop penalty -------------------------------------------------------
    def test_drop_penalty_triggers(self):
        calc = self.make_calc()
        obj_high = np.array([0.0, 0.0, 0.51])  # >= table + lift_target

        # Lift it first
        calc.compute(
            _fingertips_at(obj_high),
            obj_high,
            ZERO3,
            5,
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )

        # Now drop it below table + 1cm
        obj_dropped = np.array([0.0, 0.0, 0.405])  # 0.005 above table < 0.01
        _, info = calc.compute(
            _fingertips_at(obj_dropped),
            obj_dropped,
            ZERO3,
            0,
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        assert info["reward/drop"] == -10.0

    def test_drop_penalty_not_triggered_without_lift(self):
        calc = self.make_calc()
        obj_low = np.array([0.0, 0.0, 0.4])
        _, info = calc.compute(
            _fingertips_at(obj_low),
            obj_low,
            ZERO3,
            0,
            ZERO_ACTIONS,
            ZERO_ACTIONS,
        )
        assert_allclose(info["reward/drop"], 0.0)

    # -- action penalties ---------------------------------------------------
    def test_action_penalty_negative(self):
        calc = self.make_calc()
        actions = np.ones(20)
        obj = np.array([0.0, 0.0, 0.5])

        _, info = calc.compute(
            _fingertips_at(obj),
            obj,
            ZERO3,
            0,
            actions,
            ZERO_ACTIONS,
        )
        assert info["reward/action_penalty"] < 0.0
        assert_allclose(info["reward/action_penalty"], -0.01 * 20.0)

    def test_action_rate_zero_for_same_actions(self):
        calc = self.make_calc()
        actions = np.ones(20) * 0.5
        obj = np.array([0.0, 0.0, 0.5])

        _, info = calc.compute(
            _fingertips_at(obj),
            obj,
            ZERO3,
            0,
            actions,
            actions,
        )
        assert_allclose(info["reward/action_rate_penalty"], 0.0)

    # -- finiteness ---------------------------------------------------------
    def test_reward_is_finite(self):
        calc = self.make_calc()
        rng = np.random.default_rng(42)
        for _ in range(20):
            total, info = calc.compute(
                fingertip_positions=rng.uniform(-1, 1, (5, 3)),
                object_position=rng.uniform(-1, 1, 3),
                object_linear_velocity=rng.uniform(-5, 5, 3),
                num_fingers_in_contact=rng.integers(0, 6),
                actions=rng.uniform(-1, 1, 20),
                previous_actions=rng.uniform(-1, 1, 20),
            )
            assert np.isfinite(total)
            for v in info.values():
                assert np.isfinite(v)


# --- reorient reward tests ---
class TestReorientReward:
    def make_calc(self) -> ReorientRewardCalculator:
        return ReorientRewardCalculator(
            ReorientRewardConfig(),
            np.array([0.0, 0.0, 0.5]),
        )

    # -- key presence -------------------------------------------------------
    def test_all_keys_present(self):
        calc = self.make_calc()
        _, info, _ = calc.compute(
            cube_quat=IDENTITY_QUAT,
            target_quat=IDENTITY_QUAT,
            cube_pos=np.array([0.0, 0.0, 0.5]),
            cube_linvel=ZERO3,
            cube_angvel=ZERO3,
            fingertip_positions=_fingertips_at(np.array([0.0, 0.0, 0.5])),
            actions=ZERO_ACTIONS,
            previous_actions=ZERO_ACTIONS,
            dropped=False,
        )

        expected_keys = {
            "reward/orientation_tracking",
            "reward/orientation_success",
            "reward/cube_drop",
            "reward/velocity_penalty",
            "reward/fingertip_distance",
            "reward/position_penalty",
            "reward/action_penalty",
            "reward/action_rate_penalty",
            "reward/total",
            "metrics/angular_distance",
            "metrics/mean_fingertip_dist",
            "metrics/cube_displacement",
            "metrics/success_steps",
        }
        assert expected_keys == set(info.keys())

    # -- orientation tracking -----------------------------------------------
    def test_perfect_orientation_max_tracking(self):
        calc = self.make_calc()
        _, info, _ = calc.compute(
            cube_quat=IDENTITY_QUAT,
            target_quat=IDENTITY_QUAT,
            cube_pos=np.array([0.0, 0.0, 0.5]),
            cube_linvel=ZERO3,
            cube_angvel=ZERO3,
            fingertip_positions=_fingertips_at(np.array([0.0, 0.0, 0.5])),
            actions=ZERO_ACTIONS,
            previous_actions=ZERO_ACTIONS,
            dropped=False,
        )
        # exp(-5.0 * 0) = 1.0
        assert_allclose(info["reward/orientation_tracking"], 1.0, rtol=1e-6)

    def test_90_degree_error_lower_tracking(self):
        calc = self.make_calc()
        from dexterous_hand.utils.quaternion import quat_from_axis_angle

        q90 = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        _, info, _ = calc.compute(
            cube_quat=IDENTITY_QUAT,
            target_quat=q90,
            cube_pos=np.array([0.0, 0.0, 0.5]),
            cube_linvel=ZERO3,
            cube_angvel=ZERO3,
            fingertip_positions=_fingertips_at(np.array([0.0, 0.0, 0.5])),
            actions=ZERO_ACTIONS,
            previous_actions=ZERO_ACTIONS,
            dropped=False,
        )
        # exp(-5 * pi/2) is much less than 1
        expected = float(np.exp(-5.0 * np.pi / 2))
        assert_allclose(info["reward/orientation_tracking"], expected, rtol=1e-5)

    # -- success detection --------------------------------------------------
    def test_success_detection_after_hold_steps(self):
        calc = self.make_calc()
        cube_pos = np.array([0.0, 0.0, 0.5])
        target_reached = False
        for _ in range(25):
            _, _, target_reached = calc.compute(
                cube_quat=IDENTITY_QUAT,
                target_quat=IDENTITY_QUAT,
                cube_pos=cube_pos,
                cube_linvel=ZERO3,
                cube_angvel=ZERO3,
                fingertip_positions=_fingertips_at(cube_pos),
                actions=ZERO_ACTIONS,
                previous_actions=ZERO_ACTIONS,
                dropped=False,
            )
        assert target_reached is True

    def test_success_resets_on_failure(self):
        calc = self.make_calc()
        cube_pos = np.array([0.0, 0.0, 0.5])
        from dexterous_hand.utils.quaternion import quat_from_axis_angle
        far_quat = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi / 2)

        # Accumulate 20 steps of success
        for _ in range(20):
            calc.compute(
                IDENTITY_QUAT,
                IDENTITY_QUAT,
                cube_pos,
                ZERO3,
                ZERO3,
                _fingertips_at(cube_pos),
                ZERO_ACTIONS,
                ZERO_ACTIONS,
                False,
            )

        # Break the streak
        calc.compute(
            IDENTITY_QUAT,
            far_quat,
            cube_pos,
            ZERO3,
            ZERO3,
            _fingertips_at(cube_pos),
            ZERO_ACTIONS,
            ZERO_ACTIONS,
            False,
        )

        # Now 5 more successes (total should be 5, not 26)
        target_reached = False
        for _ in range(5):
            _, _, target_reached = calc.compute(
                IDENTITY_QUAT,
                IDENTITY_QUAT,
                cube_pos,
                ZERO3,
                ZERO3,
                _fingertips_at(cube_pos),
                ZERO_ACTIONS,
                ZERO_ACTIONS,
                False,
            )
        assert target_reached is False

    # -- drop penalty -------------------------------------------------------
    def test_drop_penalty(self):
        calc = self.make_calc()
        cube_pos = np.array([0.0, 0.0, 0.5])
        _, info, _ = calc.compute(
            IDENTITY_QUAT,
            IDENTITY_QUAT,
            cube_pos,
            ZERO3,
            ZERO3,
            _fingertips_at(cube_pos),
            ZERO_ACTIONS,
            ZERO_ACTIONS,
            dropped=True,
        )
        assert_allclose(info["reward/cube_drop"], -20.0)

    # -- position penalty ---------------------------------------------------
    def test_position_penalty_increases_with_displacement(self):
        calc = self.make_calc()
        cube_near = np.array([0.0, 0.0, 0.5])  # at initial pos
        cube_far = np.array([0.1, 0.1, 0.5])  # displaced
        _, info_near, _ = calc.compute(
            IDENTITY_QUAT,
            IDENTITY_QUAT,
            cube_near,
            ZERO3,
            ZERO3,
            _fingertips_at(cube_near),
            ZERO_ACTIONS,
            ZERO_ACTIONS,
            False,
        )
        _, info_far, _ = calc.compute(
            IDENTITY_QUAT,
            IDENTITY_QUAT,
            cube_far,
            ZERO3,
            ZERO3,
            _fingertips_at(cube_far),
            ZERO_ACTIONS,
            ZERO_ACTIONS,
            False,
        )
        # More displacement => more negative penalty
        assert info_far["reward/position_penalty"] < info_near["reward/position_penalty"]
        assert_allclose(info_near["reward/position_penalty"], 0.0, atol=1e-10)

    # -- velocity penalty ---------------------------------------------------
    def test_velocity_penalty_negative_for_moving_cube(self):
        calc = self.make_calc()
        cube_pos = np.array([0.0, 0.0, 0.5])
        _, info, _ = calc.compute(
            IDENTITY_QUAT,
            IDENTITY_QUAT,
            cube_pos,
            cube_linvel=np.array([1.0, 0.0, 0.0]),
            cube_angvel=np.array([0.0, 2.0, 0.0]),
            fingertip_positions=_fingertips_at(cube_pos),
            actions=ZERO_ACTIONS,
            previous_actions=ZERO_ACTIONS,
            dropped=False,
        )
        assert info["reward/velocity_penalty"] < 0.0


# --- peg reward tests ---
class TestPegReward:
    TABLE_HEIGHT = 0.4

    def make_calc(self) -> PegRewardCalculator:
        return PegRewardCalculator(PegRewardConfig(), table_height=self.TABLE_HEIGHT)

    def _default_kwargs(self, **overrides) -> dict:
        defaults = dict(
            stage=0,
            fingertip_positions=_fingertips_at(np.array([0.0, 0.0, 0.45])),
            peg_position=np.array([0.0, 0.0, 0.45]),
            peg_axis=np.array([0.0, 0.0, 1.0]),
            hole_position=np.array([0.1, 0.0, 0.45]),
            hole_axis=np.array([0.0, 0.0, 1.0]),
            insertion_depth=0.0,
            contact_force_magnitude=0.0,
            num_fingers_in_contact=0,
            peg_height=0.45,
            actions=ZERO_ACTIONS,
            previous_actions=ZERO_ACTIONS,
        )
        defaults.update(overrides)
        return defaults

    # -- key presence -------------------------------------------------------
    def test_all_keys_present(self):
        calc = self.make_calc()
        _, info = calc.compute(**self._default_kwargs())

        expected_keys = {
            "reward/reach",
            "reward/grasp",
            "reward/lift",
            "reward/align",
            "reward/depth",
            "reward/complete",
            "reward/force_penalty",
            "reward/drop",
            "reward/smoothness",
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

    # -- reach gating -------------------------------------------------------
    def test_reach_only_in_stage_0(self):
        calc = self.make_calc()
        _, info0 = calc.compute(**self._default_kwargs(stage=0))
        _, info1 = calc.compute(**self._default_kwargs(stage=1))
        assert info0["reward/reach"] > 0.0
        assert_allclose(info1["reward/reach"], 0.0)

    # -- grasp proportional -------------------------------------------------
    def test_grasp_proportional(self):
        calc = self.make_calc()
        _, info3 = calc.compute(**self._default_kwargs(num_fingers_in_contact=3))
        _, info1 = calc.compute(**self._default_kwargs(num_fingers_in_contact=1))
        assert_allclose(info3["reward/grasp"], 1.0)
        assert_allclose(info1["reward/grasp"], 1.0 / 3.0, rtol=1e-6)

    # -- lift ---------------------------------------------------------------
    def test_lift_scales_with_height(self):
        calc = self.make_calc()
        _, info_low = calc.compute(**self._default_kwargs(peg_height=0.4))
        _, info_high = calc.compute(**self._default_kwargs(peg_height=0.5))
        assert_allclose(info_low["reward/lift"], 0.0)
        assert_allclose(info_high["reward/lift"], 1.0)

    # -- insertion depth ----------------------------------------------------
    def test_insertion_depth_reward(self):
        calc = self.make_calc()
        peg_length = PegRewardConfig().peg_half_length * 2.0  # 0.06
        hole_pos = np.array([0.0, 0.0, 0.45])
        peg_pos = np.array([0.0, 0.0, 0.45])  # lateral_dist = 0
        _, info = calc.compute(
            **self._default_kwargs(
                peg_position=peg_pos,
                hole_position=hole_pos,
                insertion_depth=peg_length * 0.5,
            )
        )
        expected_depth = 10.0 * 0.5
        assert_allclose(info["reward/depth"], expected_depth, rtol=1e-6)

    def test_insertion_depth_gated_by_lateral(self):
        calc = self.make_calc()
        peg_length = PegRewardConfig().peg_half_length * 2.0
        # Lateral distance >= 0.005
        peg_pos = np.array([0.0, 0.0, 0.45])
        hole_pos = np.array([0.01, 0.0, 0.45])  # 10mm lateral
        _, info = calc.compute(
            **self._default_kwargs(
                peg_position=peg_pos,
                hole_position=hole_pos,
                insertion_depth=peg_length * 0.5,
            )
        )
        assert_allclose(info["reward/depth"], 0.0)

    # -- completion bonus ---------------------------------------------------
    def test_completion_bonus_after_hold(self):
        calc = self.make_calc()
        peg_length = PegRewardConfig().peg_half_length * 2.0
        kwargs = self._default_kwargs(
            insertion_depth=peg_length * 0.95,  # > 90%
            peg_position=np.array([0.0, 0.0, 0.45]),
            hole_position=np.array([0.0, 0.0, 0.45]),
        )
        for _ in range(10):
            _, info = calc.compute(**kwargs)
        assert_allclose(info["reward/complete"], 50.0)

    # -- force penalty ------------------------------------------------------
    def test_force_penalty_above_threshold(self):
        calc = self.make_calc()
        _, info = calc.compute(**self._default_kwargs(contact_force_magnitude=10.0))
        # excess = 10 - 5 = 5, penalty = -0.01 * 25 = -0.25
        assert_allclose(info["reward/force_penalty"], -0.25, rtol=1e-6)

    def test_force_penalty_below_threshold(self):
        calc = self.make_calc()
        _, info = calc.compute(**self._default_kwargs(contact_force_magnitude=3.0))
        assert_allclose(info["reward/force_penalty"], 0.0)

    # -- drop penalty -------------------------------------------------------
    def test_drop_penalty(self):
        calc = self.make_calc()

        # First, get it grasped (3+ contacts)
        calc.compute(**self._default_kwargs(num_fingers_in_contact=3))

        # Then drop it below table - 0.02
        _, info = calc.compute(
            **self._default_kwargs(
                peg_height=self.TABLE_HEIGHT - 0.03,
                num_fingers_in_contact=0,
            )
        )
        assert_allclose(info["reward/drop"], -10.0)

    # -- smoothness ---------------------------------------------------------
    def test_smoothness_zero_for_constant_actions(self):
        calc = self.make_calc()
        actions = np.ones(20) * 0.3

        _, info = calc.compute(
            **self._default_kwargs(
                actions=actions,
                previous_actions=actions,
            )
        )
        assert_allclose(info["reward/smoothness"], 0.0)
