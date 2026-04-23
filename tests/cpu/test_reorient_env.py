import numpy as np
import pytest

import dexterous_hand.envs  # noqa: F401

@pytest.mark.slow
class TestReorientEnvSpaces:
    def test_observation_shape(self, reorient_env):
        assert reorient_env.observation_space.shape == (109,)

    def test_action_shape(self, reorient_env):
        assert reorient_env.action_space.shape == (20,)

    def test_action_bounds(self, reorient_env):
        assert float(reorient_env.action_space.low.min()) == -1.0
        assert float(reorient_env.action_space.high.max()) == 1.0

@pytest.mark.slow
class TestReorientEnvReset:
    def test_reset_obs_shape(self, reorient_env):
        obs, _ = reorient_env.reset(seed=42)
        assert obs.shape == (109,)

    def test_reset_obs_finite(self, reorient_env):
        obs, _ = reorient_env.reset(seed=42)
        assert np.all(np.isfinite(obs))

    def test_deterministic_seeding(self, reorient_env):
        obs1, _ = reorient_env.reset(seed=123)
        obs2, _ = reorient_env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

@pytest.mark.slow
class TestReorientEnvStep:
    def test_step_returns(self, reorient_env):
        reorient_env.reset(seed=42)
        action = reorient_env.action_space.sample()
        obs, reward, terminated, truncated, info = reorient_env.step(action)
        assert obs.shape == (109,)
        assert isinstance(reward, float)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))

    def test_obs_finite_after_many_steps(self, reorient_env):
        reorient_env.reset(seed=42)
        for i in range(200):
            action = reorient_env.action_space.sample()
            obs, _, terminated, truncated, _ = reorient_env.step(action)
            assert np.all(np.isfinite(obs)), f"Non-finite at step {i}"
            if terminated or truncated:
                reorient_env.reset()

    def test_reward_info_keys(self, reorient_env):
        reorient_env.reset(seed=42)
        _, _, _, _, info = reorient_env.step(reorient_env.action_space.sample())
        expected = [
            "reward/angular_progress",
            "reward/orientation",
            "reward/cube_drop",
            "reward/action_penalty",
            "reward/action_rate_penalty",
            "reward/finger_contact_bonus",
            "reward/no_contact_penalty",
            "reward/total",
        ]
        for key in expected:
            assert key in info, f"Missing key: {key}"

    def test_metric_keys(self, reorient_env):
        reorient_env.reset(seed=42)
        _, _, _, _, info = reorient_env.step(reorient_env.action_space.sample())
        for key in [
            "metrics/angular_distance",
            "metrics/num_finger_contacts",
            "metrics/success_steps",
        ]:
            assert key in info, f"Missing key: {key}"

    def test_reward_reasonable(self, reorient_env):
        reorient_env.reset(seed=42)
        for _ in range(50):
            _, reward, term, trunc, _ = reorient_env.step(reorient_env.action_space.sample())
            assert np.isfinite(reward)
            assert abs(reward) < 1000
            if term or trunc:
                reorient_env.reset()

@pytest.mark.slow
class TestReorientCurriculum:
    def test_set_curriculum_stage(self, fresh_reorient_env):
        fresh_reorient_env.unwrapped.set_curriculum_stage(1.5)
        assert fresh_reorient_env.unwrapped._max_target_angle == 1.5
        obs, _ = fresh_reorient_env.reset()
        assert np.all(np.isfinite(obs))

@pytest.mark.slow
class TestReorientEnvRender:
    def test_rgb_render(self, reorient_env):
        reorient_env.reset(seed=0)
        frame = reorient_env.render()
        assert frame is not None
        assert frame.ndim == 3
        assert frame.shape[2] == 3       

@pytest.mark.slow
class TestReorientMultipleResets:
    def test_100_resets(self, reorient_env):
        for i in range(100):
            obs, _ = reorient_env.reset(seed=i)
            assert np.all(np.isfinite(obs))
