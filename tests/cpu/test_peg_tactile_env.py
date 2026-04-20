import numpy as np
import pytest

import dexterous_hand.envs  # noqa: F401

@pytest.mark.slow
class TestPegTactileEnvSpaces:
    def test_observation_shape(self, peg_tactile_env):
        assert peg_tactile_env.observation_space.shape == (371,)

    def test_action_shape(self, peg_tactile_env):
        assert peg_tactile_env.action_space.shape == (22,)

    def test_action_bounds(self, peg_tactile_env):
        assert float(peg_tactile_env.action_space.low.min()) == -1.0
        assert float(peg_tactile_env.action_space.high.max()) == 1.0

@pytest.mark.slow
class TestPegTactileEnvReset:
    def test_reset_obs_shape(self, peg_tactile_env):
        obs, _ = peg_tactile_env.reset(seed=42)
        assert obs.shape == (371,)

    def test_reset_obs_finite(self, peg_tactile_env):
        obs, _ = peg_tactile_env.reset(seed=42)
        assert np.all(np.isfinite(obs))

    def test_deterministic_seeding(self, peg_tactile_env):
        obs1, _ = peg_tactile_env.reset(seed=123)
        obs2, _ = peg_tactile_env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

@pytest.mark.slow
class TestPegTactileEnvStep:
    def test_step_returns(self, peg_tactile_env):
        peg_tactile_env.reset(seed=42)
        action = peg_tactile_env.action_space.sample()
        obs, reward, terminated, truncated, info = peg_tactile_env.step(action)
        assert obs.shape == (371,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_obs_finite_after_many_steps(self, peg_tactile_env):
        peg_tactile_env.reset(seed=42)
        for i in range(200):
            action = peg_tactile_env.action_space.sample()
            obs, _, terminated, truncated, _ = peg_tactile_env.step(action)
            assert np.all(np.isfinite(obs)), f"Non-finite at step {i}"
            if terminated or truncated:
                peg_tactile_env.reset()

    def test_reward_info_keys(self, peg_tactile_env):
        peg_tactile_env.reset(seed=42)
        _, _, _, _, info = peg_tactile_env.step(peg_tactile_env.action_space.sample())
        expected = [
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
        ]
        for key in expected:
            assert key in info, f"Missing key: {key}"

    def test_reward_reasonable(self, peg_tactile_env):
        peg_tactile_env.reset(seed=42)
        for _ in range(50):
            _, reward, term, trunc, _ = peg_tactile_env.step(peg_tactile_env.action_space.sample())
            assert np.isfinite(reward)
            assert abs(reward) < 1000
            if term or trunc:
                peg_tactile_env.reset()

@pytest.mark.slow
class TestPegTactileObs:
    def test_tactile_slice_shape(self, peg_tactile_env):
        obs, _ = peg_tactile_env.reset(seed=42)
        proprio = obs[:131]                  
        tactile_current = obs[131:211]              
        tactile_prev = obs[211:291]               
        tactile_change = obs[291:371]             

        assert proprio.shape == (131,)
        assert tactile_current.shape == (80,)
        assert tactile_prev.shape == (80,)
        assert tactile_change.shape == (80,)

    def test_tactile_nonnegative(self, peg_tactile_env):
        peg_tactile_env.reset(seed=42)
        for _ in range(10):
            obs, _, term, trunc, _ = peg_tactile_env.step(peg_tactile_env.action_space.sample())
            tactile_current = obs[131:211]
            assert np.all(tactile_current >= 0)
            if term or trunc:
                peg_tactile_env.reset()

@pytest.mark.slow
class TestPegTactileCurriculum:
    def test_set_curriculum_params(self, fresh_peg_tactile_env):
        fresh_peg_tactile_env.unwrapped.set_curriculum_params(0.002, False)
        obs, _ = fresh_peg_tactile_env.reset()
        assert np.all(np.isfinite(obs))

@pytest.mark.slow
class TestPegTactileRender:
    def test_rgb_render(self, peg_tactile_env):
        peg_tactile_env.reset(seed=0)
        frame = peg_tactile_env.render()
        assert frame is not None
        assert frame.ndim == 3

@pytest.mark.slow
class TestPegTactileMultipleResets:
    def test_50_resets(self, peg_tactile_env):
        for i in range(50):
            obs, _ = peg_tactile_env.reset(seed=i)
            assert np.all(np.isfinite(obs))
