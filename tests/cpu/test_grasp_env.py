import mujoco
import numpy as np
import pytest

import dexterous_hand.envs  # noqa: F401

@pytest.mark.slow
class TestGraspEnvSpaces:
    def test_observation_shape(self, grasp_env):
        assert grasp_env.observation_space.shape == (105,)

    def test_action_shape(self, grasp_env):
        assert grasp_env.action_space.shape == (22,)

    def test_action_bounds(self, grasp_env):
        assert float(grasp_env.action_space.low.min()) == -1.0
        assert float(grasp_env.action_space.high.max()) == 1.0

@pytest.mark.slow
class TestGraspEnvReset:
    def test_reset_obs_shape(self, grasp_env):
        obs, info = grasp_env.reset(seed=42)
        assert obs.shape == (105,)

    def test_reset_obs_finite(self, grasp_env):
        obs, _ = grasp_env.reset(seed=42)
        assert np.all(np.isfinite(obs))

    def test_deterministic_seeding(self, grasp_env):
        obs1, _ = grasp_env.reset(seed=123)
        obs2, _ = grasp_env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

@pytest.mark.slow
class TestGraspEnvStep:
    def test_step_returns(self, grasp_env):
        grasp_env.reset(seed=42)
        action = grasp_env.action_space.sample()
        obs, reward, terminated, truncated, info = grasp_env.step(action)
        assert obs.shape == (105,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_obs_finite_after_many_steps(self, grasp_env):
        grasp_env.reset(seed=42)
        for i in range(200):
            action = grasp_env.action_space.sample()
            obs, _, terminated, truncated, _ = grasp_env.step(action)
            assert np.all(np.isfinite(obs)), f"Non-finite at step {i}"
            if terminated or truncated:
                grasp_env.reset()

    def test_reward_info_keys(self, grasp_env):
        grasp_env.reset(seed=42)
        _, _, _, _, info = grasp_env.step(grasp_env.action_space.sample())
        expected = [
            "reward/reaching",
            "reward/grasping",
            "reward/lifting",
            "reward/holding",
            "reward/drop",
            "reward/success",
            "reward/action_penalty",
            "reward/total",
        ]
        for key in expected:
            assert key in info, f"Missing key: {key}"

    def test_metric_keys(self, grasp_env):
        grasp_env.reset(seed=42)
        _, _, _, _, info = grasp_env.step(grasp_env.action_space.sample())
        for key in [
            "metrics/num_finger_contacts",
            "metrics/object_height",
            "metrics/object_speed",
        ]:
            assert key in info

    def test_reward_reasonable(self, grasp_env):
        grasp_env.reset(seed=42)
        for _ in range(50):
            _, reward, term, trunc, _ = grasp_env.step(grasp_env.action_space.sample())
            assert np.isfinite(reward)
            assert abs(reward) < 1000
            if term or trunc:
                grasp_env.reset()

    def test_finger_contact_metric_detects_overlap(self, grasp_env):
        from dexterous_hand.utils.cpu.mujoco_helpers import get_finger_contacts

        grasp_env.reset(seed=0)
        env = grasp_env.unwrapped

        target_geom = next(iter(env.nm.finger_geom_ids_per_finger[0]))
        overlap_pos = env.data.geom_xpos[target_geom].copy()

        s = env.nm.obj_qpos_start
        env.data.qpos[s : s + 3] = overlap_pos
        env.data.qpos[s + 3 : s + 7] = [1.0, 0.0, 0.0, 0.0]
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)

        num_contacts, _ = get_finger_contacts(
            env.model,
            env.data,
            env.nm.finger_geom_ids_per_finger,
            env.nm.object_geom_id,
        )
        assert num_contacts >= 1, (
            "Expected at least one finger contact after placing object at finger"
        )

@pytest.mark.slow
class TestGraspEnvRender:
    def test_rgb_render(self, grasp_env):
        grasp_env.reset(seed=0)
        frame = grasp_env.render()
        assert frame is not None
        assert frame.ndim == 3
        assert frame.shape[2] == 3       

@pytest.mark.slow
class TestGraspEnvMultipleResets:
    def test_100_resets(self, grasp_env):
        for i in range(100):
            obs, _ = grasp_env.reset(seed=i)
            assert np.all(np.isfinite(obs))
