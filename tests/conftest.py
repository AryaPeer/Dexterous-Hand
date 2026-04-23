import gymnasium as gym
import numpy as np  # noqa: F401
import pytest

import dexterous_hand.envs  # noqa: F401


@pytest.fixture(scope="session")
def grasp_env():
    env = gym.make("ShadowHandGrasp-v0", render_mode="rgb_array")
    yield env
    env.close()


@pytest.fixture(scope="session")
def reorient_env():
    env = gym.make("ShadowHandReorient-v0", render_mode="rgb_array")
    yield env
    env.close()


@pytest.fixture(scope="session")
def peg_env():
    env = gym.make("ShadowHandPeg-v0", render_mode="rgb_array")
    yield env
    env.close()


@pytest.fixture
def fresh_peg_env():
    env = gym.make("ShadowHandPeg-v0")
    yield env
    env.close()


@pytest.fixture
def fresh_reorient_env():
    env = gym.make("ShadowHandReorient-v0")
    yield env
    env.close()
