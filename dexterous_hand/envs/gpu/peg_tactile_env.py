
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from dexterous_hand.config import (
    MjxTactileTrainConfig,
    PegRewardConfig,
    PegSceneConfig,
    TactileConfig,
)
from dexterous_hand.envs.gpu.peg_env import PegEnvState, ShadowHandPegMjxEnv
from dexterous_hand.utils.gpu.mjx_helpers import get_finger_touch_from_sensors

class PegTactileEnvState(NamedTuple):
    peg: PegEnvState
    previous_tactile: jnp.ndarray

class ShadowHandPegTactileMjxEnv(ShadowHandPegMjxEnv):

    def __init__(
        self,
        num_envs: int = 2048,
        seed: int = 42,
        scene_config: PegSceneConfig | None = None,
        reward_config: PegRewardConfig | None = None,
        tactile_config: TactileConfig | None = None,
        max_episode_steps: int = 500,
    ) -> None:
        self.tactile_config = tactile_config or TactileConfig()
        self._n_taxels = self.tactile_config.n_fingers * self.tactile_config.grid_size**2

        super().__init__(
            num_envs=num_envs,
            seed=seed,
            scene_config=scene_config,
            reward_config=reward_config,
            max_episode_steps=max_episode_steps,
        )

    def _obs_size(self) -> int:
        return 131 + self._n_taxels * 3                                              

    def _current_tactile(self, mjx_data: Any) -> jax.Array:
        touch_vals, _ = get_finger_touch_from_sensors(mjx_data.sensordata, self._finger_touch_adr)
        gs2 = self.tactile_config.grid_size**2
        current = jnp.repeat(touch_vals / gs2, gs2)
        return jnp.clip(current, 0.0, self.tactile_config.max_force)

    def _reset_single(  # type: ignore[override]
        self, mjx_model: Any, mjx_data: Any, key: jax.Array
    ) -> tuple[Any, PegTactileEnvState]:
        mjx_data, peg_state = super()._reset_single(mjx_model, mjx_data, key)
        tactile_state = PegTactileEnvState(
            peg=peg_state,
            previous_tactile=self._current_tactile(mjx_data),
        )
        return mjx_data, tactile_state

    def _step_single(  # type: ignore[override]
        self,
        mjx_model: Any,
        mjx_data: Any,
        env_state: PegTactileEnvState,
        action: jax.Array,
    ) -> tuple[Any, PegTactileEnvState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        mjx_data, new_peg_state, _, reward, done, info = super()._step_single(
            mjx_model, mjx_data, env_state.peg, action
        )
        new_tactile_state = PegTactileEnvState(
            peg=new_peg_state,
            previous_tactile=self._current_tactile(mjx_data),
        )
        obs = self._get_obs_single(mjx_model, mjx_data, new_tactile_state)
        return mjx_data, new_tactile_state, obs, reward, done, info

    def _get_obs_single(  # type: ignore[override]
        self, mjx_model: Any, mjx_data: Any, env_state: Any
    ) -> jax.Array:
                                                                                
        if hasattr(env_state, "peg"):
            peg_state = env_state.peg
            previous = env_state.previous_tactile
        else:
            peg_state = env_state
            previous = jnp.zeros(self._n_taxels)

        base_obs = super()._get_obs_single(mjx_model, mjx_data, peg_state)

        current = self._current_tactile(mjx_data)
        change = current - previous

        return jnp.concatenate([base_obs, current, previous, change])

    @classmethod
    def from_config(  # type: ignore[override]
        cls, config: MjxTactileTrainConfig
    ) -> ShadowHandPegTactileMjxEnv:
        return cls(
            num_envs=config.num_envs,
            seed=config.seed,
            scene_config=config.scene_config,
            reward_config=config.reward_config,
            tactile_config=config.tactile_config,
            max_episode_steps=config.max_episode_steps,
        )
