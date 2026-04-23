
from __future__ import annotations

from typing import Any

from gymnasium import spaces
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

class MjxVecEnv(VecEnv):

    def __init__(
        self,
        num_envs: int,
        seed: int = 42,
        obs_noise_std: float = 0.0,
    ) -> None:
        self._num_envs = num_envs
        self._seed = seed
        self._obs_noise_std = float(obs_noise_std)

        self._cpu_model = self._build_model()
        self._cpu_data = mujoco.MjData(self._cpu_model)
        self._mjx_model = mjx.put_model(self._cpu_model)

        n_obs = self._obs_size()
        n_act = self._action_size()

                                                                               
                                                                      
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)

        super().__init__(num_envs, observation_space, action_space)

        self._master_key = jax.random.PRNGKey(seed)
        noise_key, self._master_key = jax.random.split(self._master_key)
        self._obs_noise_key = noise_key
        self._env_keys = jax.random.split(self._master_key, num_envs)

                                          
        self._ctrl_low = jnp.array(self._cpu_model.actuator_ctrlrange[:n_act, 0])
        self._ctrl_high = jnp.array(self._cpu_model.actuator_ctrlrange[:n_act, 1])

                                   
        self._batched_reset = jax.jit(jax.vmap(self._reset_single, in_axes=(None, 0, 0)))
        self._batched_step = jax.jit(jax.vmap(self._step_single, in_axes=(None, 0, 0, 0)))
        self._batched_get_obs = jax.jit(jax.vmap(self._get_obs_single, in_axes=(None, 0, 0)))

                       
        self._mjx_data_batch = None
        self._env_state_batch = None
        self._step_count = jnp.zeros(num_envs, dtype=jnp.int32)

                                    
        self._pending_obs: np.ndarray | None = None
        self._pending_rewards: np.ndarray | None = None
        self._pending_dones: np.ndarray | None = None
        self._pending_infos: list[dict] | None = None

                                                                   

    def _build_model(self) -> mujoco.MjModel:
        raise NotImplementedError

    def _reset_single(self, mjx_model: Any, mjx_data: Any, key: jax.Array) -> Any:
        raise NotImplementedError

    def _step_single(self, mjx_model: Any, mjx_data: Any, env_state: Any, action: jax.Array) -> Any:
        raise NotImplementedError

    def _get_obs_single(self, mjx_model: Any, mjx_data: Any, env_state: Any) -> jax.Array:
        raise NotImplementedError

    def _obs_size(self) -> int:
        raise NotImplementedError

    def _action_size(self) -> int:
        raise NotImplementedError

    @property
    def _max_episode_steps(self) -> int:
        raise NotImplementedError

                                                                   

    def _noisy_obs(self, obs: jax.Array) -> np.ndarray:
        # additive Gaussian noise on policy-facing observations (additive DR,
        # per AUDIT C1). pure no-op when obs_noise_std == 0.0. applied in
        # jax on-device so the obs array stays device-resident until the final
        # np.asarray conversion.
        if self._obs_noise_std <= 0.0:
            return np.asarray(obs)
        self._obs_noise_key, subkey = jax.random.split(self._obs_noise_key)
        noise = jax.random.normal(subkey, obs.shape) * self._obs_noise_std
        return np.asarray(obs + noise)

    def reset(self) -> VecEnvObs:
        base_data = mjx.make_data(self._mjx_model)
        batch_data = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (self._num_envs,) + x.shape),
            base_data,
        )

        self._env_keys = jax.random.split(jax.random.fold_in(self._master_key, 0), self._num_envs)

        batch_data, env_state = self._batched_reset(self._mjx_model, batch_data, self._env_keys)
        self._mjx_data_batch = batch_data
        self._env_state_batch = env_state
        self._step_count = jnp.zeros(self._num_envs, dtype=jnp.int32)

        obs = self._batched_get_obs(self._mjx_model, batch_data, env_state)
        return self._noisy_obs(obs)

    def step_async(self, actions: np.ndarray) -> None:
        actions_jax = jnp.array(actions, dtype=jnp.float32)

        new_data, new_state, obs, rewards, dones, reward_info = self._batched_step(
            self._mjx_model,
            self._mjx_data_batch,
            self._env_state_batch,
            actions_jax,
        )

        self._step_count = self._step_count + 1
        truncated = self._step_count >= self._max_episode_steps
        truncated_only = truncated & ~dones
        dones = dones | truncated

                              
        needs_reset = dones
        if jnp.any(needs_reset):
            self._env_keys = jax.vmap(
                lambda k, need: jax.lax.cond(
                    need, lambda k: jax.random.split(k)[0], lambda k: k, k
                ),
                in_axes=(0, 0),
            )(self._env_keys, needs_reset)

            reset_data, reset_state = self._batched_reset(self._mjx_model, new_data, self._env_keys)

                                                                             
            new_data = jax.tree.map(
                lambda r, n: jnp.where(needs_reset.reshape(-1, *([1] * (r.ndim - 1))), r, n),
                reset_data,
                new_data,
            )
            new_state = jax.tree.map(
                lambda r, n: jnp.where(
                    needs_reset.reshape(-1, *([1] * (r.ndim - 1))) if r.ndim > 0 else needs_reset,
                    r,
                    n,
                ),
                reset_state,
                new_state,
            )

            self._step_count = jnp.where(needs_reset, 0, self._step_count)

            reset_obs = self._batched_get_obs(self._mjx_model, new_data, new_state)
            obs_np = self._noisy_obs(obs)
            reset_obs_np = self._noisy_obs(reset_obs)
        else:
            obs_np = self._noisy_obs(obs)
            reset_obs_np = None

        self._mjx_data_batch = new_data
        self._env_state_batch = new_state

        dones_np = np.asarray(dones)
        truncated_np = np.asarray(truncated_only)
        rewards_np = np.asarray(rewards, dtype=np.float64)

                                                                                   
                                                                                     
        reward_info_np: dict[str, np.ndarray] = {}
        if reward_info is not None:
            for k, v in reward_info.items():
                reward_info_np[k] = np.asarray(v)

        infos: list[dict[str, Any]] = []
        for i in range(self._num_envs):
            info: dict[str, Any] = {}
            for k, v in reward_info_np.items():
                info[k] = v[i]
            if dones_np[i]:
                info["terminal_observation"] = obs_np[i].copy()
                if truncated_np[i]:
                    info["TimeLimit.truncated"] = True
                if reset_obs_np is not None:
                    obs_np[i] = reset_obs_np[i]
            infos.append(info)

        self._pending_obs = obs_np
        self._pending_rewards = rewards_np
        self._pending_dones = dones_np
        self._pending_infos = infos

    def step_wait(self) -> VecEnvStepReturn:
        assert self._pending_obs is not None
        obs = self._pending_obs
        rewards = self._pending_rewards
        dones = self._pending_dones
        infos = self._pending_infos

        self._pending_obs = None
        self._pending_rewards = None
        self._pending_dones = None
        self._pending_infos = None

        return obs, rewards, dones, infos  # type: ignore[return-value]

    def close(self) -> None:
        pass

    def env_is_wrapped(self, wrapper_class: Any, indices: Any = None) -> list[bool]:  # type: ignore[override]
        del wrapper_class, indices                                                  
        return [False] * self._num_envs

    def env_method(  # type: ignore[override]
        self,
        method_name: str,
        *method_args: Any,
        indices: Any = None,
        **method_kwargs: Any,
    ) -> list[Any]:

        if indices is None:
            indices = range(self._num_envs)

        if hasattr(self, method_name):
            fn = getattr(self, method_name)
            return [fn(*method_args, **method_kwargs)] * len(list(indices))

        return [None] * len(list(indices))

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:  # type: ignore[override]
        if hasattr(self, attr_name):
            val = getattr(self, attr_name)
            n = self._num_envs if indices is None else len(indices)
            return [val] * n
        return [None] * (self._num_envs if indices is None else len(indices))

    def set_attr(  # type: ignore[override]
        self, attr_name: str, value: Any, indices: Any = None
    ) -> None:
        setattr(self, attr_name, value)

    def seed(self, seed: int | None = None) -> None:  # type: ignore[override]
        if seed is not None:
            self._master_key = jax.random.PRNGKey(seed)

                                                                   

    def get_cpu_model_data(self) -> tuple[mujoco.MjModel, mujoco.MjData]:

        if self._mjx_data_batch is not None:
                                                   
            qpos_0 = np.asarray(jax.tree.map(lambda x: x[0], self._mjx_data_batch.qpos))
            qvel_0 = np.asarray(jax.tree.map(lambda x: x[0], self._mjx_data_batch.qvel))
            self._cpu_data.qpos[:] = qpos_0
            self._cpu_data.qvel[:] = qvel_0
            mujoco.mj_forward(self._cpu_model, self._cpu_data)
        return self._cpu_model, self._cpu_data
