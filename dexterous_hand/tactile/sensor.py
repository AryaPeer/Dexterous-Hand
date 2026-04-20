import mujoco
import numpy as np

from dexterous_hand.config import TactileConfig
from dexterous_hand.envs.scene_builder import FINGERTIP_BODIES, FINGERTIP_OFFSETS
from dexterous_hand.utils.cpu.mujoco_helpers import distribute_contact_forces_to_taxels

class TactileSensor:

    def __init__(
        self,
        model: mujoco.MjModel,
        config: TactileConfig,
        rng: np.random.Generator,
    ) -> None:

        self.config = config
        self.rng = rng
        self.n_fingers = config.n_fingers
        self.grid_size = config.grid_size
        self.n_taxels = self.n_fingers * self.grid_size**2

        self._fingertip_body_ids: list[int] = []
        self._fingertip_geom_ids_per_finger: list[set[int]] = []
        self._taxel_local_positions: list[np.ndarray] = []
        self._previous_readings = np.zeros(self.n_taxels)

        self._setup(model)

    def _setup(self, model: mujoco.MjModel) -> None:

        half_span = (self.grid_size - 1) * self.config.grid_spacing / 2
        offsets = np.linspace(-half_span, half_span, self.grid_size)

        for body_name in FINGERTIP_BODIES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            self._fingertip_body_ids.append(bid)

            geom_ids: set[int] = set()
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] == bid:
                    geom_ids.add(gid)
            self._fingertip_geom_ids_per_finger.append(geom_ids)

            pad_z = FINGERTIP_OFFSETS[body_name][2]

            gx, gy = np.meshgrid(offsets, offsets, indexing="ij")
            positions = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, pad_z)])
            self._taxel_local_positions.append(positions)

    def get_taxel_world_positions(self, data: mujoco.MjData) -> list[np.ndarray]:

        result: list[np.ndarray] = []
        for i, bid in enumerate(self._fingertip_body_ids):
            body_pos = data.xpos[bid]
            body_rot = data.xmat[bid].reshape(3, 3)
            local = self._taxel_local_positions[i]
            world = (body_rot @ local.T).T + body_pos
            result.append(world)
        return result

    def get_readings(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        world_positions = self.get_taxel_world_positions(data)

        readings_grid = distribute_contact_forces_to_taxels(
            model,
            data,
            self._fingertip_geom_ids_per_finger,
            world_positions,
            self.config.max_force,
            self.config.noise_std,
            self.rng,
        )

        current = readings_grid.flatten()
        previous = self._previous_readings.copy()
        change = current - previous
        self._previous_readings = current.copy()
        return current, previous, change

    def reset(self) -> None:

        self._previous_readings = np.zeros(self.n_taxels)
