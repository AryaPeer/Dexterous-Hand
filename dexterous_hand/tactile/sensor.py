import mujoco
import numpy as np

from dexterous_hand.config import TactileConfig
from dexterous_hand.envs.scene_builder import FINGERTIP_BODIES, FINGERTIP_OFFSETS
from dexterous_hand.utils.mujoco_helpers import distribute_contact_forces_to_taxels


class TactileSensor:
    """Simulated tactile sensor array on the fingertips.

    Each finger gets a grid_size x grid_size grid of taxels that measure normal force.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        config: TactileConfig,
        rng: np.random.Generator,
    ) -> None:
        """Set up the taxel grid positions and map geom IDs.

        Args:
            model: MuJoCo model (for looking up body/geom IDs)
            config: sensor config (grid size, noise, etc.)
            rng: random generator for sensor noise
        """
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
        """Build the taxel grids in local coordinates and figure out which geoms belong to each finger."""
        offsets = []
        half_span = (self.grid_size - 1) * self.config.grid_spacing / 2
        for i in range(self.grid_size):
            offsets.append(-half_span + i * self.config.grid_spacing)

        for body_name in FINGERTIP_BODIES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            self._fingertip_body_ids.append(bid)

            geom_ids: set[int] = set()
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] == bid:
                    geom_ids.add(gid)
            self._fingertip_geom_ids_per_finger.append(geom_ids)

            pad_z = FINGERTIP_OFFSETS[body_name][2]

            n_taxels_per_finger = self.grid_size**2
            positions = np.zeros((n_taxels_per_finger, 3), dtype=np.float64)
            idx = 0
            for dx in offsets:
                for dy in offsets:
                    positions[idx] = [dx, dy, pad_z]
                    idx += 1
            self._taxel_local_positions.append(positions)

    def get_taxel_world_positions(self, data: mujoco.MjData) -> list[np.ndarray]:
        """Transform the local taxel grids to world coordinates using current body poses."""
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
        """Read the tactile sensors — returns current, previous, and change vectors.

        Args:
            model: MuJoCo model
            data: sim data with current contacts

        Returns:
            (current, previous, change) — each is (80,) flat array
        """
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
        """Clear the reading history (zeros out previous readings)."""
        self._previous_readings = np.zeros(self.n_taxels)
