import mujoco
import numpy as np


def get_fingertip_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    fingertip_geom_ids: set[int],
    object_geom_id: int,
) -> tuple[int, set[int]]:
    """Check which fingertips are touching the object.

    Args:
        model: MuJoCo model
        data: sim data (has the contact list)
        fingertip_geom_ids: geom IDs that belong to fingertips
        object_geom_id: the object's geom ID

    Returns:
        count: how many distinct fingertip bodies are in contact
        contact_body_ids: which fingertip bodies are touching
    """
    contact_body_ids: set[int] = set()
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        if g1 in fingertip_geom_ids and g2 == object_geom_id:
            contact_body_ids.add(model.geom_bodyid[g1])
        elif g2 in fingertip_geom_ids and g1 == object_geom_id:
            contact_body_ids.add(model.geom_bodyid[g2])
    return len(contact_body_ids), contact_body_ids


def get_fingertip_positions(
    data: mujoco.MjData,
    fingertip_site_ids: list[int],
) -> np.ndarray:
    """Get world positions of all fingertip sites.

    Args:
        data: MuJoCo sim data
        fingertip_site_ids: site IDs for each fingertip

    Returns:
        (N, 3) array of positions.
    """
    return np.array([data.site_xpos[sid].copy() for sid in fingertip_site_ids])


def get_object_state(
    data: mujoco.MjData,
    object_body_id: int,
    obj_qpos_start: int,
    obj_qvel_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get position, orientation, and velocities of a free-joint object.

    Args:
        data: MuJoCo sim data
        object_body_id: body ID
        obj_qpos_start: where this object's qpos starts in the array
        obj_qvel_start: where this object's qvel starts

    Returns:
        position (3,), quaternion (4,), linear_vel (3,), angular_vel (3,)
    """
    position = data.xpos[object_body_id].copy()
    orientation = data.qpos[obj_qpos_start + 3 : obj_qpos_start + 7].copy()
    linear_vel = data.qvel[obj_qvel_start : obj_qvel_start + 3].copy()
    angular_vel = data.qvel[obj_qvel_start + 3 : obj_qvel_start + 6].copy()
    return position, orientation, linear_vel, angular_vel


def get_palm_position(data: mujoco.MjData, palm_body_id: int) -> np.ndarray:
    """Get the palm's world position."""
    return np.array(data.xpos[palm_body_id].copy())


def get_geom_ids_for_bodies(
    model: mujoco.MjModel,
    body_names: list[str],
) -> dict[str, list[int]]:
    """Given body names, find all the geom IDs that belong to each body.

    Args:
        model: MuJoCo model
        body_names: which bodies to look up

    Returns:
        Dict mapping body name -> list of its geom IDs.
    """
    result: dict[str, list[int]] = {name: [] for name in body_names}

    body_name_to_id = {}
    for name in body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            body_name_to_id[name] = bid

    id_to_name = {v: k for k, v in body_name_to_id.items()}
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        if body_id in id_to_name:
            result[id_to_name[body_id]].append(geom_id)

    return result


def get_cube_face_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_geom_id: int,
    fingertip_geom_ids: set[int],
) -> np.ndarray:
    """Figure out which cube faces have a finger touching them.

    Returns 6 binary flags for +X, -X, +Y, -Y, +Z, -Z in the cube's local frame.
    We project contact normals into the cube's frame and pick the closest face.

    Args:
        model: MuJoCo model
        data: sim data
        cube_geom_id: the cube's geom ID
        fingertip_geom_ids: fingertip geom IDs

    Returns:
        (6,) binary flags, one per face.
    """
    flags = np.zeros(6, dtype=np.float64)
    cube_rot = data.geom_xmat[cube_geom_id].reshape(3, 3)

    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2

        is_cube_finger = (g1 == cube_geom_id and g2 in fingertip_geom_ids) or (
            g2 == cube_geom_id and g1 in fingertip_geom_ids
        )
        if not is_cube_finger:
            continue

        normal_world = contact.frame[:3]
        normal_local = cube_rot.T @ normal_world

        axis = int(np.argmax(np.abs(normal_local)))
        sign = 1 if normal_local[axis] > 0 else -1
        face_idx = axis * 2 + (0 if sign > 0 else 1)
        flags[face_idx] = 1.0

    return flags


def get_contact_forces_on_body(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_geom_ids: set[int],
    source_geom_ids: set[int],
) -> np.ndarray:
    """Sum up contact forces/torques between two sets of geoms.

    Args:
        model: MuJoCo model
        data: sim data
        target_geom_ids: geoms we care about (e.g. peg)
        source_geom_ids: geoms applying forces (e.g. hole walls)

    Returns:
        (6,) wrench [fx, fy, fz, tx, ty, tz] in world frame.
    """
    wrench = np.zeros(6, dtype=np.float64)
    force_buf = np.zeros(6, dtype=np.float64)

    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2

        is_match = (g1 in target_geom_ids and g2 in source_geom_ids) or (
            g2 in target_geom_ids and g1 in source_geom_ids
        )
        if not is_match:
            continue

        mujoco.mj_contactForce(model, data, i, force_buf)
        frame = contact.frame.reshape(3, 3)
        wrench[:3] += frame.T @ force_buf[:3]
        wrench[3:] += frame.T @ force_buf[3:]

    return wrench


def get_insertion_depth(
    data: mujoco.MjData,
    peg_body_id: int,
    hole_body_id: int,
    peg_half_length: float,
) -> float:
    """How far the peg tip is inside the hole (along the hole's Z axis).

    Args:
        data: sim data
        peg_body_id: peg body ID
        hole_body_id: hole body ID
        peg_half_length: half the peg length

    Returns:
        Depth in meters (0 if not inserted, positive = inside).
    """
    peg_pos = data.xpos[peg_body_id]
    hole_pos = data.xpos[hole_body_id]
    hole_rot = data.xmat[hole_body_id].reshape(3, 3)

    hole_axis = hole_rot[:, 2]

    peg_rot = data.xmat[peg_body_id].reshape(3, 3)
    peg_axis = peg_rot[:, 2]
    peg_tip = peg_pos - peg_axis * peg_half_length

    rel = hole_pos - peg_tip
    depth = float(np.dot(rel, hole_axis))

    return max(depth, 0.0)


def get_peg_hole_relative(
    data: mujoco.MjData,
    peg_body_id: int,
    hole_body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Get relative position and angular error between peg and hole.

    Args:
        data: sim data
        peg_body_id: peg body ID
        hole_body_id: hole body ID

    Returns:
        rel_pos: (3,) peg position relative to hole
        angular_error: (3,) axis-angle error between peg and hole Z axes
    """
    peg_pos = data.xpos[peg_body_id].copy()
    hole_pos = data.xpos[hole_body_id].copy()
    rel_pos = peg_pos - hole_pos

    peg_rot = data.xmat[peg_body_id].reshape(3, 3)
    hole_rot = data.xmat[hole_body_id].reshape(3, 3)

    peg_axis = peg_rot[:, 2]
    hole_axis = hole_rot[:, 2]

    cross = np.cross(peg_axis, hole_axis)
    dot = np.clip(np.dot(peg_axis, hole_axis), -1.0, 1.0)
    angle = np.arccos(np.abs(dot))
    norm = np.linalg.norm(cross)
    angular_error = cross / norm * angle if norm > 1e-8 else np.zeros(3)

    return rel_pos, angular_error


def distribute_contact_forces_to_taxels(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    fingertip_geom_ids_per_finger: list[set[int]],
    taxel_world_positions: list[np.ndarray],
    max_force: float = 10.0,
    noise_std: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Spread MuJoCo contact forces across nearby taxels using inverse-distance weighting.

    For each contact on a fingertip, we find the 3 nearest taxels and distribute
    the normal force based on how close each taxel is.

    Args:
        model: MuJoCo model
        data: sim data with current contacts
        fingertip_geom_ids_per_finger: list of 5 sets of geom IDs, one per finger
        taxel_world_positions: list of 5 arrays, each (16, 3) taxel positions
        max_force: clip taxel readings to this max
        noise_std: gaussian noise std in Newtons (simulates real sensor noise)
        rng: random generator for noise, None = no noise

    Returns:
        (5, 4, 4) taxel force readings per finger.
    """
    n_fingers = len(fingertip_geom_ids_per_finger)
    readings = np.zeros((n_fingers, 16), dtype=np.float64)

    geom_to_finger: dict[int, int] = {}
    for fi, geom_ids in enumerate(fingertip_geom_ids_per_finger):
        for gid in geom_ids:
            geom_to_finger[gid] = fi

    force_buf = np.zeros(6, dtype=np.float64)

    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2

        finger_idx = None
        if g1 in geom_to_finger:
            finger_idx = geom_to_finger[g1]
        elif g2 in geom_to_finger:
            finger_idx = geom_to_finger[g2]

        if finger_idx is None:
            continue

        mujoco.mj_contactForce(model, data, i, force_buf)
        normal_force = abs(force_buf[0])
        if normal_force < 1e-8:
            continue

        contact_pos = contact.pos

        taxel_pos = taxel_world_positions[finger_idx]
        dists = np.linalg.norm(taxel_pos - contact_pos, axis=1)

        nearest_idx = np.argsort(dists)[:3]
        nearest_dists = dists[nearest_idx]
        inv_dists = 1.0 / np.maximum(nearest_dists, 1e-6)
        weights = inv_dists / inv_dists.sum()

        for j, idx in enumerate(nearest_idx):
            readings[finger_idx, idx] += normal_force * weights[j]

    if rng is not None and noise_std > 0:
        readings += rng.normal(0.0, noise_std, size=readings.shape)

    np.clip(readings, 0.0, max_force, out=readings)

    return readings.reshape(n_fingers, 4, 4)


def get_contact_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    wall_geom_ids: list[int],
) -> tuple[np.ndarray, float]:
    """Get how much force each wall is exerting on a geom (e.g. peg vs hole walls).

    Args:
        model: MuJoCo model
        data: sim data
        geom_id: the geom being pushed on
        wall_geom_ids: wall geom IDs

    Returns:
        per_wall_forces: normal force on each wall
        total_magnitude: L2 norm of the force vector
    """
    per_wall = np.zeros(len(wall_geom_ids), dtype=np.float64)
    force_buf = np.zeros(6, dtype=np.float64)

    wall_to_idx = {wid: i for i, wid in enumerate(wall_geom_ids)}

    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2

        wall_idx = None
        if g1 == geom_id and g2 in wall_to_idx:
            wall_idx = wall_to_idx[g2]
        elif g2 == geom_id and g1 in wall_to_idx:
            wall_idx = wall_to_idx[g1]

        if wall_idx is None:
            continue

        mujoco.mj_contactForce(model, data, i, force_buf)
        per_wall[wall_idx] += abs(force_buf[0])

    total_mag = float(np.linalg.norm(per_wall))
    return per_wall, total_mag


def get_body_axis(
    data: mujoco.MjData,
    body_id: int,
    axis: int = 2,
) -> np.ndarray:
    """Get a body's local axis in world frame (default Z axis).

    Args:
        data: sim data
        body_id: which body
        axis: 0=X, 1=Y, 2=Z

    Returns:
        (3,) unit vector in world frame.
    """
    return np.array(data.xmat[body_id].reshape(3, 3)[:, axis].copy())
