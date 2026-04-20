
import mujoco
import numpy as np

                                          
JOINT_QPOS_SIZE = {0: 7, 1: 4, 2: 1, 3: 1}                            
JOINT_DOF_SIZE = {0: 6, 1: 3, 2: 1, 3: 1}

def get_joint_qpos_qvel_range(
    model: mujoco.MjModel,
    joint_ids: list[int],
) -> tuple[int, int, int, int]:

    first, last = joint_ids[0], joint_ids[-1]
    last_type = int(model.jnt_type[last])

    qpos_start = int(model.jnt_qposadr[first])
    qpos_end = int(model.jnt_qposadr[last] + JOINT_QPOS_SIZE[last_type])
    qvel_start = int(model.jnt_dofadr[first])
    qvel_end = int(model.jnt_dofadr[last] + JOINT_DOF_SIZE[last_type])

    return qpos_start, qpos_end, qvel_start, qvel_end

def get_fingertip_positions(
    data: mujoco.MjData,
    fingertip_site_ids: list[int],
) -> np.ndarray:

    result: np.ndarray = data.site_xpos[fingertip_site_ids].copy()
    return result

def get_finger_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    finger_geom_ids_per_finger: list[set[int]],
    object_geom_id: int,
) -> tuple[int, set[int]]:

    contact_finger_indices: set[int] = set()
    geom_to_finger: dict[int, int] = {}

    for finger_idx, geom_ids in enumerate(finger_geom_ids_per_finger):
        for gid in geom_ids:
            geom_to_finger[gid] = finger_idx

    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2

        if g1 == object_geom_id and g2 in geom_to_finger:
            contact_finger_indices.add(geom_to_finger[g2])
        elif g2 == object_geom_id and g1 in geom_to_finger:
            contact_finger_indices.add(geom_to_finger[g1])

    return len(contact_finger_indices), contact_finger_indices

def get_object_state(
    data: mujoco.MjData,
    object_body_id: int,
    obj_qpos_start: int,
    obj_qvel_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    position = data.xpos[object_body_id].copy()
    orientation = data.qpos[obj_qpos_start + 3 : obj_qpos_start + 7].copy()
    linear_vel = data.qvel[obj_qvel_start : obj_qvel_start + 3].copy()
    angular_vel = data.qvel[obj_qvel_start + 3 : obj_qvel_start + 6].copy()

    return position, orientation, linear_vel, angular_vel

def get_palm_position(data: mujoco.MjData, palm_body_id: int) -> np.ndarray:

    return np.array(data.xpos[palm_body_id].copy())

def get_cube_face_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_geom_id: int,
    fingertip_geom_ids: set[int],
) -> np.ndarray:

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
    peg_radius: float = 0.0,
) -> float:

    peg_pos = data.xpos[peg_body_id]
    hole_pos = data.xpos[hole_body_id]
    hole_rot = data.xmat[hole_body_id].reshape(3, 3)

    hole_axis = hole_rot[:, 2]

    peg_rot = data.xmat[peg_body_id].reshape(3, 3)
    peg_axis = peg_rot[:, 2]
    peg_tip = peg_pos - peg_axis * (peg_half_length + peg_radius)

    rel = hole_pos - peg_tip
    depth = float(np.dot(rel, hole_axis))

    return max(depth, 0.0)

def get_peg_hole_relative(
    data: mujoco.MjData,
    peg_body_id: int,
    hole_body_id: int,
) -> tuple[np.ndarray, np.ndarray]:

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

    return np.array(data.xmat[body_id].reshape(3, 3)[:, axis].copy())
