
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

def get_body_axis(
    data: mujoco.MjData,
    body_id: int,
    axis: int = 2,
) -> np.ndarray:

    return np.array(data.xmat[body_id].reshape(3, 3)[:, axis].copy())
