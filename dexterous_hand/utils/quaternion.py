import numpy as np


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (Hamilton product).

    Note: everything in this file uses MuJoCo's (w, x, y, z) convention.

    Args:
        q1: first quaternion [w, x, y, z]
        q2: second quaternion [w, x, y, z]

    Returns:
        Product quaternion [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of a quaternion (same as inverse for unit quats).

    Args:
        q: quaternion [w, x, y, z]

    Returns:
        [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_error(q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """Get the rotation needed to go from q_current to q_target.

    Args:
        q_current: where we are now [w, x, y, z]
        q_target: where we want to be [w, x, y, z]

    Returns:
        Error quaternion [w, x, y, z].
    """
    return quat_multiply(q_target, quat_conjugate(q_current))


def quat_to_angular_distance(q_error: np.ndarray) -> float:
    """Convert an error quaternion to an angle (geodesic distance on SO(3)).

    Args:
        q_error: error quaternion [w, x, y, z]

    Returns:
        Angle in radians [0, pi].
    """
    w = np.clip(np.abs(q_error[0]), 0.0, 1.0)
    return float(2.0 * np.arccos(w))


def quat_angular_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angle between two orientations (geodesic on SO(3)).

    Uses the formula 2 * arccos(|q1 · q2|), which handles the double-cover nicely.

    Args:
        q1: first quaternion [w, x, y, z]
        q2: second quaternion [w, x, y, z]

    Returns:
        Angle in radians [0, pi].
    """
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * float(np.arccos(dot))


def random_quaternion(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly random rotation (Marsaglia's method for SO(3)).

    Args:
        rng: numpy random generator

    Returns:
        Unit quaternion [w, x, y, z].
    """
    u1, u2, u3 = rng.uniform(0, 1, size=3)
    s1 = np.sqrt(1.0 - u1)
    s2 = np.sqrt(u1)
    a1 = 2.0 * np.pi * u2
    a2 = 2.0 * np.pi * u3
    return np.array([s2 * np.cos(a2), s1 * np.sin(a1), s1 * np.cos(a1), s2 * np.sin(a2)])


def random_quaternion_within_angle(rng: np.random.Generator, max_angle_rad: float) -> np.ndarray:
    """Random rotation with angle capped at max_angle_rad.

    Picks a random axis and a random angle up to the limit. If max_angle >= 2*pi
    we just sample the full SO(3) instead.

    Args:
        rng: numpy random generator
        max_angle_rad: max rotation angle in radians

    Returns:
        Unit quaternion [w, x, y, z].
    """
    if max_angle_rad >= np.pi * 2.0:
        return random_quaternion(rng)

    z = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    r = np.sqrt(1.0 - z * z)
    axis = np.array([r * np.cos(phi), r * np.sin(phi), z])

    angle = rng.uniform(0.0, max_angle_rad)
    half = angle / 2.0
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion to a 3x3 rotation matrix.

    Args:
        q: unit quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix.
    """
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion (q * v * q_conj sandwich product).

    Args:
        q: unit quaternion [w, x, y, z]
        v: (3,) vector to rotate

    Returns:
        Rotated vector (3,).
    """
    q_v = np.array([0.0, v[0], v[1], v[2]])
    result = quat_multiply(quat_multiply(q, q_v), quat_conjugate(q))
    return result[1:]


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Axis-angle to quaternion.

    Args:
        axis: (3,) unit rotation axis
        angle: rotation angle in radians

    Returns:
        Quaternion [w, x, y, z].
    """
    half = angle / 2.0
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
    """Quaternion to axis-angle.

    Args:
        q: quaternion [w, x, y, z]

    Returns:
        axis: (3,) unit rotation axis
        angle: rotation in radians [0, pi]
    """
    w = np.clip(q[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(np.abs(w))
    s = np.sqrt(1.0 - w * w)

    if s < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = q[1:] / s
        if w < 0:
            axis = -axis

    return axis, float(angle)
