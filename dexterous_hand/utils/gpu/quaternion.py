
import jax
import jax.numpy as jnp

def quat_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:

    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )

def quat_conjugate(q: jnp.ndarray) -> jnp.ndarray:

    return jnp.array([q[0], -q[1], -q[2], -q[3]])

def quat_angular_distance(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:

    dot = jnp.clip(jnp.abs(jnp.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * jnp.arccos(dot)

def random_quaternion_within_angle(
    key: jax.Array,
    max_angle_rad: float | jax.Array,
) -> jnp.ndarray:

    k1, k2, k3 = jax.random.split(key, 3)
    z = jax.random.uniform(k1, minval=-1.0, maxval=1.0)
    phi = jax.random.uniform(k2, minval=0.0, maxval=2.0 * jnp.pi)
    r = jnp.sqrt(1.0 - z * z)
    axis = jnp.array([r * jnp.cos(phi), r * jnp.sin(phi), z])

    angle = jax.random.uniform(k3, minval=0.0, maxval=max_angle_rad)
    half = angle / 2.0
    s = jnp.sin(half)
    return jnp.array([jnp.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])
