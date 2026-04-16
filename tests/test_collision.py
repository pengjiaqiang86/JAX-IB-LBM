import jax.numpy as jnp

from src.fluid.collision import mrt_collision


def test_mrt_collision_forcing_identity_case():
    """
    With M = I and S = I, MRT reduces to:
        f_post = feq + 0.5 * g_force
    """
    f = jnp.ones((3, 4, 9))
    feq = 2.0 * jnp.ones((3, 4, 9))
    M = jnp.eye(9)
    S = jnp.eye(9)
    g_force = jnp.full((3, 4, 9), 0.2)

    f_post = mrt_collision(f, feq, M, S, g_force)

    assert jnp.allclose(f_post, feq + 0.5 * g_force)
