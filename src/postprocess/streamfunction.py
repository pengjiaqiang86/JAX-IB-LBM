import jax
import jax.numpy as jnp


def solve_streamfunction(
    omega:      jnp.ndarray,    # (NY, NX)  vorticity
    solid_mask: jnp.ndarray,    # (NY, NX)  bool, True on solid
    n_iter:     int   = 400,
    dx:         float = 1.0,
    dy:         float = 1.0,
) -> jnp.ndarray:
    """
    Solve Δψ = -ω with ψ = 0 on solid nodes using Jacobi iteration.

    Parameters
    ----------
    omega      : (NY, NX)  vorticity field
    solid_mask : (NY, NX)  bool mask (walls + obstacles)
    n_iter     : number of Jacobi iterations
    dx, dy     : grid spacings

    Returns
    -------
    psi : (NY, NX)  streamfunction
    """
    dx2  = dx * dx
    dy2  = dy * dy
    coef = 1.0 / (2.0 * (dx2 + dy2))

    psi0 = jnp.zeros_like(omega)

    def body_fn(_, psi):
        psi_e = jnp.roll(psi, -1, axis=1)
        psi_w = jnp.roll(psi,  1, axis=1)
        psi_n = jnp.roll(psi, -1, axis=0)
        psi_s = jnp.roll(psi,  1, axis=0)

        psi_new = coef * (
            (psi_e + psi_w) * dy2 +
            (psi_n + psi_s) * dx2 +
            omega * dx2 * dy2
        )
        return jnp.where(solid_mask, 0.0, psi_new)

    psi = jax.lax.fori_loop(0, n_iter, body_fn, psi0)
    return psi
