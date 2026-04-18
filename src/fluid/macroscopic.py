from typing import Optional, Tuple

import jax.numpy as jnp

from src.core.lattice import Lattice


def compute_macroscopic(
    f:       jnp.ndarray,
    lattice: Lattice,
    g:       Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute density and velocity from distribution functions.

    With body force (Guo 2002 scheme), the macroscopic velocity is defined as:

        ρ = Σ_q f_q
        ρu = Σ_q f_q c_q  +  g Δt/2

    The g/(2ρ) correction ensures that u is the physical fluid velocity
    when an external body force is present.  Without it (g=None) the
    second term vanishes and the standard LBM definition is recovered.

    Parameters
    ----------
    f       : (*spatial, Q)
    lattice : Lattice
    g       : (*spatial, D) or None  — Eulerian body-force density.
              Pass state.g to get the physically correct velocity.

    Returns
    -------
    rho : (*spatial,)
    u   : (*spatial, D)
    """
    rho = jnp.sum(f, axis=-1)                                   # (*spatial,)
    u   = jnp.einsum("qd,...q->...d", lattice.c, f) / rho[..., None]

    if g is not None:
        # Guo (2002) velocity correction: u_phys = u_raw + g / (2ρ)
        u = u + g / (2.0 * rho[..., None])

    return rho, u
