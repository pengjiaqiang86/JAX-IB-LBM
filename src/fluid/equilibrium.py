"""
Maxwell-Boltzmann equilibrium distribution for LBM.

Works for any lattice (D2Q9, D3Q19, D3Q27) through the generic
Lattice interface.
"""

import jax.numpy as jnp

from src.core.lattice import Lattice


def compute_equilibrium(
    rho: jnp.ndarray,
    u:   jnp.ndarray,
    lattice: Lattice,
) -> jnp.ndarray:
    """
    Second-order Maxwell-Boltzmann equilibrium.

        f_eq[q] = w[q] * rho * (1 + 3(c[q]·u) + 4.5(c[q]·u)^2 - 1.5|u|^2)

    Parameters
    ----------
    rho     : (*spatial,)      density
    u       : (*spatial, D)    velocity
    lattice : Lattice

    Returns
    -------
    feq : (*spatial, Q)
    """
    # c[q]·u  for every spatial point and every direction q
    # c : (Q, D),  u : (*spatial, D)  →  cu : (*spatial, Q)
    cu  = jnp.einsum("qd,...d->...q", lattice.c, u)

    # |u|^2 : (*spatial, 1)  — broadcast over Q
    usq = jnp.sum(u ** 2, axis=-1, keepdims=True)

    feq = lattice.w * rho[..., None] * (
        1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq
    )
    return feq
