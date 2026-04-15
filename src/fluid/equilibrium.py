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

    General form:
        f_eq[q] = w[q] * ρ * (1 + (c·u)/cs²
                                 + (c·u)² / (2 cs⁴)
                                 - |u|²   / (2 cs²))

    cs² is read from lattice.cs2 (1/3 for D2Q9, D3Q19, D3Q27).

    Parameters
    ----------
    rho     : (*spatial,)      density
    u       : (*spatial, D)    velocity
    lattice : Lattice

    Returns
    -------
    feq : (*spatial, Q)
    """
    cs2 = lattice.cs2

    # c[q]·u  for every spatial point and every direction q
    # c : (Q, D),  u : (*spatial, D)  →  cu : (*spatial, Q)
    cu  = jnp.einsum("qd,...d->...q", lattice.c, u)

    # |u|^2 : (*spatial, 1)  — broadcast over Q
    usq = jnp.sum(u ** 2, axis=-1, keepdims=True)

    feq = lattice.w * rho[..., None] * (
        1.0
        + cu  / cs2
        + cu ** 2 / (2.0 * cs2 ** 2)
        - usq / (2.0 * cs2)
    )
    return feq
