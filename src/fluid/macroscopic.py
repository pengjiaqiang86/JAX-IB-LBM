"""
Macroscopic quantities from distribution functions.

All functions are pure JAX — no side effects, JIT-safe.
"""

from typing import Tuple

import jax.numpy as jnp

from src.core.lattice import Lattice


def compute_macroscopic(
    f: jnp.ndarray,
    lattice: Lattice,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute density and velocity from distribution functions.

    Parameters
    ----------
    f       : (*spatial, Q)
    lattice : Lattice

    Returns
    -------
    rho : (*spatial,)      density
    u   : (*spatial, D)    velocity
    """
    rho = jnp.sum(f, axis=-1)                                   # (*spatial,)
    # sum_q  c[q, d] * f[..., q]  /  rho
    u = jnp.einsum("qd,...q->...d", lattice.c, f) / rho[..., None]
    return rho, u
