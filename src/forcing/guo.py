"""
Guo et al. (2002) body-force scheme for LBM.

Converts a physical body-force density g (*spatial, D) into a source
term F (*spatial, Q) that is added to the BGK collision:

    f_post = f - omega*(f - feq) + (1 - omega/2) * F

where

    F_q = w_q * [ (c_q - u)/cs^2 + (c_q · u) * c_q / cs^4 ] · g

with cs^2 = 1/3 for standard lattices.

Reference
---------
Guo, Zheng, Shi, Phys Rev E 65 (2002), 046308.
"""

import jax.numpy as jnp

from src.core.lattice import Lattice


def guo_forcing_term(
    g:       jnp.ndarray,    # (*spatial, D)  body force density
    u:       jnp.ndarray,    # (*spatial, D)  fluid velocity
    lattice: Lattice,
    cs2:     float = 1.0 / 3.0,
) -> jnp.ndarray:
    """
    Compute the Guo forcing source term F (*spatial, Q).

    Parameters
    ----------
    g       : (*spatial, D)  body force density (physical units)
    u       : (*spatial, D)  fluid velocity
    lattice : Lattice
    cs2     : lattice speed of sound squared (1/3 for standard LBM)

    Returns
    -------
    F : (*spatial, Q)  source term to be added to BGK collision
    """
    # c_q · u  →  (*spatial, Q)
    cu = jnp.einsum("qd,...d->...q", lattice.c, u)

    # c_q · g  →  (*spatial, Q)
    cg = jnp.einsum("qd,...d->...q", lattice.c, g)

    # u · g  →  (*spatial, 1)  — not needed in Guo; term is (c_q - u) · g
    # (c_q - u) · g = c_q·g - u·g
    ug = jnp.sum(u * g, axis=-1, keepdims=True)   # (*spatial, 1)

    F = lattice.w * (
        (cg - ug) / cs2
        + cu * cg / cs2 ** 2
    )
    return F
