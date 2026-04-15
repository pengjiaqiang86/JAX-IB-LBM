"""
Guo et al. (2002) body-force scheme for LBM.

Converts a physical body-force density g (*spatial, D) into a source
term F (*spatial, Q) that is added to the BGK collision:

    f_post = f - omega*(f - feq) + (1 - omega/2) * F

where

    F_q = w_q * [ (c_q - u)/cs² + (c_q · u) * c_q / cs⁴ ] · g

cs² is read from lattice.cs2.

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
) -> jnp.ndarray:
    """
    Compute the Guo forcing source term F (*spatial, Q).

    Parameters
    ----------
    g       : (*spatial, D)  body force density (physical units)
    u       : (*spatial, D)  fluid velocity
    lattice : Lattice        cs² is taken from lattice.cs2

    Returns
    -------
    F : (*spatial, Q)  source term to be added to BGK collision
    """
    cs2 = lattice.cs2

    # c_q · u  →  (*spatial, Q)
    cu = jnp.einsum("qd,...d->...q", lattice.c, u)

    # c_q · g  →  (*spatial, Q)
    cg = jnp.einsum("qd,...d->...q", lattice.c, g)

    # (c_q - u) · g = c_q·g - u·g
    ug = jnp.sum(u * g, axis=-1, keepdims=True)   # (*spatial, 1)

    F = lattice.w * (
        (cg - ug) / cs2
        + cu * cg / cs2 ** 2
    )
    return F
