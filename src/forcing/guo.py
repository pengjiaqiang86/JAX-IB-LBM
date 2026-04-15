"""
Guo et al. (2002) body-force scheme for LBM.

The full BGK update with body force is (Guo 2002, Eq. 20):

    f_post = f - ω(f - feq) + (1 - Δt/(2τ)) · F_q

where F_q is the discrete forcing term:

    F_q = w_q · [(c_q − u)/cs²  +  (c_q · u) c_q / cs⁴] · g

This module computes F_q only.  The prefactor (1 − ω/2) = (1 − Δt/(2τ))
is applied separately in bgk_collision / mrt_collision, so that the
collision operator remains responsible for all terms it adds to f.

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
