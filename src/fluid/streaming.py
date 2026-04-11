"""
Streaming (advection) step.

Uses jnp.roll — inherently periodic on all axes.
Boundary conditions override the boundary cells after this step.
"""

import jax
import jax.numpy as jnp

from src.core.lattice import Lattice


def stream(
    f: jnp.ndarray,
    lattice: Lattice,
) -> jnp.ndarray:
    """
    Shift each f[..., q] by the corresponding lattice velocity c[q].

    Parameters
    ----------
    f       : (*spatial, Q)  — spatial axes first, Q last
    lattice : Lattice

    Returns
    -------
    f_streamed : (*spatial, Q)

    Notes
    -----
    Spatial axes in f are indexed from 0 to ndim-1 (left to right = z, y, x).
    c[q] has components ordered (cx, cy[, cz]) but stored as (x, y[, z]).
    The roll axes must map component d of c[q] to the correct spatial axis.

    For D=2: c[:,0]=cx -> axis 1 (NX-axis), c[:,1]=cy -> axis 0 (NY-axis)
    For D=3: c[:,0]=cx -> axis 2, c[:,1]=cy -> axis 1, c[:,2]=cz -> axis 0
    """
    ndim = lattice.D
    # spatial axis for each velocity component: component 0 -> last spatial axis
    spatial_axes = list(range(ndim - 1, -1, -1))  # [1,0] for 2D, [2,1,0] for 3D

    def shift_one(fi, cq):
        # fi: (*spatial,), cq: (D,)
        for comp, ax in enumerate(spatial_axes):
            fi = jnp.roll(fi, int(cq[comp]), axis=ax)
        return fi

    # vmap over the Q axis (last axis of f)
    f_shifted = jax.vmap(shift_one, in_axes=(2, 0), out_axes=2)(f, lattice.c)
    return f_shifted
