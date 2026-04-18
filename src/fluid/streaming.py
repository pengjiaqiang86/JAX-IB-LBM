import numpy as np
import jax.numpy as jnp

from src.core.lattice import Lattice


def stream(
    f: jnp.ndarray,
    lattice: Lattice,
) -> jnp.ndarray:
    """
    Streaming (advection) step.  
    Shift each f[..., q] by the corresponding lattice velocity c[q].

    Uses jnp.roll — inherently periodic on all axes.  
    Boundary conditions override the boundary cells after this step.

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
    # spatial axis for velocity component d: component 0 (x) -> last spatial axis
    spatial_axes = list(range(ndim - 1, -1, -1))  # [1,0] for 2D, [2,1,0] for 3D

    # np.asarray forces a concrete NumPy array regardless of JAX tracing context.
    # LatticeDescriptor is a NamedTuple, so JAX treats it as a pytree and
    # makes its array fields (c, w, opp) abstract tracers when it crosses any
    # JIT/scan boundary. np.asarray extracts concrete values before indexing.
    c_np = np.asarray(lattice.c)   # (Q, D)  always concrete Python ints

    slices = []
    for q in range(lattice.Q):
        fi = f[..., q]                              # (*spatial,)
        for comp, ax in enumerate(spatial_axes):
            shift = int(c_np[q, comp])              # always a concrete Python int
            if shift != 0:
                fi = jnp.roll(fi, shift, axis=ax)
        slices.append(fi)

    return jnp.stack(slices, axis=-1)           # (*spatial, Q)
