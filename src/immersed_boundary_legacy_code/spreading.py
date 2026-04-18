"""
Force spreading: Lagrangian markers → Eulerian grid.

    f(x) = Σ_k F_k * δ_h(x - X_k) * ds_k

For each marker k we add the weighted force contribution to the
(2*support)^D Eulerian stencil surrounding X_k.

JAX does not support scatter-add inside vmap with overlapping indices,
so we loop over the stencil offsets explicitly and use jnp.index_update
(via .at[].add()) in a fori_loop.
"""

import jax
import jax.numpy as jnp

from src.core.grid import EulerianGrid
from src.immersed_boundary.markers import LagrangianBody
from src.immersed_boundary.delta import DeltaKernel, PESKIN_4PT


def ib_force_spreading(
    body:   LagrangianBody,
    grid:   EulerianGrid,
    kernel: DeltaKernel = PESKIN_4PT,
) -> jnp.ndarray:
    """
    Spread Lagrangian force density onto the Eulerian grid.

    Parameters
    ----------
    body   : LagrangianBody  —  body.F : (N, D), body.X : (N, D),
                                 body.ds : (N,)
    grid   : EulerianGrid
    kernel : delta kernel (default: Peskin 4-point)

    Returns
    -------
    g : (*spatial, D)   Eulerian body-force density
    """
    D       = grid.ndim
    spacing = grid.spacing           # (D,) = (dy, dx) or (dz, dy, dx)
    support = kernel.support
    offsets = list(range(-support + 1, support + 1))   # e.g. [-1,0,1,2]
    N       = body.X.shape[0]

    g = jnp.zeros(grid.shape + (D,))   # (*spatial, D)

    # Physical coordinate arrays for each axis (for delta evaluation)
    # axis order in grid.shape: (NY, NX) or (NZ, NY, NX)
    # spacing order: (dy, dx) or (dz, dy, dx)

    # Pre-compute spacing as a plain JAX array once (outside the loop).
    sp = jnp.array(spacing)          # (D,)  always concrete at build time

    def spread_one(g, k):
        X_k  = body.X[k]             # (D,)  — traced inside fori_loop
        F_k  = body.F[k]             # (D,)
        ds_k = body.ds[k]            # scalar

        # Base grid index: component d of X_k maps to spatial axis (D-1-d).
        # Keep as JAX int32 — do NOT call int() on traced values.
        idx = jnp.array([X_k[d] / spacing[D - 1 - d] for d in range(D)])
        i0  = jnp.floor(idx).astype(jnp.int32)   # (D,)  traced int32

        if D == 2:
            for oy in offsets:       # Python loop: oy, ox are concrete Python ints
                for ox in offsets:
                    # jnp.mod with a traced base + concrete offset → traced int32
                    iy = jnp.mod(i0[1] + oy, grid.NY)
                    ix = jnp.mod(i0[0] + ox, grid.NX)
                    # Physical position of this stencil cell (traced floats)
                    x_grid = jnp.stack([ix * sp[-1], iy * sp[-2]])
                    r = (x_grid - X_k) / sp[::-1]
                    w = jnp.prod(kernel.phi(r)) * ds_k
                    # .at[traced_idx].add() is fully supported in JAX
                    g = g.at[iy, ix, :].add(F_k * w)

        elif D == 3:
            for oz in offsets:
                for oy in offsets:
                    for ox in offsets:
                        iz = jnp.mod(i0[2] + oz, grid.NZ)
                        iy = jnp.mod(i0[1] + oy, grid.NY)
                        ix = jnp.mod(i0[0] + ox, grid.NX)
                        x_grid = jnp.stack([ix * sp[-1], iy * sp[-2], iz * sp[-3]])
                        r = (x_grid - X_k) / sp[::-1]
                        w = jnp.prod(kernel.phi(r)) * ds_k
                        g = g.at[iz, iy, ix, :].add(F_k * w)
        return g

    # fori_loop over markers (JIT-safe, no Python-level scatter race)
    g = jax.lax.fori_loop(0, N, lambda k, g: spread_one(g, k), g)
    return g
