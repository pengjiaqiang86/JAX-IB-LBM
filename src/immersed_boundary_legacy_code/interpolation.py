"""
Velocity interpolation: Eulerian grid → Lagrangian markers.

    U_k = Σ_{x in stencil(X_k)} u(x) * δ_h(x - X_k) * dx^D

For each marker k, we sum the Eulerian velocity u over the
(2*support)^D stencil centred on the nearest grid point to X_k.

This uses jax.vmap over markers and jnp.roll-based stencil indexing
so it is fully JIT-compatible.
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from src.core.grid import EulerianGrid
from src.immersed_boundary.markers import LagrangianBody
from src.immersed_boundary.delta import DeltaKernel, PESKIN_4PT


def ib_velocity_interpolation(
    u:      jnp.ndarray,      # (*spatial, D)  Eulerian velocity
    body:   LagrangianBody,
    grid:   EulerianGrid,
    kernel: DeltaKernel = PESKIN_4PT,
) -> jnp.ndarray:
    """
    Interpolate Eulerian velocity to each Lagrangian marker.

    Parameters
    ----------
    u      : (*spatial, D)  Eulerian velocity field
    body   : LagrangianBody  with X in physical coords
    grid   : EulerianGrid
    kernel : delta kernel (default: Peskin 4-point)

    Returns
    -------
    U_L : (N, D)  velocity at each marker
    """
    spacing = grid.spacing         # (D,) = (dy, dx) or (dz, dy, dx)
    support = kernel.support
    D = grid.ndim

    def interp_one_marker(X_k):
        # nearest grid index (physical -> index)
        idx = jnp.array([X_k[d] / spacing[D - 1 - d] for d in range(D)])
        i0  = jnp.floor(idx).astype(jnp.int32)   # (D,) base index

        # generate stencil offsets: range(-support+1, support+1)
        # For support=2: offsets = [-1, 0, 1, 2]  (4 points)
        offsets = jnp.arange(-support + 1, support + 1)   # (2*support,)

        # Build all stencil multi-indices via meshgrid
        if D == 2:
            oy, ox = jnp.meshgrid(offsets, offsets, indexing="ij")  # (S,S)
            iy = jnp.mod(i0[1] + oy, grid.NY)
            ix = jnp.mod(i0[0] + ox, grid.NX)

            # physical position of each stencil point
            xy = jnp.stack([
                ix.astype(float) * spacing[-1],   # x
                iy.astype(float) * spacing[-2],   # y
            ], axis=-1)   # (S, S, 2)

            # delta weights
            r = (xy - X_k) / jnp.array(spacing[::-1])   # (S, S, D)
            w = jnp.prod(kernel.phi(r), axis=-1)          # (S, S)

            # weighted sum over stencil: u[iy, ix, :]
            u_stencil = u[iy, ix, :]    # (S, S, D)
            U_k = jnp.sum(w[..., None] * u_stencil, axis=(0, 1))  # (D,)

        elif D == 3:
            oz, oy, ox = jnp.meshgrid(offsets, offsets, offsets, indexing="ij")
            iz = jnp.mod(i0[2] + oz, grid.NZ)
            iy = jnp.mod(i0[1] + oy, grid.NY)
            ix = jnp.mod(i0[0] + ox, grid.NX)

            xyz = jnp.stack([
                ix.astype(float) * spacing[-1],
                iy.astype(float) * spacing[-2],
                iz.astype(float) * spacing[-3],
            ], axis=-1)

            r = (xyz - X_k) / jnp.array(spacing[::-1])
            w = jnp.prod(kernel.phi(r), axis=-1)

            u_stencil = u[iz, iy, ix, :]
            U_k = jnp.sum(w[..., None] * u_stencil, axis=(0, 1, 2))
        else:
            raise ValueError(f"Unsupported ndim={D}")

        return U_k

    # vmap over N markers
    U_L = jax.vmap(interp_one_marker)(body.X)   # (N, D)
    return U_L
