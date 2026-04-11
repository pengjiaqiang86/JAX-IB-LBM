"""
Bounce-back boundary conditions for no-slip walls.

BounceBackBC  — full bounce-back on an arbitrary boolean solid mask.
                Supports stationary walls and moving walls (lid-driven cavity).

Notes
-----
Full bounce-back: after streaming, the post-collision population that entered
a solid node is reflected back:

    f(x, opp[q], t+1) = f_post(x, q, t)    for solid x

For a moving wall with velocity u_wall, the half-way bounce-back adds a
momentum correction:

    Δf = 2 * w[q] * rho * (c[q] · u_wall) / cs^2
"""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from src.core.lattice import Lattice
from src.core.grid import EulerianGrid
from src.core.state import FluidState


class BounceBackBC(NamedTuple):
    """
    Full bounce-back on a boolean solid mask.

    Parameters
    ----------
    solid_mask      : (*spatial,)   bool — True on solid nodes (walls + obstacles)
    moving_velocity : (*spatial, D) or None
                      Velocity of the wall surface at each solid node.
                      None for stationary walls (no correction term).
                      For a moving-lid cavity pass an array with u_wall on
                      the moving boundary nodes.
    """
    solid_mask:       jnp.ndarray
    moving_velocity:  Optional[jnp.ndarray] = None

    def apply(
        self,
        f:       jnp.ndarray,
        state:   FluidState,
        lattice: Lattice,
        grid:    EulerianGrid,
    ) -> jnp.ndarray:
        return _bounce_back(f, self.solid_mask, lattice, self.moving_velocity)


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

def _bounce_back(
    f:                jnp.ndarray,
    solid_mask:       jnp.ndarray,
    lattice:          Lattice,
    moving_velocity:  Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Vectorised full bounce-back.

    At solid nodes, replace f[..., q] with f[..., opp[q]].
    At fluid nodes, leave f unchanged.
    """
    # f_reflected[..., q] = f[..., opp[q]]
    f_reflected = f[..., lattice.opp]

    if moving_velocity is not None:
        # Momentum correction: 2 * w[q] * rho * (c[q] · u_wall)
        rho = jnp.sum(f, axis=-1)                              # (*spatial,)
        # cu_wall: (*spatial, Q)
        cu_wall = jnp.einsum("qd,...d->...q", lattice.c, moving_velocity)
        correction = 2.0 * lattice.w * rho[..., None] * cu_wall
        f_reflected = f_reflected + correction

    # Apply only on solid nodes (broadcast mask over Q)
    f_out = jnp.where(solid_mask[..., None], f_reflected, f)
    return f_out


def make_solid_mask_rectangle(
    grid:     EulerianGrid,
    y_start:  int,
    y_end:    int,
    x_start:  int,
    x_end:    int,
    add_walls: bool = True,
) -> jnp.ndarray:
    """
    Helper: build a 2D solid mask with top/bottom walls and a rectangular
    obstacle.  Returns bool array of shape (NY, NX).
    """
    NY, NX = grid.NY, grid.NX
    mask = jnp.zeros((NY, NX), dtype=bool)
    if add_walls:
        mask = mask.at[0, :].set(True)
        mask = mask.at[NY - 1, :].set(True)
    mask = mask.at[y_start:y_end, x_start:x_end].set(True)
    return mask
