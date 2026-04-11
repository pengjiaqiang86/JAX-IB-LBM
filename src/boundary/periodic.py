"""
Periodic boundary conditions.

The streaming step uses jnp.roll which is already periodic on all axes,
so PeriodicBC is a metadata/no-op object that:
  1. Documents which axes are periodic (for diagnostics, postprocessing).
  2. Can be queried by the solver to skip non-periodic ghost-cell handling.

No modification to f is performed.
"""

from typing import NamedTuple, Tuple

import jax.numpy as jnp

from src.core.lattice import Lattice
from src.core.grid import EulerianGrid
from src.core.state import FluidState


class PeriodicBC(NamedTuple):
    """
    Marks one or more spatial axes as periodic.

    Parameters
    ----------
    axes : tuple of int
        Spatial axes that are periodic.
        For 2D grids: axis 0 = y-direction, axis 1 = x-direction.
        For 3D grids: axis 0 = z, axis 1 = y, axis 2 = x.

    Examples
    --------
    Fully periodic 2D domain:
        PeriodicBC(axes=(0, 1))

    Periodic in y only (channel flow: periodic streamwise = x, walls in y):
        PeriodicBC(axes=(1,))   # if x is axis 1
    """
    axes: Tuple[int, ...]

    def apply(
        self,
        f:       jnp.ndarray,
        state:   FluidState,
        lattice: Lattice,
        grid:    EulerianGrid,
    ) -> jnp.ndarray:
        # jnp.roll in stream() already handles periodicity.
        # This method intentionally returns f unchanged.
        return f

    def is_periodic(self, axis: int) -> bool:
        """Return True if the given spatial axis is periodic."""
        return axis in self.axes
