from typing import Protocol, runtime_checkable

import jax.numpy as jnp

from src.core.lattice import Lattice
from src.core.grid import EulerianGrid
from src.core.state import FluidState


@runtime_checkable
class BoundaryCondition(Protocol):
    """
    BoundaryCondition protocol.

    Every BC implements a single method:

        apply(f, state, lattice, grid) -> f

    BCs are stateless (or carry only static configuration) so they can be
    captured inside a jax.jit closure without breaking tracing.

    Applying a sequence of BCs:
        import functools
        f = functools.reduce(lambda f_, bc: bc.apply(f_, state, lattice, grid), bcs, f)
    """

    def apply(
        self,
        f:       jnp.ndarray,
        state:   FluidState,
        lattice: Lattice,
        grid:    EulerianGrid,
    ) -> jnp.ndarray:
        """
        Apply the BC to distribution functions f.

        Parameters
        ----------
        f       : (*spatial, Q)  current distribution functions
        state   : FluidState     carries rho, u, g, t for time-dependent BCs
        lattice : Lattice
        grid    : EulerianGrid

        Returns
        -------
        f : (*spatial, Q)  modified distribution functions
        """
        ...


# ---------------------------------------------------------------------------
# Face helpers shared across BC implementations
# ---------------------------------------------------------------------------

FACE_TO_AXIS_SIGN = {
    # face_name : (spatial_axis, slice_near_boundary, slice_interior)
    # Axis indexing: 2D -> axis 0 = y, axis 1 = x
    #                3D -> axis 0 = z, axis 1 = y, axis 2 = x
    "west":   (1,  0,  1,  1),    # x-axis, first column,  second column
    "east":   (1, -1, -2, -1),    # x-axis, last column,   second-to-last
    "south":  (0,  0,  1,  1),    # y-axis, first row,     second row
    "north":  (0, -1, -2, -1),    # y-axis, last row,      second-to-last
    "bottom": (0,  0,  1,  1),    # z-axis (3D only)
    "top":    (0, -1, -2, -1),    # z-axis (3D only)
}


def face_slice(face: str, ndim: int):
    """
    Return index tuples (boundary_idx, interior_idx) along the face axis.
    Works for both 2D and 3D; caller selects the relevant spatial axis.
    """
    mapping = {
        "west":   (slice(None), 0,  slice(None), 1),
        "east":   (slice(None), -1, slice(None), -2),
        "south":  (0,  slice(None), 1,  slice(None)),
        "north":  (-1, slice(None), -2, slice(None)),
        "bottom": (0,  slice(None), slice(None), 1),
        "top":    (-1, slice(None), slice(None), -2),
    }
    return mapping[face]
