from typing import Tuple, Union
from functools import cached_property

import jax.numpy as jnp


class EulerianGrid:
    """
    Uniform Cartesian grid. Immutable after construction.

    Convention
    ----------
    - shape is always *spatial-first*:
        2D:  (NY, NX)
        3D:  (NZ, NY, NX)
    - Physical coordinates: x[i] = i * dx  (origin at 0)
    - distribution function arrays: (*shape, Q)  — Q is always the last axis
    - velocity / force arrays:       (*shape, D)  — D is always the last axis

    Parameters
    ----------
    shape : tuple of int
        (NY, NX) for 2D, (NZ, NY, NX) for 3D.
    dx : float
        Grid spacing.  Assumed isotropic; pass a tuple (dx, dy[, dz])
        for anisotropic grids.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dx: Union[float, Tuple[float, ...]],
    ):
        if len(shape) not in (2, 3):
            raise ValueError(f"shape must be 2- or 3-tuple, got {shape}")
        self._shape = tuple(shape)

        if isinstance(dx, (int, float)):
            self._dx = tuple(float(dx) for _ in shape)
        else:
            if len(dx) != len(shape):
                raise ValueError("len(dx) must match len(shape)")
            self._dx = tuple(float(d) for d in dx)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        """Spatial shape: (NY, NX) or (NZ, NY, NX)."""
        return self._shape

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions (2 or 3)."""
        return len(self._shape)

    @property
    def dx(self) -> float:
        """Grid spacing in the x-direction (last axis)."""
        return self._dx[-1]

    @property
    def dy(self) -> float:
        """Grid spacing in the y-direction (second-to-last axis)."""
        return self._dx[-2]

    @property
    def dz(self) -> float:
        """Grid spacing in the z-direction (first axis, 3D only)."""
        if self.ndim < 3:
            raise AttributeError("dz is only defined for 3D grids")
        return self._dx[0]

    @property
    def spacing(self) -> Tuple[float, ...]:
        """All grid spacings: (dz, dy, dx) or (dy, dx)."""
        return self._dx

    @property
    def NX(self) -> int:
        return self._shape[-1]

    @property
    def NY(self) -> int:
        return self._shape[-2]

    @property
    def NZ(self) -> int:
        if self.ndim < 3:
            raise AttributeError("NZ is only defined for 3D grids")
        return self._shape[0]

    @cached_property
    def cell_volume(self) -> float:
        """Volume (area in 2D) of a single grid cell."""
        vol = 1.0
        for d in self._dx:
            vol *= d
        return vol

    # ------------------------------------------------------------------
    # Coordinate arrays
    # ------------------------------------------------------------------

    def x_coords(self) -> jnp.ndarray:
        """x-coordinates of cell centres, shape (NX,)."""
        return jnp.arange(self.NX) * self.dx

    def y_coords(self) -> jnp.ndarray:
        """y-coordinates of cell centres, shape (NY,)."""
        return jnp.arange(self.NY) * self.dy

    def z_coords(self) -> jnp.ndarray:
        """z-coordinates of cell centres, shape (NZ,). 3D only."""
        return jnp.arange(self.NZ) * self.dz

    def meshgrid(self):
        """
        Returns coordinate arrays broadcastable to (*shape,).
        2D: (Y, X) each shape (NY, NX)
        3D: (Z, Y, X) each shape (NZ, NY, NX)
        """
        if self.ndim == 2:
            Y, X = jnp.meshgrid(self.y_coords(), self.x_coords(), indexing="ij")
            return Y, X
        Z, Y, X = jnp.meshgrid(
            self.z_coords(), self.y_coords(), self.x_coords(), indexing="ij"
        )
        return Z, Y, X

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EulerianGrid(shape={self._shape}, "
            f"dx={self._dx if len(self._dx) > 1 else self._dx[0]})"
        )
