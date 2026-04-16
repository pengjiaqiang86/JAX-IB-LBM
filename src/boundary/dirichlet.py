"""
Dirichlet boundary conditions via the Zou-He scheme.

Implemented coverage:
  - D2Q9 velocity BCs on west/east faces
  - D2Q9 pressure BCs on west/east faces
  - Spatially and temporally varying profiles via callable u_fn / rho_fn

Reference
---------
Zou & He, Phys. Fluids 9 (1997), 1591-1598.
"""

from typing import Callable, Union, NamedTuple

import jax.numpy as jnp

from src.core.lattice import Lattice
from src.core.grid import EulerianGrid
from src.core.state import FluidState


# ---------------------------------------------------------------------------
# Internal Zou-He implementations (D2Q9)
# ---------------------------------------------------------------------------

def _zou_he_velocity_west_d2q9(
    f: jnp.ndarray,
    u_bc: jnp.ndarray,   # (NY, D)
    lattice: Lattice,
) -> jnp.ndarray:
    """Apply Zou-He velocity BC on the west (x=0) face for D2Q9."""
    ux = u_bc[:, 0]
    uy = u_bc[:, 1]

    f0 = f[:, 0, 0]
    f2 = f[:, 0, 2]; f4 = f[:, 0, 4]
    f3 = f[:, 0, 3]; f6 = f[:, 0, 6]; f7 = f[:, 0, 7]

    rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux)

    f1 = f3 + (2.0 / 3.0) * rho * ux
    f5 = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux + 0.5 * rho * uy
    f8 = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux - 0.5 * rho * uy

    f = f.at[:, 0, 1].set(f1)
    f = f.at[:, 0, 5].set(f5)
    f = f.at[:, 0, 8].set(f8)
    return f


def _zou_he_velocity_east_d2q9(
    f: jnp.ndarray,
    u_bc: jnp.ndarray,   # (NY, D)
    lattice: Lattice,
) -> jnp.ndarray:
    """Apply Zou-He velocity BC on the east (x=-1) face for D2Q9."""
    ux = u_bc[:, 0]
    uy = u_bc[:, 1]

    f0 = f[:, -1, 0]
    f2 = f[:, -1, 2]; f4 = f[:, -1, 4]
    f1 = f[:, -1, 1]; f5 = f[:, -1, 5]; f8 = f[:, -1, 8]

    rho = (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / (1.0 + ux)

    f3 = f1 - (2.0 / 3.0) * rho * ux
    f7 = f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho * ux - 0.5 * rho * uy
    f6 = f8 - 0.5 * (f2 - f4) - (1.0 / 6.0) * rho * ux + 0.5 * rho * uy

    f = f.at[:, -1, 3].set(f3)
    f = f.at[:, -1, 7].set(f7)
    f = f.at[:, -1, 6].set(f6)
    return f


def _zou_he_pressure_east_d2q9(
    f: jnp.ndarray,
    rho_bc: jnp.ndarray,   # (NY,)
    lattice: Lattice,
) -> jnp.ndarray:
    """Apply Zou-He pressure (density) BC on the east face for D2Q9."""
    f0 = f[:, -1, 0]
    f2 = f[:, -1, 2]; f4 = f[:, -1, 4]
    f1 = f[:, -1, 1]; f5 = f[:, -1, 5]; f8 = f[:, -1, 8]

    ux = -1.0 + (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho_bc

    f3 = f1 - (2.0 / 3.0) * rho_bc * ux
    f7 = f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_bc * ux
    f6 = f8 - 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_bc * ux

    f = f.at[:, -1, 3].set(f3)
    f = f.at[:, -1, 7].set(f7)
    f = f.at[:, -1, 6].set(f6)
    return f


def _zou_he_pressure_west_d2q9(
    f: jnp.ndarray,
    rho_bc: jnp.ndarray,   # (NY,)
    lattice: Lattice,
) -> jnp.ndarray:
    """Apply Zou-He pressure (density) BC on the west face for D2Q9."""
    f0 = f[:, 0, 0]
    f2 = f[:, 0, 2]; f4 = f[:, 0, 4]
    f3 = f[:, 0, 3]; f6 = f[:, 0, 6]; f7 = f[:, 0, 7]

    ux = 1.0 - (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / rho_bc

    f1 = f3 + (2.0 / 3.0) * rho_bc * ux
    f5 = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_bc * ux
    f8 = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_bc * ux

    f = f.at[:, 0, 1].set(f1)
    f = f.at[:, 0, 5].set(f5)
    f = f.at[:, 0, 8].set(f8)
    return f


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DirichletVelocityBC(NamedTuple):
    """
    Zou-He velocity Dirichlet BC.

    Parameters
    ----------
    face  : one of "west" | "east"
    u_fn  : Callable[[int], jnp.ndarray]
            Maps time-step t to a velocity array on the face:
              2D west/east : (NY, 2)
            For a uniform inlet:  lambda t: jnp.full((NY, 2), [u_in, 0.0])
            For a parabolic profile use a time-independent function.
    """
    face: str
    u_fn: Callable

    def apply(
        self,
        f:       jnp.ndarray,
        state:   FluidState,
        lattice: Lattice,
        grid:    EulerianGrid,
    ) -> jnp.ndarray:
        u_bc = self.u_fn(state.t)   # evaluated outside JIT trace

        if lattice.D == 2 and lattice.Q == 9:
            if self.face == "west":
                return _zou_he_velocity_west_d2q9(f, u_bc, lattice)
            if self.face == "east":
                return _zou_he_velocity_east_d2q9(f, u_bc, lattice)
        raise NotImplementedError(
            f"DirichletVelocityBC: face='{self.face}' not yet implemented "
            f"for D{lattice.D}Q{lattice.Q}."
        )


class DirichletPressureBC(NamedTuple):
    """
    Zou-He pressure (density) Dirichlet BC.

    Parameters
    ----------
    face   : one of "west" | "east"
    rho_fn : Callable[[int], jnp.ndarray]  or float
             Returns target density on the face.
             Float is broadcast to the full face shape.
    """
    face:   str
    rho_fn: Union[Callable, float]

    def apply(
        self,
        f:       jnp.ndarray,
        state:   FluidState,
        lattice: Lattice,
        grid:    EulerianGrid,
    ) -> jnp.ndarray:
        if callable(self.rho_fn):
            rho_bc = self.rho_fn(state.t)
        else:
            # scalar → broadcast to face length
            face_len = grid.NY if self.face in ("west", "east") else grid.NX
            rho_bc = jnp.full((face_len,), self.rho_fn)

        if lattice.D == 2 and lattice.Q == 9:
            if self.face == "west":
                return _zou_he_pressure_west_d2q9(f, rho_bc, lattice)
            if self.face == "east":
                return _zou_he_pressure_east_d2q9(f, rho_bc, lattice)
        raise NotImplementedError(
            f"DirichletPressureBC: face='{self.face}' not yet implemented "
            f"for D{lattice.D}Q{lattice.Q}."
        )
