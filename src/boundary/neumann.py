"""
Neumann (zero normal-gradient) boundary conditions.

Strategies
----------
zero_gradient  : f[:, -1, :] = f[:, -2, :]  (simple copy)
convective     : f_bc^{n+1} = f_bc^n - u_c * dt/dx * (f_bc^n - f_int^n)
                 where u_c is a user-supplied convective velocity.
                 Useful for outflow BCs that prevent reflections.
"""

from typing import NamedTuple, Literal

import jax.numpy as jnp

from src.core.lattice import Lattice
from src.core.grid import EulerianGrid
from src.core.state import FluidState


class NeumannBC(NamedTuple):
    """
    Zero normal-gradient boundary condition.

    Parameters
    ----------
    face     : "west" | "east" | "south" | "north" | "bottom" | "top"
    strategy : "zero_gradient" (default) | "convective"
    u_conv   : convective velocity for strategy="convective" (LBM units).
               Ignored for zero_gradient.
    """
    face:     str
    strategy: Literal["zero_gradient", "convective"] = "zero_gradient"
    u_conv:   float = 0.0

    def apply(
        self,
        f:       jnp.ndarray,
        state:   FluidState,
        lattice: Lattice,
        grid:    EulerianGrid,
    ) -> jnp.ndarray:
        if self.strategy == "zero_gradient":
            return _zero_gradient(f, self.face)
        if self.strategy == "convective":
            return _convective(f, self.face, self.u_conv)
        raise ValueError(f"Unknown Neumann strategy: {self.strategy}")


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

def _zero_gradient(f: jnp.ndarray, face: str) -> jnp.ndarray:
    """Copy the first interior layer to the boundary layer."""
    if face == "east":
        return f.at[:, -1, :].set(f[:, -2, :])
    if face == "west":
        return f.at[:, 0, :].set(f[:, 1, :])
    if face == "north":
        return f.at[-1, :, :].set(f[-2, :, :])
    if face == "south":
        return f.at[0, :, :].set(f[1, :, :])
    if face == "top":       # 3D: axis 0
        return f.at[-1, :, :, :].set(f[-2, :, :, :])
    if face == "bottom":
        return f.at[0, :, :, :].set(f[1, :, :, :])
    raise ValueError(f"Unknown face: {face}")


def _convective(f: jnp.ndarray, face: str, u_conv: float) -> jnp.ndarray:
    """
    First-order upwind convective outflow:
        f_b^{n+1} = f_b^n - u_c * (f_b^n - f_int^n)
    where u_c is dimensionless (lattice units, typically u_in).
    """
    u_c = jnp.clip(u_conv, 0.0, 1.0)   # stability: u_c <= 1 (CFL)
    if face == "east":
        f_new = f[:, -1, :] - u_c * (f[:, -1, :] - f[:, -2, :])
        return f.at[:, -1, :].set(f_new)
    if face == "west":
        f_new = f[:, 0, :] + u_c * (f[:, 1, :] - f[:, 0, :])
        return f.at[:, 0, :].set(f_new)
    if face == "north":
        f_new = f[-1, :, :] - u_c * (f[-1, :, :] - f[-2, :, :])
        return f.at[-1, :, :].set(f_new)
    if face == "south":
        f_new = f[0, :, :] + u_c * (f[1, :, :] - f[0, :, :])
        return f.at[0, :, :].set(f_new)
    raise ValueError(f"Convective BC not implemented for face: {face}")
