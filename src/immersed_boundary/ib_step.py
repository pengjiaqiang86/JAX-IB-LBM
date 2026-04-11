"""
IB coupling step: one full Lagrangian-Eulerian interaction cycle.

Order of operations (classic IB / Peskin scheme):
  1. Interpolate Eulerian velocity u^n to markers → U_L^n
  2. Advance marker positions: X^{n+1} = X^n + dt * U_L^n
  3. Compute elastic restoring force F_L = elasticity_model(body^{n+1})
  4. Spread F_L → Eulerian body-force field g
  5. Store g in FluidState so the collision step applies it via Guo forcing.

Returns updated (FluidState, LagrangianBody).
"""

from typing import Callable

import jax.numpy as jnp

from src.core.state import FluidState
from src.core.grid import EulerianGrid
from src.core.lattice import Lattice
from src.immersed_boundary.markers import LagrangianBody
from src.immersed_boundary.delta import DeltaKernel, PESKIN_4PT
from src.immersed_boundary.interpolation import ib_velocity_interpolation
from src.immersed_boundary.spreading import ib_force_spreading
from src.fluid.macroscopic import compute_macroscopic


def ib_step(
    state:             FluidState,
    body:              LagrangianBody,
    grid:              EulerianGrid,
    lattice:           Lattice,
    elasticity_model:  Callable,
    kernel:            DeltaKernel = PESKIN_4PT,
    dt:                float = 1.0,
) -> tuple:
    """
    One IB-LBM coupling cycle.

    Parameters
    ----------
    state            : current FluidState  (g will be overwritten)
    body             : current LagrangianBody
    grid             : EulerianGrid
    lattice          : Lattice
    elasticity_model : Callable (body) -> LagrangianBody  with updated F
    kernel           : Peskin delta kernel
    dt               : time-step size (LBM units, typically 1.0)

    Returns
    -------
    (state_new, body_new) : updated fluid state (g set) and Lagrangian body
    """
    # 1. Get current Eulerian velocity
    _, u = compute_macroscopic(state.f, lattice)   # (*spatial, D)

    # 2. Interpolate u → Lagrangian markers
    U_L = ib_velocity_interpolation(u, body, grid, kernel)   # (N, D)

    # 3. Advance marker positions (forward Euler)
    X_new = body.X + dt * U_L

    # 4. Compute elastic restoring forces at new positions
    body_new = body.with_positions(X_new).with_velocities(U_L)
    body_new = elasticity_model(body_new)

    # 5. Spread Lagrangian forces to Eulerian grid
    g = ib_force_spreading(body_new, grid, kernel)   # (*spatial, D)

    # 6. Attach body force to fluid state
    state_new = state.with_g(g)

    return state_new, body_new
