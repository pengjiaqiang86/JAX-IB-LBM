from typing import List, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from src.core.lattice import Lattice
from src.core.grid import EulerianGrid
from src.core.state import FluidState
from src.core.params import SimulationParams
from src.immersed_boundary.markers import LagrangianBody
from src.immersed_boundary.delta import DeltaKernel, PESKIN_4PT
from src.immersed_boundary.ib_step import ib_step
from src.solvers.lbm_solver import make_lbm_step


def make_fsi_step(
    lattice:          Lattice,
    grid:             EulerianGrid,
    params:           SimulationParams,
    bcs:              List,
    elasticity_model: Callable,
    kernel:           DeltaKernel = PESKIN_4PT,
    dt:               float = 1.0,
    external_force:   Optional[jnp.ndarray] = None,
    collision:        str = "BGK",
) -> Callable[[FluidState, LagrangianBody],
              Tuple[FluidState, LagrangianBody]]:
    """
    Build a JIT-compiled FSI step function.

    Coupling sequence
    -----------------
    1. IB step: interpolate u → markers, advance positions,
                compute elastic forces, spread to Eulerian grid (sets state.g)
    2. LBM step: f* = f + Ω + S·Δt
                 where S is built from state.g (IB) + external_force (background)

    Parameters
    ----------
    external_force : optional static body-force array (*spatial, D) or (D,).
                     Combined with the IB spreading force inside the LBM step.
                     See make_lbm_step for details.

    Returns
    -------
    fsi_step : Callable  —  fsi_step(state, body) -> (state_new, body_new)
    """
    lbm_step_fn = make_lbm_step(
        lattice, grid, params, bcs,
        external_force=external_force,
        collision=collision,
    )

    @jax.jit
    def fsi_step(
        state: FluidState,
        body:  LagrangianBody,
    ) -> Tuple[FluidState, LagrangianBody]:
        # IB coupling: spread elastic forces → state.g  (dynamic channel)
        state_with_force, body_new = ib_step(
            state, body, grid, lattice, elasticity_model, kernel, dt
        )
        # LBM advance: Ω + S·Δt where S combines state.g and external_force
        state_new = lbm_step_fn(state_with_force)
        return state_new, body_new

    return fsi_step
