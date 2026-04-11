from typing import List, Callable, Tuple

import jax

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
    collision:        str = "BGK",
) -> Callable[[FluidState, LagrangianBody],
              Tuple[FluidState, LagrangianBody]]:
    """
    Build a JIT-compiled FSI step function.

    Coupling sequence
    -----------------
    1. IB step: interpolate u → markers, advance positions,
                compute elastic forces, spread to Eulerian grid (sets state.g)
    2. LBM step: collision (with Guo forcing from state.g) + streaming + BCs

    Returns
    -------
    fsi_step : Callable
        fsi_step(state, body) -> (state_new, body_new)
    """
    lbm_step_fn = make_lbm_step(lattice, grid, params, bcs, collision)

    @jax.jit
    def fsi_step(
        state: FluidState,
        body:  LagrangianBody,
    ) -> Tuple[FluidState, LagrangianBody]:
        # IB coupling: spread forces into state.g
        state_with_force, body_new = ib_step(
            state, body, grid, lattice, elasticity_model, kernel, dt
        )
        # LBM advance: uses state.g via Guo forcing
        state_new = lbm_step_fn(state_with_force)
        return state_new, body_new

    return fsi_step
