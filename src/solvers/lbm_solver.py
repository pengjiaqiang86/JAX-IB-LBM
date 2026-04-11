import functools
from typing import List, Callable

import jax
import jax.numpy as jnp

from src.core.lattice import Lattice
from src.core.grid import EulerianGrid
from src.core.state import FluidState
from src.core.params import SimulationParams
from src.fluid.macroscopic import compute_macroscopic
from src.fluid.equilibrium import compute_equilibrium
from src.fluid.streaming import stream
from src.fluid.collision import bgk_collision
from src.forcing.guo import guo_forcing_term
from src.boundary.base import BoundaryCondition


def make_lbm_step(
    lattice:   Lattice,
    grid:      EulerianGrid,
    params:    SimulationParams,
    bcs:       List,                  # List[BoundaryCondition]
    collision: str = "BGK",
) -> Callable[[FluidState], FluidState]:
    """
    Build a JIT-compiled single LBM step function.

    The returned function signature is:
        step(state: FluidState) -> FluidState

    The lattice, grid, params, and bcs are captured in the closure so the
    JAX trace sees only the FluidState pytree as a dynamic argument.

    Parameters
    ----------
    lattice   : Lattice (D2Q9, D3Q19, D3Q27)
    grid      : EulerianGrid
    params    : SimulationParams
    bcs       : list of BoundaryCondition objects (applied left-to-right
                after streaming)
    collision : "BGK" (default) or "MRT" (MRT requires passing M, S)
    """
    if collision != "BGK":
        raise NotImplementedError("Only BGK is currently supported.")

    @jax.jit
    def step(state: FluidState) -> FluidState:
        # --- macroscopic ---
        rho, u = compute_macroscopic(state.f, lattice)

        # --- Guo forcing source term (if body force present) ---
        g_force = None
        if state.g is not None:
            g_force = guo_forcing_term(state.g, u, lattice)
            # Velocity correction for Guo: u_eff = u + g*dt/(2*rho)
            u = u + 0.5 * state.g / rho[..., None]

        # --- equilibrium & collision ---
        feq    = compute_equilibrium(rho, u, lattice)
        f_post = bgk_collision(state.f, feq, params.omega, g_force)

        # --- streaming ---
        f_str = stream(f_post, lattice)

        # --- boundary conditions (applied in order) ---
        f_bc = functools.reduce(
            lambda f_, bc: bc.apply(f_, state, lattice, grid),
            bcs,
            f_str,
        )

        return FluidState(f=f_bc, g=None, t=state.t + 1)

    return step


def make_lbm_trajectory(
    lattice:          Lattice,
    grid:             EulerianGrid,
    params:           SimulationParams,
    bcs:              List,
    initial_state:    FluidState,
    n_steps:          int,
    record_interval:  int,
    collision:        str = "BGK",
) -> Callable:
    """
    Build a JIT-compiled trajectory function using jax.lax.scan.

    Returns a callable with no arguments that, when called, runs the full
    simulation and returns (final_state, snapshots).

    snapshots is a tuple of arrays each with shape (n_records, *spatial):
        (rho_hist, ux_hist, uy_hist)    for 2D
        (rho_hist, ux_hist, uy_hist, uz_hist)  for 3D
    """
    step = make_lbm_step(lattice, grid, params, bcs, collision)
    n_records = n_steps // record_interval

    @jax.jit
    def run():
        def one_chunk(state, _):
            def inner(s, _):
                return step(s), None
            state_new, _ = jax.lax.scan(inner, state, None,
                                        length=record_interval)
            rho, u = compute_macroscopic(state_new.f, lattice)
            return state_new, (rho, u)

        final_state, (rho_hist, u_hist) = jax.lax.scan(
            one_chunk, initial_state, None, length=n_records
        )
        return final_state, rho_hist, u_hist
        # rho_hist : (n_records, *spatial)
        # u_hist   : (n_records, *spatial, D)

    return run
