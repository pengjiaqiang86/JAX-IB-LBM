import functools
from typing import List, Callable, Optional

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
    lattice:          Lattice,
    grid:             EulerianGrid,
    params:           SimulationParams,
    bcs:              List,
    external_force:   Optional[jnp.ndarray] = None,
    collision:        str = "BGK",
) -> Callable[[FluidState], FluidState]:
    """
    Build a JIT-compiled single LBM step function.

    The full update follows the literature form:

        f*(x, t) = f(x, t)  +  Ω  +  S·Δt

    where
        Ω   = -ω (f - feq)                  BGK relaxation
        S·Δt = (1 - ω/2) · F_q              Guo source term (body forces)

    F_q is built from the combined body-force field:

        g_total = state.g  +  external_force

    Two independent force channels feed into S·Δt:
      * state.g        — dynamic force set each step by the IB coupling
                         (ib_step spreads Lagrangian forces to the Eulerian grid).
      * external_force — static background force captured in the closure
                         (gravity, imposed pressure gradient, etc.).
                         Shape must broadcast to (*spatial, D).

    Keeping the two channels separate in the code mirrors the physical
    distinction: the IB force changes every step; the external force is
    fixed at solver-build time.  Both contribute to the Guo source term
    via the same (1 - ω/2) · F_q formula.

    Parameters
    ----------
    lattice         : Lattice (D2Q9, D3Q19, D3Q27)
    grid            : EulerianGrid
    params          : SimulationParams
    bcs             : list of BoundaryCondition objects
    external_force  : optional static body-force array, shape (*spatial, D)
                      or (D,) broadcast.  Examples:
                        gravity  = jnp.array([0.0, -1e-4])
                        pg_force = jnp.full((NY, NX, 2), [5e-5, 0.0])
    collision       : "BGK" (default)
    """
    if collision != "BGK":
        raise NotImplementedError("Only BGK is currently supported.")

    @jax.jit
    def step(state: FluidState) -> FluidState:
        # ── Combine the two force channels ───────────────────────────────
        # state.g       : dynamic IB force (None between IB steps)
        # external_force: static background force (None if not set)
        if state.g is not None and external_force is not None:
            g_total = state.g + external_force
        elif external_force is not None:
            g_total = external_force
        else:
            g_total = state.g      # may be None (pure-fluid, no forcing)

        # ── Ω: macroscopic quantities ─────────────────────────────────────
        # Guo velocity correction  u_phys = u_raw + g/(2ρ)  is applied here.
        rho, u = compute_macroscopic(state.f, lattice, g=g_total)

        # ── S·Δt: Guo source term F_q  (built from combined g_total) ─────
        # Kept separate from Ω so the split f* = f + Ω + S·Δt is explicit.
        # bgk_collision applies the (1 - ω/2) prefactor when it adds S·Δt.
        S = None
        if g_total is not None:
            S = guo_forcing_term(g_total, u, lattice)

        # ── f* = f + Ω + S·Δt ────────────────────────────────────────────
        feq    = compute_equilibrium(rho, u, lattice)
        f_post = bgk_collision(state.f, feq, params.omega, S)

        # ── Streaming ─────────────────────────────────────────────────────
        f_str = stream(f_post, lattice)

        # ── Boundary conditions ───────────────────────────────────────────
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
    inner_step:       int,
    outer_step:       int,
    external_force:   Optional[jnp.ndarray] = None,
    collision:        str = "BGK",
) -> Callable:
    """
    Build a JIT-compiled trajectory function using jax.lax.scan.

    Returns a callable with no arguments that, when called, runs the full
    simulation and returns (final_state, snapshots).

    The solver advances the state by `inner_step` numerical steps, records
    one snapshot, and repeats that process `outer_step` times.

    snapshots is a tuple of arrays each with shape (outer_step, *spatial):
        (rho_hist, ux_hist, uy_hist)    for 2D
        (rho_hist, ux_hist, uy_hist, uz_hist)  for 3D
    """
    step = make_lbm_step(lattice, grid, params, bcs, external_force, collision)

    @jax.jit
    def run():
        def one_record(state, _):
            def inner(s, _):
                return step(s), None
            state_new, _ = jax.lax.scan(inner, state, None, length=inner_step)
            rho, u = compute_macroscopic(state_new.f, lattice, state_new.g)
            return state_new, (rho, u)

        final_state, (rho_hist, u_hist) = jax.lax.scan(
            one_record, initial_state, None, length=outer_step
        )
        return final_state, rho_hist, u_hist
        # rho_hist : (outer_step, *spatial)
        # u_hist   : (outer_step, *spatial, D)

    return run
