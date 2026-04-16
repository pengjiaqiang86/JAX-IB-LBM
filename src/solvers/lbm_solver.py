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
    lattice:        Lattice,
    grid:           EulerianGrid,
    params:         SimulationParams,
    bcs:            List,
    inner_steps:    int,
    outer_steps:    int,
    external_force: Optional[jnp.ndarray] = None,
    collision:      str = "BGK",
) -> Callable[[FluidState], tuple]:
    """
    Build a JIT-compiled rollout function using funcutils.

    Thin wrapper around funcutils.repeated + funcutils.trajectory.
    For full control over post-processing use those directly:

        step_fn    = funcutils.repeated(lbm_step, steps=inner_steps)
        rollout_fn = jax.jit(funcutils.trajectory(step_fn, outer_steps,
                                                   post_process=my_fn,
                                                   start_with_input=True))
        final_state, history = rollout_fn(initial_state)

    Parameters
    ----------
    inner_steps : LBM steps between recorded snapshots
    outer_steps : number of snapshots to record

    Returns
    -------
    rollout_fn : Callable[[FluidState], tuple]
        rollout_fn(initial_state) -> (final_state, (rho_hist, u_hist))

        rho_hist : (outer_steps, *spatial)
        u_hist   : (outer_steps, *spatial, D)

        frame[0] is the initial state; frame[k] is after k * inner_steps steps.
    """
    from src.core import funcutils

    lbm_step = make_lbm_step(lattice, grid, params, bcs, external_force, collision)

    def post_process(state: FluidState):
        rho, u = compute_macroscopic(state.f, lattice, g=state.g)
        return rho, u

    step_fn    = funcutils.repeated(lbm_step, steps=inner_steps)
    rollout_fn = jax.jit(
        funcutils.trajectory(step_fn, outer_steps,
                             post_process=post_process,
                             start_with_input=True)
    )
    return rollout_fn
