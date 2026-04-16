"""
Convergence / validation tests.

test_poiseuille_profile — checks that 2D channel flow recovers the
    parabolic Poiseuille velocity profile at steady state.
"""

import jax
import jax.numpy as jnp
import pytest

from src.core import D2Q9, EulerianGrid, FluidState, SimulationParams
from src.boundary import BounceBackBC, PeriodicBC
from src.fluid import compute_equilibrium, compute_macroscopic
from src.solvers import make_lbm_step, make_lbm_trajectory


def test_poiseuille_profile():
    """
    Driven channel flow with a constant body force and periodic streamwise BC.
    At steady state the profile should be parabolic:
        ux(y) = G / (2*nu) * y * (H - y)
    Tolerance: L2 error < 5%.
    """
    NX, NY = 64, 40
    H      = NY - 2         # channel height (wall-to-wall, in fluid cells)
    grid   = EulerianGrid(shape=(NY, NX), dx=1.0)
    params = SimulationParams.from_Re(Re=10.0, u_ref=0.01, L_ref=float(H))
    body_force = jnp.array([1.0e-6, 0.0])

    solid_mask = jnp.zeros((NY, NX), dtype=bool)
    solid_mask = solid_mask.at[0, :].set(True)
    solid_mask = solid_mask.at[NY - 1, :].set(True)

    bcs = [
        PeriodicBC(axes=(1,)),
        BounceBackBC(solid_mask=solid_mask),
    ]

    rho0   = jnp.ones((NY, NX))
    u0     = jnp.zeros((NY, NX, 2))
    f0     = compute_equilibrium(rho0, u0, D2Q9)
    state0 = FluidState(f=f0, g=None, t=0)

    step = make_lbm_step(D2Q9, grid, params, bcs, external_force=body_force)

    @jax.jit
    def run(state):
        def inner(s, _): return step(s), None
        return jax.lax.scan(inner, state, None, length=12000)[0]

    final = run(state0)
    _, u  = compute_macroscopic(final.f, D2Q9, final.g)
    ux    = u[1:-1, NX // 2, 0]   # centre column, fluid rows only

    # Analytical Poiseuille: u(y) = G/(2nu) * y*(H-y)
    G  = float(body_force[0])
    y  = jnp.arange(1, H + 1, dtype=float) - 0.5
    ux_analytic = G / (2.0 * params.nu) * y * (H - y)

    rel_err = jnp.linalg.norm(ux - ux_analytic) / jnp.linalg.norm(ux_analytic)
    assert float(rel_err) < 0.05, f"Poiseuille L2 error too large: {rel_err:.4f}"


def test_lbm_trajectory_inner_outer_contract():
    """Trajectory records outer_step snapshots, each after inner_step solves."""
    NX, NY = 8, 6
    grid   = EulerianGrid(shape=(NY, NX), dx=1.0)
    params = SimulationParams.from_Re(Re=10.0, u_ref=0.01, L_ref=float(NY - 2))

    rho0   = jnp.ones((NY, NX))
    u0     = jnp.zeros((NY, NX, 2))
    f0     = compute_equilibrium(rho0, u0, D2Q9)
    state0 = FluidState(f=f0, g=None, t=0)

    run = make_lbm_trajectory(
        D2Q9,
        grid,
        params,
        [PeriodicBC(axes=(0, 1))],
        state0,
        inner_step=3,
        outer_step=4,
    )

    final_state, rho_hist, u_hist = run()
    assert final_state.t == 12
    assert rho_hist.shape == (4, NY, NX)
    assert u_hist.shape == (4, NY, NX, 2)
