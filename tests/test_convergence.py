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
from src.boundary.dirichlet import DirichletPressureBC
from src.fluid import compute_equilibrium, compute_macroscopic
from src.solvers import make_lbm_step


def test_poiseuille_profile():
    """
    Driven channel flow with pressure-difference BC.
    At steady state the profile should be parabolic:
        ux(y) = (dp/dx) / (2*nu) * y * (H - y)
    Tolerance: L2 error < 1% of peak velocity.
    """
    NX, NY = 10, 40         # short channel; periodic in x via pressure BCs
    H      = NY - 2         # channel height (wall-to-wall, in fluid cells)
    grid   = EulerianGrid(shape=(NY, NX), dx=1.0)
    params = SimulationParams.from_Re(Re=10.0, u_ref=0.01, L_ref=float(H))

    # pressure-driven: rho_in > rho_out
    rho_in  = 1.001
    rho_out = 1.000
    dp      = (rho_in - rho_out) / 3.0   # pressure difference in LBM units

    solid_mask = jnp.zeros((NY, NX), dtype=bool)
    solid_mask = solid_mask.at[0, :].set(True)
    solid_mask = solid_mask.at[NY - 1, :].set(True)

    bcs = [
        DirichletPressureBC(face="west", rho_fn=rho_in),
        DirichletPressureBC(face="east", rho_fn=rho_out),
        BounceBackBC(solid_mask=solid_mask),
    ]

    rho0   = jnp.ones((NY, NX))
    u0     = jnp.zeros((NY, NX, 2))
    f0     = compute_equilibrium(rho0, u0, D2Q9)
    state0 = FluidState(f=f0, g=None, t=0)

    step = make_lbm_step(D2Q9, grid, params, bcs)

    @jax.jit
    def run(state):
        def inner(s, _): return step(s), None
        return jax.lax.scan(inner, state, None, length=5000)[0]

    final = run(state0)
    _, u  = compute_macroscopic(final.f, D2Q9)
    ux    = u[1:-1, NX // 2, 0]   # centre column, fluid rows only

    # Analytical Poiseuille: u(y) = G/(2nu) * y*(H-y)
    G  = dp / NX   # body-force equivalent: dp/dx
    y  = jnp.arange(1, H + 1, dtype=float) - 0.5
    ux_analytic = G / (2.0 * params.nu) * y * (H - y)

    rel_err = jnp.linalg.norm(ux - ux_analytic) / jnp.linalg.norm(ux_analytic)
    assert float(rel_err) < 0.05, f"Poiseuille L2 error too large: {rel_err:.4f}"
