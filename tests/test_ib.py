"""
Smoke tests for the IB layer.
"""

import jax.numpy as jnp
import pytest

from src.core.lattice import D2Q9
from src.core.grid import EulerianGrid
from src.core.state import FluidState
from src.fluid.equilibrium import compute_equilibrium
from src.immersed_boundary import (
    LagrangianBody, PESKIN_4PT,
    ib_velocity_interpolation, ib_force_spreading,
)

NY, NX  = 32, 32
grid    = EulerianGrid(shape=(NY, NX), dx=1.0)
lattice = D2Q9

rho0 = jnp.ones((NY, NX))
u0   = jnp.zeros((NY, NX, 2)).at[..., 0].set(0.1)
f0   = compute_equilibrium(rho0, u0, lattice)
state0 = FluidState(f=f0, g=None, t=0)


def test_circle_markers():
    body = LagrangianBody.make_circle(center=(16.0, 16.0), radius=5.0,
                                      n_markers=32, dx=1.0)
    assert body.X.shape == (32, 2)
    assert body.ds.shape == (32,)


def test_interpolation_shape():
    body  = LagrangianBody.make_circle(center=(16.0, 16.0), radius=5.0,
                                       n_markers=16, dx=1.0)
    U_L   = ib_velocity_interpolation(u0, body, grid, PESKIN_4PT)
    assert U_L.shape == (16, 2)


def test_spreading_shape():
    body  = LagrangianBody.make_circle(center=(16.0, 16.0), radius=5.0,
                                       n_markers=16, dx=1.0)
    F_body = body.with_forces(jnp.ones((16, 2)))
    g = ib_force_spreading(F_body, grid, PESKIN_4PT)
    assert g.shape == (NY, NX, 2)


def test_spreading_conserves_force():
    """Total spread force should equal total Lagrangian force * ds."""
    body  = LagrangianBody.make_circle(center=(16.0, 16.0), radius=5.0,
                                       n_markers=16, dx=1.0)
    F_val = jnp.ones((16, 2))
    F_body = body.with_forces(F_val)
    g = ib_force_spreading(F_body, grid, PESKIN_4PT)

    total_L = jnp.sum(F_val * body.ds[:, None], axis=0)   # (2,)
    total_E = jnp.sum(g, axis=(0, 1)) * (grid.dx * grid.dy)

    # Should be approximately equal (within delta kernel accuracy)
    assert jnp.allclose(total_L, total_E, atol=1e-3), \
        f"Force conservation failed: {total_L} vs {total_E}"
