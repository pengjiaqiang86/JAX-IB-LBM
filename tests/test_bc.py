"""
Smoke tests for boundary conditions.
"""

import jax.numpy as jnp
import pytest

from src.core.lattice import D2Q9
from src.core.grid import EulerianGrid
from src.core.state import FluidState
from src.core.params import SimulationParams
from src.fluid.equilibrium import compute_equilibrium
from src.boundary import (
    DirichletVelocityBC, NeumannBC, PeriodicBC, BounceBackBC,
)


NY, NX = 20, 30
grid    = EulerianGrid(shape=(NY, NX), dx=1.0)
lattice = D2Q9

rho0 = jnp.ones((NY, NX))
u0   = jnp.zeros((NY, NX, 2)).at[..., 0].set(0.1)
f0   = compute_equilibrium(rho0, u0, lattice)
state0 = FluidState(f=f0, g=None, t=0)


def test_neumann_bc_shape():
    bc  = NeumannBC(face="east")
    f_  = bc.apply(f0, state0, lattice, grid)
    assert f_.shape == f0.shape


def test_neumann_bc_zero_gradient():
    bc = NeumannBC(face="east")
    f_ = bc.apply(f0, state0, lattice, grid)
    assert jnp.allclose(f_[:, -1, :], f_[:, -2, :])


def test_periodic_bc_noop():
    bc = PeriodicBC(axes=(0, 1))
    f_ = bc.apply(f0, state0, lattice, grid)
    assert jnp.allclose(f_, f0)


def test_bounce_back_preserves_fluid():
    mask = jnp.zeros((NY, NX), dtype=bool).at[0, :].set(True)
    bc   = BounceBackBC(solid_mask=mask)
    f_   = bc.apply(f0, state0, lattice, grid)
    # fluid cells (rows 1..NY-1) must be unchanged
    assert jnp.allclose(f_[1:, :, :], f0[1:, :, :])


def test_dirichlet_velocity_bc_shape():
    def inlet(t): return jnp.tile(jnp.array([0.1, 0.0]), (NY, 1))
    bc = DirichletVelocityBC(face="west", u_fn=inlet)
    f_ = bc.apply(f0, state0, lattice, grid)
    assert f_.shape == f0.shape
