"""
3D Poiseuille channel flow — demonstrating D3Q19 and 3D EulerianGrid.

The only changes from 2D (compared to 2d_cylinder.py):
  - lattice = D3Q19
  - grid    = EulerianGrid(shape=(NZ, NY, NX), dx=1.0)
  - velocity arrays are (*spatial, 3) instead of (*spatial, 2)
Everything else (params, BC API, solver) is identical.
"""

import jax
import jax.numpy as jnp

from src.core import D3Q19, EulerianGrid, FluidState, SimulationParams
from src.boundary import DirichletVelocityBC, NeumannBC, BounceBackBC
from src.fluid import compute_equilibrium
from src.solvers import make_lbm_step

# ── Domain ────────────────────────────────────────────────────────────────
NX, NY, NZ = 60, 40, 40
grid    = EulerianGrid(shape=(NZ, NY, NX), dx=1.0)   # 3D grid
lattice = D3Q19
params  = SimulationParams.from_Re(Re=50, u_ref=0.05, L_ref=float(NY))
print(params)

# ── Solid mask: top/bottom/front/back walls ───────────────────────────────
solid_mask = jnp.zeros((NZ, NY, NX), dtype=bool)
solid_mask = solid_mask.at[0, :, :].set(True)
solid_mask = solid_mask.at[NZ - 1, :, :].set(True)
solid_mask = solid_mask.at[:, 0, :].set(True)
solid_mask = solid_mask.at[:, NY - 1, :].set(True)

# ── BCs ───────────────────────────────────────────────────────────────────
def inlet(t):
    # uniform inlet: u = (u_ref, 0, 0) on the west face, shape (NZ, NY, 3)
    return jnp.tile(jnp.array([params.u_ref, 0.0, 0.0]), (NZ, NY, 1))

bcs = [
    DirichletVelocityBC(face="west", u_fn=inlet),
    NeumannBC(face="east"),
    BounceBackBC(solid_mask=solid_mask),
]

# ── Initial state ─────────────────────────────────────────────────────────
rho0   = jnp.ones((NZ, NY, NX))
u0     = jnp.zeros((NZ, NY, NX, 3)).at[..., 0].set(params.u_ref)
f0     = compute_equilibrium(rho0, u0, lattice)
state0 = FluidState(f=f0, g=None, t=0)

# ── Build step & run ──────────────────────────────────────────────────────
step = make_lbm_step(lattice, grid, params, bcs)

@jax.jit
def run(state):
    def inner(s, _): return step(s), None
    return jax.lax.scan(inner, state, None, length=1000)

print("Compiling…")
final_state, _ = run(state0)
print(f"Done at t={final_state.t}")
