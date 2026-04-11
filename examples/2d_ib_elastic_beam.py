"""
2D IB-LBM: elastic beam clamped at x=NX//3, deflected by flow.

Demonstrates:
  - LagrangianBody.make_beam
  - linear_beam elasticity model
  - make_fsi_step
"""

import jax
import jax.numpy as jnp

from src.core import D2Q9, EulerianGrid, FluidState, SimulationParams
from src.boundary import DirichletVelocityBC, NeumannBC, BounceBackBC
from src.fluid import compute_equilibrium
from src.solvers import make_fsi_step
from src.immersed_boundary import LagrangianBody
from src.immersed_boundary.elasticity import linear_beam
from src.utils.viz import plot_field_2d

# ── Domain ────────────────────────────────────────────────────────────────
NX, NY = 120, 60
grid    = EulerianGrid(shape=(NY, NX), dx=1.0)
lattice = D2Q9
params  = SimulationParams.from_Re(Re=50, u_ref=0.05, L_ref=20.0)

# ── Solid mask: channel walls only ───────────────────────────────────────
import jax.numpy as jnp
solid_mask = jnp.zeros((NY, NX), dtype=bool)
solid_mask = solid_mask.at[0, :].set(True)
solid_mask = solid_mask.at[NY - 1, :].set(True)

# ── BCs ───────────────────────────────────────────────────────────────────
def inlet(t): return jnp.tile(jnp.array([params.u_ref, 0.0]), (NY, 1))

bcs = [
    DirichletVelocityBC(face="west", u_fn=inlet),
    NeumannBC(face="east"),
    BounceBackBC(solid_mask=solid_mask),
]

# ── Lagrangian beam ───────────────────────────────────────────────────────
beam_x = float(NX // 3)
beam_y0, beam_y1 = float(NY // 2 - 10), float(NY // 2 + 10)
body = LagrangianBody.make_beam(
    x0=(beam_x, beam_y0),
    x1=(beam_x, beam_y1),
    n_markers=40,
    dx=grid.dx,
)

elasticity = linear_beam(stiffness=1e3, bending_stiffness=1e2)

# ── Initial state ─────────────────────────────────────────────────────────
rho0   = jnp.ones((NY, NX))
u0     = jnp.zeros((NY, NX, 2)).at[..., 0].set(params.u_ref)
f0     = compute_equilibrium(rho0, u0, lattice)
state0 = FluidState(f=f0, g=None, t=0)

# ── Solver ───────────────────────────────────────────────────────────────
fsi_step = make_fsi_step(lattice, grid, params, bcs, elasticity)

# ── Run ──────────────────────────────────────────────────────────────────
state, body_cur = state0, body
for i in range(5000):
    state, body_cur = fsi_step(state, body_cur)
    if i % 500 == 0:
        print(f"step {i:5d}  max|F|={float(jnp.max(jnp.abs(body_cur.F))):.4e}")

print("Done.")
