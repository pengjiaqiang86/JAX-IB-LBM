"""
2D flow past a square cylinder  —  migrated from temp_LBM.py.

Demonstrates the new API:
  - SimulationParams.from_Re
  - DirichletVelocityBC / NeumannBC / BounceBackBC
  - make_lbm_step + jax.lax.scan trajectory
  - compute_vorticity / solve_streamfunction
"""

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.core import D2Q9, EulerianGrid, FluidState, SimulationParams
from src.boundary import DirichletVelocityBC, NeumannBC, BounceBackBC
from src.boundary.bounce_back import make_solid_mask_rectangle
from src.fluid import compute_equilibrium, compute_macroscopic
from src.solvers import make_lbm_step
from src.postprocess import compute_vorticity, solve_streamfunction

# ── Domain ────────────────────────────────────────────────────────────────
NX, NY    = 180, 80
OBST_SIZE = 20
obst_x    = NX // 3
obst_y    = NY // 2 - OBST_SIZE // 2

grid   = EulerianGrid(shape=(NY, NX), dx=1.0)
lattice = D2Q9
params  = SimulationParams.from_Re(Re=100, u_ref=0.1, L_ref=float(OBST_SIZE))
print(params)

# ── Solid mask ────────────────────────────────────────────────────────────
solid_mask = make_solid_mask_rectangle(
    grid,
    y_start=obst_y, y_end=obst_y + OBST_SIZE,
    x_start=obst_x, x_end=obst_x + OBST_SIZE,
    add_walls=True,
)

# ── Boundary conditions ───────────────────────────────────────────────────
def uniform_inlet(t):
    """Uniform inlet velocity profile at x=0."""
    return jnp.tile(jnp.array([params.u_ref, 0.0]), (NY, 1))

bcs = [
    DirichletVelocityBC(face="west", u_fn=uniform_inlet),
    NeumannBC(face="east", strategy="zero_gradient"),
    BounceBackBC(solid_mask=solid_mask),
]

# ── Initial state ─────────────────────────────────────────────────────────
rho0 = jnp.ones((NY, NX))
u0   = jnp.zeros((NY, NX, 2)).at[..., 0].set(params.u_ref)
f0   = compute_equilibrium(rho0, u0, lattice)
state0 = FluidState(f=f0, g=None, t=0)

# ── Build step function ───────────────────────────────────────────────────
step = make_lbm_step(lattice, grid, params, bcs)

# ── Run with scan ─────────────────────────────────────────────────────────
N_STEPS   = 20_000
RECORD_INT = 1_000
n_records  = N_STEPS // RECORD_INT

@jax.jit
def run_trajectory(state0):
    def one_chunk(state, _):
        def inner(s, _): return step(s), None
        state_new, _ = jax.lax.scan(inner, state, None, length=RECORD_INT)
        rho, u = compute_macroscopic(state_new.f, lattice)
        omega  = compute_vorticity(u, dx=grid.dx, dy=grid.dy)
        return state_new, (rho, u, omega)

    return jax.lax.scan(one_chunk, state0, None, length=n_records)

print("Compiling & running…")
final_state, (rho_hist, u_hist, omega_hist) = run_trajectory(state0)

# ── Post-process last snapshot ────────────────────────────────────────────
rho   = rho_hist[-1]
u     = u_hist[-1]
omega = omega_hist[-1]
psi   = solve_streamfunction(omega, solid_mask)

# ── Plot ──────────────────────────────────────────────────────────────────
from src.utils.viz import plot_field_2d, plot_velocity_2d

plot_field_2d(omega, solid_mask, title="Vorticity", label="ω", save_path="vorticity.png")
plot_field_2d(psi,   solid_mask, title="Streamfunction", label="ψ", save_path="streamfunction.png")
plot_velocity_2d(u,  solid_mask, save_path="velocity.png")
print("Saved vorticity.png, streamfunction.png, velocity.png")
