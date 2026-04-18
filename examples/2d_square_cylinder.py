"""
2D flow past a square cylinder.

Demonstrates:
  - SimulationParams.from_Re
  - DirichletVelocityBC / NeumannBC / BounceBackBC
  - funcutils.repeated + funcutils.trajectory rollout
  - compute_vorticity / solve_streamfunction
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.core import D2Q9, EulerianGrid, FluidState, SimulationParams, funcutils
from src.boundary import DirichletVelocityBC, NeumannBC, BounceBackBC
from src.boundary.bounce_back import make_solid_mask_rectangle
from src.fluid import compute_equilibrium, compute_macroscopic
from src.solvers import make_lbm_step
from src.postprocess import compute_vorticity, solve_streamfunction

# =============================================================================
# Domain
# =============================================================================
NX, NY    = 180, 80
OBST_SIZE = 20
obst_x    = NX // 3
obst_y    = NY // 2 - OBST_SIZE // 2

grid    = EulerianGrid(shape=(NY, NX), dx=1.0)
lattice = D2Q9
params  = SimulationParams.from_Re(Re=100, u_ref=0.1, L_ref=float(OBST_SIZE), lattice=lattice)
print(params)

# =============================================================================
# Solid mask
# =============================================================================
solid_mask = make_solid_mask_rectangle(
    grid,
    y_start=obst_y, y_end=obst_y + OBST_SIZE,
    x_start=obst_x, x_end=obst_x + OBST_SIZE,
    add_walls=True,
)

# =============================================================================
# Boundary conditions
# =============================================================================
def uniform_inlet(t):
    return jnp.tile(jnp.array([params.u_ref, 0.0]), (NY, 1))

bcs = [
    DirichletVelocityBC(face="west", u_fn=uniform_inlet),
    NeumannBC(face="east", strategy="zero_gradient"),
    BounceBackBC(solid_mask=solid_mask),
]

# =============================================================================
# Initial state
# =============================================================================
rho0   = jnp.ones((NY, NX))
u0     = jnp.zeros((NY, NX, 2)).at[..., 0].set(params.u_ref)
f0     = compute_equilibrium(rho0, u0, lattice)
state0 = FluidState(f=f0, g=None, t=0)

# =============================================================================
# Rollout
# =============================================================================
INNER_STEPS = 10
OUTER_STEPS = 100

lbm_step = make_lbm_step(lattice, grid, params, bcs)

def post_process(state):
    rho, u = compute_macroscopic(state.f, lattice, g=state.g)
    omega  = compute_vorticity(u, dx=grid.dx, dy=grid.dy)
    return rho, u, omega

step_fn    = funcutils.repeated(lbm_step, steps=INNER_STEPS)
rollout_fn = jax.jit(
    funcutils.trajectory(step_fn, OUTER_STEPS,
                         post_process=post_process,
                         start_with_input=True)
)

print("Compiling & running…")
final_state, (rho_hist, u_hist, omega_hist) = rollout_fn(state0)

# =============================================================================
# Post-process last snapshot
# =============================================================================
rho   = rho_hist[-1]
u     = u_hist[-1]
omega = omega_hist[-1]
psi   = solve_streamfunction(omega, solid_mask)

# =============================================================================
# Plot
# =============================================================================
from src.utils.viz import plot_field_2d, plot_velocity_2d

plot_field_2d(omega, solid_mask, title="Vorticity",      label="ω", save_path="vorticity.png")
plot_field_2d(psi,   solid_mask, title="Streamfunction", label="ψ", save_path="streamfunction.png")
plot_velocity_2d(u,  solid_mask, save_path="velocity.png")
print("Saved vorticity.png, streamfunction.png, velocity.png")
