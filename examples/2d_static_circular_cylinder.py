"""
2D flow past a static circular cylinder.

Demonstrates:
  - Bounce-back on a circular solid mask
  - Inlet / outlet / wall boundary conditions
  - funcutils.repeated + funcutils.trajectory rollout
  - Final drag / lift estimate from momentum exchange
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from src.core import D2Q9, EulerianGrid, FluidState, SimulationParams, funcutils
from src.boundary import DirichletVelocityBC, NeumannBC, BounceBackBC
from src.boundary.bounce_back import make_solid_mask_circle
from src.fluid import compute_equilibrium, compute_macroscopic
from src.solvers import make_lbm_step
from src.postprocess import compute_vorticity, solve_streamfunction
from src.postprocess.diagnostics import compute_drag_lift
from src.postprocess import save_trajectory_netcdf

# =============================================================================
# Output directory
# =============================================================================
OUT_DIR = "experiments/output_static_circular_cylinder"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Domain
# =============================================================================
NX, NY = 400, 150
RADIUS = 10.0
CYL_X  = 60.0
CYL_Y  = NY / 2

grid    = EulerianGrid(shape=(NY, NX), dx=1.0)
lattice = D2Q9
params  = SimulationParams.from_Re(Re=100.0, u_ref=0.08, L_ref=2.0 * RADIUS, lattice=lattice)
print(params)

# =============================================================================
# Boundary conditions
# =============================================================================
def inlet(t):
    return jnp.tile(jnp.array([params.u_ref, 0.0]), (NY, 1))

solid_mask = make_solid_mask_circle(
    grid,
    center_x=CYL_X,
    center_y=CYL_Y,
    radius=RADIUS,
    add_walls=True,
)

bcs = [
    DirichletVelocityBC(face="west", u_fn=inlet),
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
INNER_STEPS = 200
OUTER_STEPS = 50

lbm_step = make_lbm_step(lattice, grid, params, bcs)

def post_process(state):
    rho, u = compute_macroscopic(state.f, lattice, g=state.g)
    return rho, u

step_fn    = funcutils.repeated(lbm_step, steps=INNER_STEPS)
rollout_fn = jax.jit(
    funcutils.trajectory(step_fn, OUTER_STEPS,
                         post_process=post_process,
                         start_with_input=True)
)

print(f"Running {INNER_STEPS * OUTER_STEPS} steps "
      f"({OUTER_STEPS} records, every {INNER_STEPS} steps)…")
final_state, (rho_hist, u_hist) = rollout_fn(state0)

# =============================================================================
# Post-process
# =============================================================================
rho  = rho_hist[-1]
u    = u_hist[-1]
omega = compute_vorticity(u, dx=grid.dx, dy=grid.dy)
psi   = solve_streamfunction(omega, solid_mask)
drag, lift = compute_drag_lift(final_state.f, solid_mask, lattice)

print(f"Final drag  ≈ {float(drag):.6f}")
print(f"Final lift  ≈ {float(lift):.6f}")

rho_np   = np.array(rho)
u_np     = np.array(u)
omega_np = np.array(omega)
psi_np   = np.array(psi)
mask_np  = np.array(solid_mask)
speed    = np.sqrt(u_np[..., 0] ** 2 + u_np[..., 1] ** 2)
extent   = [0, NX, 0, NY]

# =============================================================================
# Figure 1 — velocity magnitude and vorticity
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
fig.suptitle("Static Circular Cylinder", fontsize=12)

im0 = axes[0].imshow(speed, origin="lower", extent=extent, cmap="viridis")
fig.colorbar(im0, ax=axes[0], label="|u|")
axes[0].contour(mask_np, levels=[0.5], colors="k", linewidths=1.2)
axes[0].set_title("Velocity magnitude")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

vlim = np.percentile(np.abs(omega_np[~mask_np]), 99) + 1e-12
im1  = axes[1].imshow(omega_np, origin="lower", extent=extent,
                      cmap="RdBu_r", vmin=-vlim, vmax=vlim)
fig.colorbar(im1, ax=axes[1], label="ω")
axes[1].contour(mask_np, levels=[0.5], colors="k", linewidths=1.2)
axes[1].set_title("Vorticity")
axes[1].set_xlabel("x")

fig.savefig(os.path.join(OUT_DIR, "static_cylinder_fields.png"), dpi=150)
plt.close(fig)

# =============================================================================
# Figure 2 — streamfunction and density
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

im0 = axes[0].imshow(psi_np, origin="lower", extent=extent, cmap="coolwarm")
fig.colorbar(im0, ax=axes[0], label="ψ")
axes[0].contour(psi_np, levels=30, colors="k", linewidths=0.4,
                origin="lower", extent=extent)
axes[0].contour(mask_np, levels=[0.5], colors="gray", linewidths=1.2)
axes[0].set_title("Streamfunction")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

im1 = axes[1].imshow(rho_np, origin="lower", extent=extent, cmap="plasma")
fig.colorbar(im1, ax=axes[1], label="ρ")
axes[1].contour(mask_np, levels=[0.5], colors="k", linewidths=1.2)
axes[1].set_title("Density")
axes[1].set_xlabel("x")

fig.savefig(os.path.join(OUT_DIR, "static_cylinder_stream_density.png"), dpi=150)
plt.close(fig)

print(f"Saved results to {OUT_DIR}")

# =============================================================================
# Save as NetCDF / VTK file
# =============================================================================
save_trajectory_netcdf(
    {"rho": np.array(rho_hist), "u": np.array(u_hist)},
    grid, os.path.join(OUT_DIR, "output.nc"),
    t_vals=np.arange(OUTER_STEPS),
)
