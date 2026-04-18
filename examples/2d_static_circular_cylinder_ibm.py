"""
2D flow past a static circular cylinder using the Immersed Boundary Method.

The cylinder surface is represented by Lagrangian markers.  A direct-forcing
IB scheme (Guo body force) enforces the no-slip condition at the markers each
step, without placing the cylinder in a step-wise bounce-back solid mask.

Key differences from 2d_static_circular_cylinder.py
----------------------------------------------------
  - No solid mask inside the fluid domain for the cylinder.
  - Cylinder geometry lives entirely in Lagrangian space (PointCloud2D).
  - make_fsi_step is used in place of make_lbm_step.
  - A static "do-nothing" solid model is passed (body never moves).

Stability note
--------------
IB direct forcing is stricter than pure BGK: keep omega < 1.8 (tau > 0.56)
to avoid instability from the Guo body-force source term.
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
from src.fluid import compute_equilibrium, compute_macroscopic
from src.solvers import make_fsi_step
from src.immersed_boundary.geometry import PointCloud2D
from src.postprocess import compute_vorticity
from src.utils.export import save_trajectory_netcdf

# =============================================================================
# Output directory
# =============================================================================
OUT_DIR = "experiments/output_static_circular_cylinder_ib"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Domain & fluid parameters
# =============================================================================
NX, NY = 400, 160
RADIUS = 15.0
CYL_X  = 90.0
CYL_Y  = NY / 2

grid    = EulerianGrid(shape=(NY, NX), dx=1.0)
lattice = D2Q9

# Re=40, u_ref=0.04, L_ref=30  →  tau=0.59, omega=1.695  (safe IB margin)
params  = SimulationParams.from_Re(Re=80.0, u_ref=0.04, L_ref=2.0 * RADIUS, lattice=lattice)
print(params)

# =============================================================================
# Lagrangian body
# =============================================================================
# Space markers ~dx/2 apart along the arc to avoid gaps in the kernel stencil.
N_MARKERS = int(4 * np.pi * RADIUS)   # ds_k ≈ 0.5 dx
body = PointCloud2D.make_circle(
    center=(CYL_X, CYL_Y),
    radius=RADIUS,
    n_markers=N_MARKERS,
)

# Static solid model: body never moves, IB no-slip is handled entirely by
# direct forcing inside ib_step (body.V = 0 → force = rho * (0 - u_interp)).
def static_model(body):
    return body

# =============================================================================
# Boundary conditions  (walls only — no bounce-back mask for the cylinder)
# =============================================================================
def inlet(t):
    return jnp.tile(jnp.array([params.u_ref, 0.0]), (NY, 1))

solid_mask = jnp.zeros((NY, NX), dtype=bool)
solid_mask = solid_mask.at[0,      :].set(True)
solid_mask = solid_mask.at[NY - 1, :].set(True)

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
INNER_STEPS = 100
OUTER_STEPS = 400

fsi_step = make_fsi_step(lattice, grid, params, bcs, static_model)

def fsi_step_combined(carry):
    state, body = carry
    return fsi_step(state, body)

def post_process(carry):
    state, body = carry
    rho, u = compute_macroscopic(state.f, lattice, g=state.g)
    return rho, u

step_fn    = funcutils.repeated(fsi_step_combined, steps=INNER_STEPS)
rollout_fn = jax.jit(
    funcutils.trajectory(step_fn, OUTER_STEPS,
                         post_process=post_process,
                         start_with_input=True)
)

print(f"Running {INNER_STEPS * OUTER_STEPS} steps ({OUTER_STEPS} records)…")
carry0 = (state0, body)
(final_state, _), (rho_hist, u_hist) = rollout_fn(carry0)

# =============================================================================
# Post-process final snapshot
# =============================================================================
u_last     = np.array(u_hist[-1])
omega_last = np.array(compute_vorticity(u_hist[-1], dx=grid.dx, dy=grid.dy))
speed      = np.sqrt(u_last[..., 0] ** 2 + u_last[..., 1] ** 2)

X_np   = np.array(body.X)   # (N_MARKERS, 2)
extent = [0, NX, 0, NY]

# Interior mask for visualisation: grey out cells inside the cylinder.
yy, xx = np.mgrid[0:NY, 0:NX]
inside = (xx - CYL_X) ** 2 + (yy - CYL_Y) ** 2 < RADIUS ** 2

# =============================================================================
# Figure 1 — velocity magnitude and vorticity
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
fig.suptitle(f"IB Static Circular Cylinder  Re={params.Re:.0f}", fontsize=12)

speed_plot        = speed.copy()
speed_plot[inside] = np.nan
im0 = axes[0].imshow(speed_plot, origin="lower", extent=extent, cmap="viridis")
fig.colorbar(im0, ax=axes[0], label="|u|")
axes[0].plot(X_np[:, 0], X_np[:, 1], "w-", lw=1.5, label="IB markers")
axes[0].set_title("Velocity magnitude")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

omega_plot         = omega_last.copy()
omega_plot[inside] = np.nan
vlim = np.nanpercentile(np.abs(omega_plot), 99) + 1e-12
im1  = axes[1].imshow(omega_plot, origin="lower", extent=extent,
                      cmap="RdBu_r", vmin=-vlim, vmax=vlim)
fig.colorbar(im1, ax=axes[1], label="ω")
axes[1].plot(X_np[:, 0], X_np[:, 1], "k-", lw=1.5)
axes[1].set_title("Vorticity")
axes[1].set_xlabel("x")

fig.savefig(os.path.join(OUT_DIR, "ib_static_cylinder_fields.png"), dpi=150)
plt.close(fig)

# =============================================================================
# Figure 2 — IB no-slip quality: velocity profile at cylinder surface
# =============================================================================
# Interpolate speed onto the marker ring to check how well no-slip is enforced.
from src.immersed_boundary.interpolation import interpolation as ib_interpolation

u_final_jnp  = u_hist[-1]
u_at_markers = np.array(ib_interpolation(u_final_jnp, body, grid))
speed_markers = np.linalg.norm(u_at_markers, axis=1)

theta = np.linspace(0, 2 * np.pi, N_MARKERS, endpoint=False)

fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
ax.plot(np.degrees(theta), speed_markers, lw=1.5)
ax.axhline(0.0, color="gray", lw=0.8, ls="--", label="exact no-slip")
ax.set_xlabel("Marker angle (°)")
ax.set_ylabel("|u| at marker")
ax.set_title("No-slip enforcement at IB markers (final step)")
ax.set_xlim(0, 360)
ax.set_xticks(range(0, 361, 45))
ax.legend()
ax.grid(True, alpha=0.3)

fig.savefig(os.path.join(OUT_DIR, "ib_static_cylinder_noslip.png"), dpi=150)
plt.close(fig)

print(f"Saved results to {OUT_DIR}")
print(f"Max residual velocity at markers: {speed_markers.max():.4f}  "
      f"(target: 0,  u_ref={params.u_ref})")


# =============================================================================
# Visualization
# =============================================================================
save_trajectory_netcdf(
    {
        "u": u_hist,
        "rho": rho_hist
    },
    grid=grid, path=os.path.join(OUT_DIR, "output.nc"),
    t_vals=jnp.arange(OUTER_STEPS)
)
