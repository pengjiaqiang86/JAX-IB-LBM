"""
2D Lid-Driven Cavity Flow

Physical setup
--------------
A square box of side L filled with fluid.  The top wall (north face) moves
at constant velocity u_lid in the +x direction.  All other walls are
stationary no-slip.  There is no inlet or outlet — the flow is entirely
driven by wall friction.

Key features demonstrated
--------------------------
  - BounceBackBC  with  moving_velocity  for the moving lid
  - BounceBackBC  for stationary walls (encoded in the same solid_mask)
  - Re = u_lid * L / nu  →  SimulationParams.from_Re
  - Convergence check via velocity change between successive frames
  - Comparison with the classical Ghia et al. (1982) benchmark data
    (centreline velocity profiles at Re=100 and Re=400)
  - solve_streamfunction for flow visualisation

Physics
-------
At low Re (≤ 100) the flow has a single primary vortex centred near the
geometric centre.  At Re = 400 a pair of corner eddies appears at the
bottom.  At Re ≥ 1000 the primary vortex drifts toward the centre and
secondary eddies become significant.

Numerical parameters
--------------------
  - D2Q9, BGK collision, bounce-back walls
  - CFL << 1 guaranteed by u_lid ≤ 0.05 (LBM Mach << 0.1)
  - tau controlled by Re: tau = 0.5 + L * u_lid / (Re * cs2)
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from src.core import D2Q9, EulerianGrid, FluidState, SimulationParams, funcutils
from src.boundary import BounceBackBC
from src.fluid import compute_equilibrium, compute_macroscopic
from src.solvers import make_lbm_step
from src.postprocess import compute_vorticity, solve_streamfunction, compute_cfl
from src.utils.export import save_trajectory_netcdf

# =============================================================================
# Output directory
# =============================================================================
OUT_DIR = "experiments/output_cavity"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Parameters
# =============================================================================
N     = 200          # grid points per side (L = N-1 in LBM units)
u_lid = 0.05        # lid speed (LBM units).  Keep << cs = 1/sqrt(3) ≈ 0.577
RE    = 400         # Reynolds number  Re = u_lid * (N-1) / nu

NX = NY = N
L_ref   = float(N - 1)   # cavity side length in lattice units

grid    = EulerianGrid(shape=(NY, NX), dx=1.0)
lattice = D2Q9
params  = SimulationParams.from_Re(Re=RE, u_ref=u_lid, L_ref=L_ref, lattice=lattice)
print(params)
print(f"CFL_max ≈ {u_lid:.3f}  (should be << 1)")

# =============================================================================
# Solid mask & moving-lid velocity
# =============================================================================
#   True  = solid (bounce-back node)
#   The lid is the north wall (row NY-1) — it is solid too, but we give it
#   a non-zero moving_velocity so BounceBackBC adds the momentum correction:
#       Δf_q = 2 * w_q * ρ * (c_q · u_wall) / cs²
solid_mask = jnp.zeros((NY, NX), dtype=bool)
solid_mask = solid_mask.at[0,      :].set(True)   # south wall (stationary)
solid_mask = solid_mask.at[NY - 1, :].set(True)   # north wall (moving lid)
solid_mask = solid_mask.at[:,      0].set(True)   # west wall  (stationary)
solid_mask = solid_mask.at[:, NX - 1].set(True)   # east wall  (stationary)

moving_velocity = jnp.zeros((NY, NX, 2))
moving_velocity = moving_velocity.at[NY - 1, :, 0].set(u_lid)

# =============================================================================
# Boundary conditions
# =============================================================================
bcs = [
    BounceBackBC(solid_mask=solid_mask, moving_velocity=moving_velocity),
]

# =============================================================================
# Initial state
# =============================================================================
rho0   = jnp.ones((NY, NX))
u0     = jnp.zeros((NY, NX, 2))
f0     = compute_equilibrium(rho0, u0, lattice)
state0 = FluidState(f=f0, g=None, t=0)

# =============================================================================
# Rollout
# =============================================================================
INNER_STEPS = 100
OUTER_STEPS = 500
CONV_TOL    = 1e-6

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

print(f"Running Re={RE} cavity ({INNER_STEPS * OUTER_STEPS} steps total,"
      f" {OUTER_STEPS} records)…")
final_state, (rho_hist, u_hist) = rollout_fn(state0)

rho_hist_np = np.array(rho_hist)
u_hist_np   = np.array(u_hist)

# frame[0] = initial state; frame[k] = state after k * INNER_STEPS steps
selected_idx = OUTER_STEPS - 1
for rec_idx in range(1, OUTER_STEPS):
    step_idx = rec_idx * INNER_STEPS
    u_cur    = u_hist_np[rec_idx]
    u_prev   = u_hist_np[rec_idx - 1]
    delta    = float(np.max(np.abs(u_cur - u_prev)))
    cfl      = compute_cfl(jnp.asarray(u_cur))
    print(f"  step {step_idx:6d}  Δu_max={delta:.2e}  CFL={cfl:.4f}")
    if delta < CONV_TOL:
        selected_idx = rec_idx
        print("  Converged.")
        break

# =============================================================================
# Final fields
# =============================================================================
rho_f   = jnp.asarray(rho_hist_np[selected_idx])
u_f     = jnp.asarray(u_hist_np[selected_idx])
omega_f = compute_vorticity(u_f, dx=grid.dx, dy=grid.dy)
psi_f   = solve_streamfunction(omega_f, solid_mask)

u_np    = np.array(u_f)
omega_np = np.array(omega_f)
psi_np  = np.array(psi_f)
mask_np = np.array(solid_mask)

# =============================================================================
# Figure 1 — vorticity, streamfunction, velocity magnitude
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
fig.suptitle(f"Lid-Driven Cavity  Re={RE}  ({N}×{N})", fontsize=12)
ext = [0, NX, 0, NY]

vlim = np.percentile(np.abs(omega_np[~mask_np]), 99) + 1e-10
im0  = axes[0].imshow(omega_np, origin="lower", extent=ext,
                      cmap="RdBu_r", vmin=-vlim, vmax=vlim)
axes[0].contour(mask_np, levels=[0.5], colors="k", linewidths=1.5)
fig.colorbar(im0, ax=axes[0], label="ω")
axes[0].set_title("Vorticity")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

im1 = axes[1].imshow(psi_np, origin="lower", extent=ext, cmap="coolwarm")
axes[1].contour(psi_np, levels=30, colors="k", linewidths=0.5,
                origin="lower", extent=ext)
axes[1].contour(mask_np, levels=[0.5], colors="gray", linewidths=1.5)
fig.colorbar(im1, ax=axes[1], label="ψ")
axes[1].set_title("Streamfunction")
axes[1].set_xlabel("x")

speed = np.sqrt(u_np[..., 0]**2 + u_np[..., 1]**2)
im2   = axes[2].imshow(speed, origin="lower", extent=ext, cmap="viridis")
fig.colorbar(im2, ax=axes[2], label="|u|")
axes[2].set_title("Velocity magnitude")
axes[2].set_xlabel("x")

fig.savefig(os.path.join(OUT_DIR, f"cavity_Re{RE}_fields.png"), dpi=150)
plt.close(fig)
print(f"Saved cavity_Re{RE}_fields.png")

# =============================================================================
# Figure 2 — centreline profile vs Ghia (1982) benchmark
# =============================================================================
#
# Ghia et al. (1982) tabulated ux along the vertical centreline (x = L/2)
# and uy along the horizontal centreline (y = L/2) for a unit cavity.
# We normalise our coordinates by L = N-1.
#
# Data below is for Re=100 and Re=400 (two most commonly cited).
# Source: Ghia, Ghia & Shin, J. Comp. Phys. 48, 387-411 (1982).
# y* = y / L,  u*x = ux / u_lid

GHIA_RE100_UX = {  # (y*, u*x) along x = L/2
    0.0000: -0.0000, 0.0547: -0.0372, 0.0625: -0.0419,
    0.0703: -0.0478, 0.1016: -0.0643, 0.1719: -0.1015,
    0.2813: -0.1566, 0.4531: -0.2109, 0.5000: -0.2058,
    0.6172: -0.1364, 0.7344:  0.0033, 0.8516:  0.2315,
    0.9531:  0.6872, 0.9609:  0.7371, 0.9688:  0.7892,
    0.9766:  0.8412, 1.0000:  1.0000,
}
GHIA_RE400_UX = {
    0.0000: -0.0000, 0.0547: -0.0872, 0.0625: -0.1009,
    0.0703: -0.1148, 0.1016: -0.1794, 0.1719: -0.2973,
    0.2813: -0.3829, 0.4531: -0.2781, 0.5000: -0.2186,
    0.6172: -0.0608, 0.7344:  0.2083, 0.8516:  0.5519,
    0.9531:  0.9011, 0.9609:  0.9183, 0.9688:  0.9306,
    0.9766:  0.9382, 1.0000:  1.0000,
}

cx      = NX // 2
y_norm  = np.linspace(0, 1, NY)
ux_norm = u_np[:, cx, 0] / u_lid
ghia    = GHIA_RE400_UX if RE == 400 else GHIA_RE100_UX

fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
ax.plot(ux_norm, y_norm, "b-", lw=2, label="LBM (this run)")
ax.plot(list(ghia.values()), list(ghia.keys()), "ro", ms=6,
        label=f"Ghia 1982 Re={RE}")
ax.axvline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("$u_x / u_{lid}$")
ax.set_ylabel("$y / L$")
ax.set_title(f"Vertical centreline  $u_x$  —  Re={RE}")
ax.legend()
ax.grid(True, alpha=0.3)

fig.savefig(os.path.join(OUT_DIR, f"cavity_Re{RE}_profile.png"), dpi=150)
plt.close(fig)
print(f"Saved cavity_Re{RE}_profile.png")

print("Done.")

# =============================================================================
# Save as NetCDF / VTK file
# =============================================================================
save_trajectory_netcdf(
    {"rho": np.array(rho_hist), "u": np.array(u_hist)},
    grid, os.path.join(OUT_DIR, "output.nc"),
    t_vals=np.arange(OUTER_STEPS),
)
