"""
2D Poiseuille Flow (Pressure-Driven Channel)

Physical setup
--------------
A 2D channel of width H and length L.  A pressure (density) difference
ΔP = ρ_in − ρ_out is imposed between the west and east faces.  The top
and bottom walls are no-slip (bounce-back).

The exact steady-state solution for incompressible Newtonian flow is a
parabolic profile:

    u_x(y) = (ΔP / (2μL)) * y * (H - y)

where μ = ρ * ν = ρ * (tau - 0.5) / 3  (LBM units, cs² = 1/3).

Key features demonstrated
--------------------------
  - DirichletPressureBC at inlet and outlet (Zou-He pressure BC)
  - BounceBackBC for top/bottom no-slip walls
  - Analytical verification: L² error of u_x vs parabolic profile
  - Mass-flux conservation check across three cross-sections

Physics note
------------
In LBM, "pressure" maps to density via P = ρ * cs² = ρ/3.
A pressure difference ΔP = 1/300 corresponds to Δρ = 1/100, giving a
small density variation (compressibility error << 1 %).

The Zou-He scheme [Zou & He, Phys. Fluids 9, 1591 (1997)] reconstructs
the unknown distribution functions at an open boundary using the known
macroscopic density and the existing populations.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from src.core import D2Q9, EulerianGrid, FluidState, funcutils
from src.core.params import SimulationParams
from src.boundary import DirichletPressureBC, BounceBackBC
from src.fluid import compute_equilibrium, compute_macroscopic
from src.solvers import make_lbm_step
from src.postprocess.diagnostics import compute_mass_flux
from src.utils.export import save_trajectory_netcdf

# =============================================================================
# Output directory
# =============================================================================
OUT_DIR = "experiments/output_poiseuille"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Domain & fluid parameters
# =============================================================================
NX, NY  = 400, 80        # long channel: NX >> NY for fully-developed flow
H       = float(NY - 2)  # effective channel width (between wall centres)
grid    = EulerianGrid(shape=(NY, NX), dx=1.0)
lattice = D2Q9

#   Choose tau to set viscosity; then derive Re.
tau = 0.8              # relaxation time (> 0.5 for stability)
nu  = (tau - 0.5) / 3  # kinematic viscosity  ν = (τ - ½) cs²

# Pressure boundary conditions: higher density at inlet, lower at outlet.
# ΔP = cs² * Δρ — keep Δρ small for low compressibility error.
rho_in  = 1.001          # inlet density
rho_out = 1.000          # outlet density
dP      = (rho_in - rho_out) * lattice.cs2   # pressure drop across channel
L_chan  = float(NX)      # channel length in lattice units

# Theoretical maximum velocity (at centreline y = H/2):
#   u_max = dP * H² / (8 * μ * L)   where μ = ρ * ν ≈ ν (for ρ ≈ 1)
u_max_theory = dP * H**2 / (8.0 * nu * L_chan)
Re = u_max_theory * H / nu

print(f"tau={tau:.3f}  nu={nu:.5f}  dP={dP:.5f}")
print(f"Theoretical u_max = {u_max_theory:.5f}")
print(f"Effective Re = {Re:.1f}")
print(f"CFL_max ≈ {u_max_theory:.4f}")

# =============================================================================
# Boundary conditions
# =============================================================================
#   Pressure (Zou-He) at west (inlet) and east (outlet).
#   BounceBack for south (y=0) and north (y=NY-1) walls.
solid_mask = jnp.zeros((NY, NX), dtype=bool)
solid_mask = solid_mask.at[0,      :].set(True)
solid_mask = solid_mask.at[NY - 1, :].set(True)

bcs = [
    # Zou-He pressure inlet/outlet: specify density; velocity is reconstructed
    # internally from the known populations on each face.
    DirichletPressureBC(face="west", rho_fn=rho_in),
    DirichletPressureBC(face="east", rho_fn=rho_out),
    BounceBackBC(solid_mask=solid_mask),
]

# =============================================================================
# Initial state
# =============================================================================
#   Linear pressure gradient to speed up convergence.
#   We drive flow via the pressure BC, so u_ref is only informational.
params = SimulationParams(
    Re=Re, u_ref=u_max_theory, L_ref=H,
    nu=nu, tau=tau, omega=1.0 / tau,
)

x_idx  = jnp.arange(NX)[None, :]           # (1, NX)
rho0   = rho_in - (rho_in - rho_out) * x_idx / float(NX - 1)
rho0   = jnp.broadcast_to(rho0, (NY, NX))
u0     = jnp.zeros((NY, NX, 2))
f0     = compute_equilibrium(rho0, u0, lattice)
state0 = FluidState(f=f0, g=None, t=0)

# =============================================================================
# Rollout
# =============================================================================
INNER_STEPS = 1000
OUTER_STEPS = 50
CONV_TOL    = 1e-7

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

print(f"\nRunning {INNER_STEPS * OUTER_STEPS} steps "
      f"({OUTER_STEPS} records)…")
final_state, (rho_hist, u_hist) = rollout_fn(state0)

rho_hist_np = np.array(rho_hist)
u_hist_np   = np.array(u_hist)

# frame[0] = initial state; convergence checked between consecutive frames
selected_idx = OUTER_STEPS - 1
for rec_idx in range(1, OUTER_STEPS):
    step_idx = rec_idx * INNER_STEPS
    delta = float(np.max(np.abs(u_hist_np[rec_idx] - u_hist_np[rec_idx - 1])))
    print(f"  step {step_idx:6d}  Δu_max={delta:.2e}")
    if delta < CONV_TOL:
        selected_idx = rec_idx
        print("  Converged.")
        break

# =============================================================================
# Analysis — extract fields and compare with analytical solution
# =============================================================================
rho_f  = jnp.asarray(rho_hist_np[selected_idx])
u_f    = jnp.asarray(u_hist_np[selected_idx])
u_np   = np.array(u_f)
rho_np = np.array(rho_f)
mask_np = np.array(solid_mask)

# Derive actual pressure drop from the simulation
dp_sim = float((rho_np[:, 0].mean() - rho_np[:, -1].mean()) * lattice.cs2)
print(f"\nExpected ΔP = {dP:.5f}   Measured ΔP = {dp_sim:.5f}")

# Analytical parabolic profile (using measured dp_sim for fair comparison)
# Channel walls are at y=0 and y=NY-1; effective wall at half-cell offset.
y_cells  = np.arange(NY, dtype=float)
y_half   = 0.5
H_eff    = float(NY - 1) - 2 * y_half   # = NY - 2
y_phys   = y_cells - y_half
u_analytic = (dp_sim / (2.0 * nu * float(NX))) * y_phys * (H_eff - y_phys)
u_analytic = np.where(mask_np[:, NX // 2], 0.0, u_analytic)

# =============================================================================
# Figure 1 — 2D fields
# =============================================================================
mid_x = NX // 2
ux_np = u_np[..., 0]
ext   = [0, NX, 0, NY]

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
fig.suptitle("Poiseuille Channel Flow", fontsize=12)

im0 = axes[0].imshow(ux_np, origin="lower", extent=ext, cmap="viridis")
fig.colorbar(im0, ax=axes[0], label="$u_x$")
axes[0].contour(mask_np, levels=[0.5], colors="k", linewidths=1.5)
axes[0].axvline(mid_x, color="r", ls="--", lw=1, label=f"x={mid_x}")
axes[0].set_title("Streamwise velocity $u_x$")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
axes[0].legend(fontsize=8)

im1 = axes[1].imshow(rho_np, origin="lower", extent=ext, cmap="plasma")
fig.colorbar(im1, ax=axes[1], label="ρ")
axes[1].contour(mask_np, levels=[0.5], colors="k", linewidths=1.5)
axes[1].set_title("Density (∝ pressure)")
axes[1].set_xlabel("x")

fig.savefig(os.path.join(OUT_DIR, "poiseuille_fields.png"), dpi=150)
plt.close(fig)
print("Saved poiseuille_fields.png")

# =============================================================================
# Figure 2 — profile comparison with analytical solution
# =============================================================================
y_fluid = y_cells[1:-1]
ux_sim  = ux_np[1:-1, mid_x]
u_ana_f = u_analytic[1:-1]

l2_err = np.sqrt(np.mean((ux_sim - u_ana_f)**2)) / (np.max(u_ana_f) + 1e-14)
print(f"L2 relative error vs analytic: {l2_err:.4e}")

fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
ax.plot(ux_sim,  y_fluid, "b-",  lw=2, label="LBM")
ax.plot(u_ana_f, y_fluid, "r--", lw=2, label="Parabolic (analytic)")
ax.set_xlabel("$u_x$ (LBM units)")
ax.set_ylabel("y (lattice cells)")
ax.set_title(f"Poiseuille profile at x={mid_x}\nL2 error = {l2_err:.2e}")
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(OUT_DIR, "poiseuille_profile.png"), dpi=150)
plt.close(fig)
print("Saved poiseuille_profile.png")

# =============================================================================
# Mass-flux conservation check
# =============================================================================
x_checks = [NX // 4, NX // 2, 3 * NX // 4]
print("\nMass-flux conservation (should be equal across sections):")
for xi in x_checks:
    q = compute_mass_flux(u_f, rho_f, x_idx=xi, dy=grid.dy)
    print(f"  x={xi:4d}  Q = {q:.6f}")

print("Done.")

# =============================================================================
# Save as NetCDF / VTK file
# =============================================================================
save_trajectory_netcdf(
    {"rho": np.array(rho_hist), "u": np.array(u_hist)},
    grid, os.path.join(OUT_DIR, "output.nc"),
    t_vals=np.arange(OUTER_STEPS),
)
