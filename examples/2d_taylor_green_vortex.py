"""
2D Taylor-Green Vortex Decay

Physical setup
--------------
A doubly-periodic square domain of side 2π with the initial condition:

    u_x(x, y, 0) =  A * sin(k_x * x) * cos(k_y * y)
    u_y(x, y, 0) = -A * cos(k_x * x) * sin(k_y * y)
    P(x, y, 0)   =  ρ_0 * A² / 4 * (cos(2k_x*x) + cos(2k_y*y))

For k_x = k_y = 1 (single-mode vortex) the physical solution decays as:

    u_x(x, y, t) =  A * exp(-2νt) * sin(x) * cos(y)
    u_y(x, y, t) = -A * exp(-2νt) * cos(x) * sin(y)

Kinetic energy E(t) = A²/2 * exp(-4νt).

Key features demonstrated
--------------------------
  - PeriodicBC  (streaming already periodic; PeriodicBC is a no-op marker)
  - Non-uniform initial condition set via equilibrium
  - Analytical solution comparison for velocity and kinetic energy
  - L2 error vs time to quantify numerical diffusion
  - Kinetic energy decay rate extraction

Physics note
------------
The Taylor-Green vortex is a classical test for viscous solvers because it
has a known closed-form solution.  In LBM the effective viscosity is

    ν_eff = (τ - 0.5) / 3   (BGK, cs² = 1/3)

so matching the analytical decay rate directly validates τ.

The LBM has second-order spatial accuracy; the dominant error at early
times is O(Ma²) (compressibility) and at long times is numerical
diffusion from discretisation.

Coordinate mapping
------------------
The physical domain is [0, 2π) × [0, 2π).
Grid cells are indexed (i, j) with i ∈ [0, NY-1], j ∈ [0, NX-1].
Cell centres sit at x = j * 2π/NX, y = i * 2π/NY.
When compared against the numerical solution, the decay rate is expressed
in lattice time using the corresponding discrete wavenumbers 2π/NX, 2π/NY.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from src.core import D2Q9, EulerianGrid, FluidState, SimulationParams, funcutils
from src.boundary import PeriodicBC
from src.fluid import compute_equilibrium, compute_macroscopic
from src.solvers import make_lbm_step
from src.utils.export import save_trajectory_netcdf

# =============================================================================
# Output directory
# =============================================================================
OUT_DIR = "experiments/output_taylor_green"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Simulation parameters
# =============================================================================
N   = 256              # grid points per side (must be even)
tau = 0.65            # relaxation time (controls viscosity)
A   = 0.05            # velocity amplitude (Ma = A * sqrt(3) ≈ 0.087 << 1)

NX = NY = N
grid    = EulerianGrid(shape=(NY, NX), dx=2.0 * np.pi / NX)
lattice = D2Q9

nu  = (tau - 0.5) / 3.0        # kinematic viscosity (lattice units)
dx  = grid.dx                  # = 2π / N  (physical cell size)

# The domain is [0, 2π)^2 sampled on an N×N lattice, so the mode
# sin(x) cos(y) corresponds to lattice wavenumbers kx = ky = 2π/N.
kx = 2.0 * np.pi / NX
ky = 2.0 * np.pi / NY
amp_decay_rate    = nu * (kx**2 + ky**2)      # velocity amplitude decay
energy_decay_rate = 2.0 * amp_decay_rate       # kinetic energy decay

Re     = A * float(N) / nu
params = SimulationParams.from_Re(Re=Re, u_ref=A, L_ref=float(N), lattice=lattice)
print(f"tau={tau:.3f}  nu={nu:.5f}  A={A:.4f}")
print(f"Effective Re = {Re:.1f}")
print(f"Analytical amplitude decay rate = {amp_decay_rate:.5f} per LBM step")
print(f"Analytical energy decay rate    = {energy_decay_rate:.5f} per LBM step")

# =============================================================================
# Cell-centre coordinates
# =============================================================================
#   y-axis → rows (axis 0), x-axis → columns (axis 1)
j = jnp.arange(NX, dtype=float)   # column indices
i = jnp.arange(NY, dtype=float)   # row indices
x = j * 2.0 * jnp.pi / NX         # (NX,)  x coordinates ∈ [0, 2π)
y = i * 2.0 * jnp.pi / NY         # (NY,)  y coordinates ∈ [0, 2π)
X, Y = jnp.meshgrid(x, y)         # (NY, NX) each

# =============================================================================
# Analytical solution helpers
# =============================================================================
def analytic_velocity(t: float) -> jnp.ndarray:
    """Return (NY, NX, 2) velocity at time t (LBM steps)."""
    factor = A * jnp.exp(-amp_decay_rate * t)
    ux =  factor * jnp.sin(X) * jnp.cos(Y)
    uy = -factor * jnp.cos(X) * jnp.sin(Y)
    return jnp.stack([ux, uy], axis=-1)

def analytic_energy(t: float) -> float:
    """Total kinetic energy summed over cells."""
    return 0.5 * A**2 * float(N**2) * float(jnp.exp(-energy_decay_rate * t))

# =============================================================================
# Initial state
# =============================================================================
#   Set f = f_eq(ρ₀, u₀) where u₀ matches the Taylor-Green initial velocity.
#   Pressure initial condition:
#       P = P₀ + ρ₀ * A² / 4 * (cos(2x) + cos(2y))
#   In LBM: ρ = P / cs²
rho0_pressure = 1.0 + (A**2 / (4.0 * lattice.cs2)) * (jnp.cos(2 * X) + jnp.cos(2 * Y))
u0            = analytic_velocity(0.0)
f0            = compute_equilibrium(rho0_pressure, u0, lattice)
state0        = FluidState(f=f0, g=None, t=0)

# =============================================================================
# Boundary conditions
# =============================================================================
#   Streaming with jnp.roll is already periodic; PeriodicBC is a marker only.
bcs = [PeriodicBC(axes=(0, 1))]

# =============================================================================
# Rollout
# =============================================================================
INNER_STEPS = 50
OUTER_STEPS = 100

lbm_step = make_lbm_step(lattice, grid, params, bcs)

def post_process(state):
    _, u = compute_macroscopic(state.f, lattice, g=state.g)
    return u                  # record velocity only; energy computed from this

step_fn    = funcutils.repeated(lbm_step, steps=INNER_STEPS)
rollout_fn = jax.jit(
    funcutils.trajectory(step_fn, OUTER_STEPS,
                         post_process=post_process,
                         start_with_input=True)
)

print(f"Running {INNER_STEPS * OUTER_STEPS} steps "
      f"({OUTER_STEPS} recorded intervals, every {INNER_STEPS} steps)…")
final_state, u_hist = rollout_fn(state0)
# u_hist : (OUTER_STEPS, NY, NX, 2)
# frame[k] is after k * INNER_STEPS steps; frame[0] = initial state

# =============================================================================
# Analysis — L2 error and energy decay
# =============================================================================
t_vals    = []
E_lbm     = []
E_exact   = []
l2_errors = []

for rec_idx in range(OUTER_STEPS):
    step_idx = rec_idx * INNER_STEPS
    u_cur    = u_hist[rec_idx]
    u_ana    = analytic_velocity(float(step_idx))

    E_lbm_i = 0.5 * float(jnp.sum(u_cur[..., 0]**2 + u_cur[..., 1]**2))
    E_ana_i = analytic_energy(float(step_idx))
    err     = float(jnp.sqrt(jnp.mean((u_cur - u_ana)**2)))

    t_vals.append(step_idx)
    E_lbm.append(E_lbm_i)
    E_exact.append(E_ana_i)
    l2_errors.append(err / A)

    if rec_idx % 5 == 0:
        print(f"  step {step_idx:5d}  E_lbm={E_lbm_i:.4e}  "
              f"E_exact={E_ana_i:.4e}  L2_err={err/A:.4e}")

# =============================================================================
# Figure 1 — final vorticity comparison
# =============================================================================
T_final     = INNER_STEPS * OUTER_STEPS
u_final     = u_hist[-1]
u_ana_final = analytic_velocity(float(T_final))

uy_np = np.array(u_final[..., 1])
ux_np = np.array(u_final[..., 0])
# Vorticity ω = ∂u_y/∂x - ∂u_x/∂y  (central differences, periodic wrap)
omega_lbm = (np.roll(uy_np, -1, axis=1) - np.roll(uy_np, 1, axis=1)) / (2 * dx) \
          - (np.roll(ux_np, -1, axis=0) - np.roll(ux_np, 1, axis=0)) / (2 * dx)

omega_ana_np = np.array(
    -2.0 * A * jnp.exp(-amp_decay_rate * T_final)
    * jnp.sin(X) * jnp.sin(Y)
)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
fig.suptitle(f"Taylor-Green Vortex at t={T_final}  (τ={tau}, N={N}×{N})",
             fontsize=11)

vlim = float(np.max(np.abs(omega_ana_np))) * 1.05
kw   = dict(origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim)

im0 = axes[0].imshow(omega_lbm, **kw)
fig.colorbar(im0, ax=axes[0], label="ω")
axes[0].set_title("LBM vorticity")
axes[0].set_xlabel("x cell"); axes[0].set_ylabel("y cell")

im1 = axes[1].imshow(omega_ana_np, **kw)
fig.colorbar(im1, ax=axes[1], label="ω")
axes[1].set_title("Analytic vorticity")
axes[1].set_xlabel("x cell")

err_field = omega_lbm - omega_ana_np
elim = float(np.max(np.abs(err_field))) + 1e-14
im2  = axes[2].imshow(err_field, origin="lower", cmap="PiYG",
                      vmin=-elim, vmax=elim)
fig.colorbar(im2, ax=axes[2], label="error")
axes[2].set_title("Vorticity error (LBM − analytic)")
axes[2].set_xlabel("x cell")

fig.savefig(os.path.join(OUT_DIR, "tgv_vorticity.png"), dpi=150)
plt.close(fig)
print("Saved tgv_vorticity.png")

# =============================================================================
# Figure 2 — energy decay and L2 velocity error
# =============================================================================
t_arr       = np.array(t_vals, dtype=float)
E_lbm_arr   = np.array(E_lbm)
E_exact_arr = np.array(E_exact)

# Fit decay rate from LBM data (log-linear regression)
# E(t) = E0 * exp(-Γ * t)  →  log E = log E0 - Γ t
log_E    = np.log(E_lbm_arr + 1e-20)
coeffs   = np.polyfit(t_arr, log_E, 1)
gamma_fit = -coeffs[0]
print(f"\nFitted decay rate: Γ_fit = {gamma_fit:.5f}")
print(f"Theoretical rate:  Γ_exact = {energy_decay_rate:.5f}")
print(f"Relative error: {abs(gamma_fit - energy_decay_rate)/energy_decay_rate * 100:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

E_fit = np.exp(coeffs[1]) * np.exp(coeffs[0] * t_arr)
axes[0].semilogy(t_arr, E_lbm_arr,   "b-",  lw=2, label="LBM")
axes[0].semilogy(t_arr, E_exact_arr, "r--", lw=2, label="Analytic")
axes[0].semilogy(t_arr, E_fit, "g:", lw=1.5,
                 label=f"LBM fit  Γ={gamma_fit:.4f}")
axes[0].set_xlabel("Time step t")
axes[0].set_ylabel("Kinetic energy E(t)")
axes[0].set_title("Energy decay  (log scale)")
axes[0].legend()
axes[0].grid(True, which="both", alpha=0.3)

axes[1].semilogy(t_arr, l2_errors, "m-o", ms=4, lw=1.5)
axes[1].set_xlabel("Time step t")
axes[1].set_ylabel("$‖u - u_{exact}‖_2 / A$")
axes[1].set_title("Normalised L2 velocity error")
axes[1].grid(True, which="both", alpha=0.3)

fig.savefig(os.path.join(OUT_DIR, "tgv_energy_decay.png"), dpi=150)
plt.close(fig)
print("Saved tgv_energy_decay.png")

print("Done.")

# =============================================================================
# Visualization
# =============================================================================
save_trajectory_netcdf(
    {
        "u": u_hist
    },
    grid=grid, path=os.path.join(OUT_DIR, "output.nc"),
    t_vals=jnp.arange(OUTER_STEPS)
)
