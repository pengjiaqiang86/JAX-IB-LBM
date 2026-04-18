"""
2D flow past a tethered deformable circular membrane.

This example is lighter than the elastically mounted cylinder: the membrane is
anchored around its reference position with marker-wise springs, so it deforms
under the flow but does not drift far downstream.
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
from src.immersed_boundary import LagrangianBody
from src.immersed_boundary.elasticity import linear_spring_tether
from src.postprocess import compute_vorticity

# =============================================================================
# Output directory
# =============================================================================
OUT_DIR = "experiments/output_tethered_circular_membrane"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Domain
# =============================================================================
NX, NY = 220, 100
RADIUS = 10.0
CENTER = (70.0, NY / 2)

grid    = EulerianGrid(shape=(NY, NX), dx=1.0)
lattice = D2Q9
params  = SimulationParams.from_Re(Re=40.0, u_ref=0.015, L_ref=2.0 * RADIUS, lattice=lattice)
print(params)

# =============================================================================
# Boundary conditions
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
# Lagrangian body & elasticity
# =============================================================================
body = LagrangianBody.make_circle(
    center=CENTER,
    radius=RADIUS,
    n_markers=120,
    dx=grid.dx,
)

elasticity = linear_spring_tether(stiffness=0.08)

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
OUTER_STEPS = 50

fsi_step = make_fsi_step(lattice, grid, params, bcs, elasticity)

def fsi_step_combined(carry):
    state, body = carry
    return fsi_step(state, body)

def post_process(carry):
    state, body = carry
    _, u  = compute_macroscopic(state.f, lattice, g=state.g)
    omega = compute_vorticity(u, dx=grid.dx, dy=grid.dy)
    return u, omega, body.X

step_fn    = funcutils.repeated(fsi_step_combined, steps=INNER_STEPS)
rollout_fn = jax.jit(
    funcutils.trajectory(step_fn, OUTER_STEPS,
                         post_process=post_process,
                         start_with_input=True)
)

print(f"Running {INNER_STEPS * OUTER_STEPS} FSI steps ({OUTER_STEPS} records)…")
carry0 = (state0, body)
(final_state, _), (u_hist, omega_hist, X_hist) = rollout_fn(carry0)

# X0 is the reference configuration (fixed)
X0_np  = np.array(body.X0)
frames = []
for rec_idx in range(OUTER_STEPS):
    step_idx = rec_idx * INNER_STEPS
    disp     = np.linalg.norm(np.array(X_hist[rec_idx]) - X0_np, axis=1)
    frames.append({
        "step":  step_idx,
        "u":     np.array(u_hist[rec_idx]),
        "omega": np.array(omega_hist[rec_idx]),
        "X":     np.array(X_hist[rec_idx]),
        "disp":  disp,
    })
    print(f"step {step_idx:5d}  max marker displacement={disp.max():.4f}")

# =============================================================================
# Figure 1 — final field and membrane shape
# =============================================================================
last      = frames[-1]
u_last    = last["u"]
omega_last = last["omega"]
X_last    = last["X"]
extent    = [0, NX, 0, NY]

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
fig.suptitle("Flow Past a Tethered Circular Membrane", fontsize=12)

speed = np.sqrt(u_last[..., 0] ** 2 + u_last[..., 1] ** 2)
im0   = axes[0].imshow(speed, origin="lower", extent=extent, cmap="viridis")
fig.colorbar(im0, ax=axes[0], label="|u|")
axes[0].plot(X_last[:, 0], X_last[:, 1], "w-", lw=2)
axes[0].set_title("Velocity magnitude")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

vlim = np.percentile(np.abs(omega_last), 99) + 1e-12
im1  = axes[1].imshow(omega_last, origin="lower", extent=extent,
                      cmap="RdBu_r", vmin=-vlim, vmax=vlim)
fig.colorbar(im1, ax=axes[1], label="ω")
axes[1].plot(X_last[:, 0], X_last[:, 1], "k-", lw=2)
axes[1].set_title("Vorticity")
axes[1].set_xlabel("x")

fig.savefig(os.path.join(OUT_DIR, "tethered_membrane_fields.png"), dpi=150)
plt.close(fig)

# =============================================================================
# Figure 2 — deformation history
# =============================================================================
steps     = np.array([frame["step"] for frame in frames])
max_disp  = np.array([frame["disp"].max()  for frame in frames])
mean_disp = np.array([frame["disp"].mean() for frame in frames])

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
ax.plot(steps, max_disp,  "b-o", ms=3, label="max displacement")
ax.plot(steps, mean_disp, "r-s", ms=3, label="mean displacement")
ax.set_xlabel("Time step")
ax.set_ylabel("Marker displacement")
ax.set_title("Membrane deformation history")
ax.grid(True, alpha=0.3)
ax.legend()

fig.savefig(os.path.join(OUT_DIR, "tethered_membrane_deformation.png"), dpi=150)
plt.close(fig)

print(f"Saved results to {OUT_DIR}")
