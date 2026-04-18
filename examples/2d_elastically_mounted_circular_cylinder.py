"""
2D flow past an elastically mounted circular cylinder.

The cylinder is represented by Lagrangian markers and coupled through the
immersed-boundary solver. A stiff shape-restoring force keeps the circle
nearly rigid, while a softer centre-of-mass spring / damper models the mount.
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

# jax.config.update("jax_disable_jit", True)

# =============================================================================
# Output directory
# =============================================================================
OUT_DIR = "experiments/output_elastic_mounted_cylinder"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Domain
# =============================================================================
NX, NY  = 260, 100
RADIUS  = 10.0
CENTER0 = jnp.array([70.0, NY / 2], dtype=float)

grid    = EulerianGrid(shape=(NY, NX), dx=1.0)
lattice = D2Q9
params  = SimulationParams.from_Re(Re=60.0, u_ref=0.001, L_ref=2.0 * RADIUS, lattice=lattice)
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
    # BounceBackBC(solid_mask=solid_mask),
]

# =============================================================================
# Lagrangian body & elasticity model
# =============================================================================
body = PointCloud2D.make_circle(
    center=tuple(np.array(CENTER0)),
    radius=RADIUS,
    n_markers=120,
)


def mounted_circle_model(
    shape_stiffness: float,
    mount_stiffness: tuple[float, float],
    damping: float,
):
    k_mount = jnp.array(mount_stiffness, dtype=float)

    def model(body):
        center  = jnp.mean(body.X,  axis=0)
        center0 = jnp.mean(body.X0, axis=0)
        v_cm    = jnp.mean(body.V,  axis=0)

        # Shape-restoring: keep markers close to a translated copy of the reference.
        X_rel  = body.X  - center
        X0_rel = body.X0 - center0
        F_shape = -shape_stiffness * (X_rel - X0_rel)

        # Mount spring / damper acts on the centre of mass, distributed uniformly.
        F_mount         = -k_mount * (center - center0) - damping * v_cm
        F_mount_density = F_mount[None, :] / jnp.sum(body.ds)

        return body.with_forces(F_shape + F_mount_density)

    return model


elasticity = mounted_circle_model(
    shape_stiffness=1.0,
    mount_stiffness=(1.5e-3, 8.0e-4),
    damping=2.5e-2,
)

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
OUTER_STEPS = 60

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

frames = []
for rec_idx in range(OUTER_STEPS):
    step_idx = rec_idx * INNER_STEPS
    center   = np.array(jnp.mean(X_hist[rec_idx], axis=0))
    frames.append({
        "step":   step_idx,
        "u":      np.array(u_hist[rec_idx]),
        "omega":  np.array(omega_hist[rec_idx]),
        "X":      np.array(X_hist[rec_idx]),
        "center": center,
    })
    print(f"step {step_idx:5d}  center=({center[0]:.3f}, {center[1]:.3f})")

# =============================================================================
# Figure 1 — final wake and cylinder position
# =============================================================================
last      = frames[-1]
u_last    = last["u"]
omega_last = last["omega"]
X_last    = last["X"]
extent    = [0, NX, 0, NY]

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
fig.suptitle("Elastically Mounted Circular Cylinder", fontsize=12)

speed = np.sqrt(u_last[..., 0] ** 2 + u_last[..., 1] ** 2)
im0   = axes[0].imshow(speed, origin="lower", extent=extent, cmap="viridis")
fig.colorbar(im0, ax=axes[0], label="|u|")
axes[0].plot(X_last[:, 0], X_last[:, 1], "w-", lw=2)
axes[0].plot(X_last[:, 0], X_last[:, 1], "wo", ms=2)
axes[0].set_title("Velocity magnitude")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

vlim = np.percentile(np.abs(omega_last), 99) + 1e-12
im1  = axes[1].imshow(omega_last, origin="lower", extent=extent,
                      cmap="RdBu_r", vmin=-vlim, vmax=vlim)
fig.colorbar(im1, ax=axes[1], label="ω")
axes[1].plot(X_last[:, 0], X_last[:, 1], "k-", lw=2)
axes[1].set_title("Vorticity")
axes[1].set_xlabel("x")

fig.savefig(os.path.join(OUT_DIR, "mounted_cylinder_fields.png"), dpi=150)
plt.close(fig)

# =============================================================================
# Figure 2 — centre-of-mass displacement history
# =============================================================================
steps   = np.array([frame["step"]   for frame in frames])
centers = np.array([frame["center"] for frame in frames])

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

axes[0].plot(steps, centers[:, 0] - float(CENTER0[0]), "b-o", ms=3)
axes[0].axhline(0.0, color="gray", lw=0.8, ls="--")
axes[0].set_xlabel("Time step")
axes[0].set_ylabel("x displacement")
axes[0].set_title("Streamwise displacement")

axes[1].plot(steps, centers[:, 1] - float(CENTER0[1]), "r-o", ms=3)
axes[1].axhline(0.0, color="gray", lw=0.8, ls="--")
axes[1].set_xlabel("Time step")
axes[1].set_ylabel("y displacement")
axes[1].set_title("Cross-stream displacement")

fig.savefig(os.path.join(OUT_DIR, "mounted_cylinder_displacement.png"), dpi=150)
plt.close(fig)

print(f"Saved results to {OUT_DIR}")
