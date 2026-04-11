"""
2D IB-LBM: elastic beam deflected by channel flow.

Demonstrates:
  - LagrangianBody.make_beam
  - linear_beam elasticity model
  - make_fsi_step
  - Snapshot and animation visualisation
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import jax
import jax.numpy as jnp

from src.core import D2Q9, EulerianGrid, FluidState, SimulationParams
from src.boundary import DirichletVelocityBC, NeumannBC, BounceBackBC
from src.fluid import compute_equilibrium, compute_macroscopic
from src.solvers import make_fsi_step
from src.immersed_boundary import LagrangianBody
from src.immersed_boundary.elasticity import linear_beam
from src.postprocess import compute_vorticity

# ── Output directory ──────────────────────────────────────────────────────
OUT_DIR = "output_beam"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Domain ────────────────────────────────────────────────────────────────
NX, NY  = 240, 120
grid    = EulerianGrid(shape=(NY, NX), dx=1.0)
lattice = D2Q9
# Re=20 → tau=0.65, safely above the SRT instability limit of 0.5
params  = SimulationParams.from_Re(Re=20, u_ref=0.02, L_ref=20.0)
print(params)

# ── Solid mask: top/bottom channel walls ─────────────────────────────────
solid_mask = jnp.zeros((NY, NX), dtype=bool)
solid_mask = solid_mask.at[0, :].set(True)
solid_mask = solid_mask.at[NY - 1, :].set(True)

# ── Boundary conditions ───────────────────────────────────────────────────
def inlet(t):
    return jnp.tile(jnp.array([params.u_ref, 0.0]), (NY, 1))

bcs = [
    DirichletVelocityBC(face="west", u_fn=inlet),
    NeumannBC(face="east"),
    BounceBackBC(solid_mask=solid_mask),
]

# ── Lagrangian beam ───────────────────────────────────────────────────────
beam_x  = float(NX // 3)                       # x = 40
beam_y0 = float(NY // 2 - 10)                  # y = 20
beam_y1 = float(NY // 2 + 10)                  # y = 40

body = LagrangianBody.make_beam(
    x0=(beam_x, beam_y0),
    x1=(beam_x, beam_y1),
    n_markers=100,
    dx=grid.dx,
)

# IB stability bound: stiffness << 160 for this setup (see elasticity.py docstring)
# n_clamped=3 pins the root markers so the beam acts as a cantilever (root fixed, tip free).
# clamp_stiffness should be higher than beam stiffness but still within IB stability.
elasticity = linear_beam(
    stiffness=0.5,
    bending_stiffness=0.05,
    n_clamped=3,
    clamp_stiffness=2.0,
)

# ── Initial state ─────────────────────────────────────────────────────────
rho0   = jnp.ones((NY, NX))
u0     = jnp.zeros((NY, NX, 2)).at[..., 0].set(params.u_ref)
f0     = compute_equilibrium(rho0, u0, lattice)
state0 = FluidState(f=f0, g=None, t=0)

# ── Solver ────────────────────────────────────────────────────────────────
fsi_step = make_fsi_step(lattice, grid, params, bcs, elasticity)

# ── Simulation loop with snapshot collection ─────────────────────────────
N_STEPS       = 5000
RECORD_EVERY  = 200      # save a frame every this many steps

state, body_cur = state0, body
frames = []              # list of (rho, u, omega, beam_X) numpy arrays

for i in range(N_STEPS + 1):
    if i % RECORD_EVERY == 0:
        _, u   = compute_macroscopic(state.f, lattice)
        omega  = compute_vorticity(u, dx=grid.dx, dy=grid.dy)
        frames.append({
            "step":  i,
            "u":     np.array(u),
            "omega": np.array(omega),
            "X":     np.array(body_cur.X),   # (N, 2)  marker positions
        })
        maxF = float(jnp.max(jnp.abs(body_cur.F)))
        print(f"step {i:5d}  max|F|={maxF:.4e}  "
              f"x_tip={float(body_cur.X[-1, 0]):.2f}")

    if i < N_STEPS:
        state, body_cur = fsi_step(state, body_cur)


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _add_beam(ax, X, color="white", lw=2, marker_size=3):
    """Draw the Lagrangian beam on axes ax."""
    ax.plot(X[:, 0], X[:, 1], "-", color=color, lw=lw, zorder=5)
    ax.plot(X[:, 0], X[:, 1], "o", color=color, ms=marker_size, zorder=6)


def _domain_extent():
    """Return (left, right, bottom, top) for imshow extent."""
    return [0, NX, 0, NY]


# ── 1. Steady-state snapshots (last recorded frame) ───────────────────────
last = frames[-1]
u_ss    = last["u"]
omega_ss = last["omega"]
X_ss    = last["X"]
speed_ss = np.sqrt(u_ss[..., 0]**2 + u_ss[..., 1]**2)

fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
fig.suptitle(f"Steady state  (step {last['step']})", fontsize=12)

# Speed
im0 = axes[0].imshow(speed_ss, origin="lower", extent=_domain_extent(),
                     cmap="viridis", vmin=0)
fig.colorbar(im0, ax=axes[0], label="|u|")
_add_beam(axes[0], X_ss)
axes[0].set_title("Velocity magnitude")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

# Vorticity
vlim = np.percentile(np.abs(omega_ss), 98)
im1 = axes[1].imshow(omega_ss, origin="lower", extent=_domain_extent(),
                     cmap="RdBu_r", vmin=-vlim, vmax=vlim)
fig.colorbar(im1, ax=axes[1], label="ω")
_add_beam(axes[1], X_ss, color="k")
axes[1].set_title("Vorticity")
axes[1].set_xlabel("x")

# Ux
im2 = axes[2].imshow(u_ss[..., 0], origin="lower", extent=_domain_extent(),
                     cmap="coolwarm")
fig.colorbar(im2, ax=axes[2], label="ux")
_add_beam(axes[2], X_ss, color="k")
axes[2].set_title("Streamwise velocity  ux")
axes[2].set_xlabel("x")

snap_path = os.path.join(OUT_DIR, "steady_state.png")
fig.savefig(snap_path, dpi=150)
plt.close(fig)
print(f"Saved {snap_path}")


# ── 2. Beam deflection history ────────────────────────────────────────────
steps     = [f["step"] for f in frames]
# tip = last marker; track its x-displacement from the reference position
x_tip_ref = beam_x
x_tips    = [f["X"][-1, 0] - x_tip_ref for f in frames]
y_tips    = [f["X"][-1, 1]              for f in frames]

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

axes[0].plot(steps, x_tips, "b-o", ms=4)
axes[0].axhline(0, color="gray", lw=0.8, ls="--")
axes[0].set_xlabel("Time step"); axes[0].set_ylabel("Tip x-displacement (cells)")
axes[0].set_title("Beam tip x-deflection vs time")

axes[1].plot(steps, y_tips, "r-o", ms=4)
axes[1].axhline(NY / 2, color="gray", lw=0.8, ls="--", label="centreline")
axes[1].set_xlabel("Time step"); axes[1].set_ylabel("Tip y-position (cells)")
axes[1].set_title("Beam tip y-position vs time")
axes[1].legend()

defl_path = os.path.join(OUT_DIR, "beam_deflection.png")
fig.savefig(defl_path, dpi=150)
plt.close(fig)
print(f"Saved {defl_path}")


# ── 3. Animation: vorticity + beam shape evolving over time ───────────────
fig_ani, ax_ani = plt.subplots(figsize=(9, 4.5), constrained_layout=True)

first = frames[0]
vlim  = max(np.percentile(np.abs(f["omega"]) , 98) for f in frames) or 0.01

im_ani  = ax_ani.imshow(first["omega"], origin="lower",
                         extent=_domain_extent(),
                         cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                         animated=True)
fig_ani.colorbar(im_ani, ax=ax_ani, label="ω")
beam_line, = ax_ani.plot([], [], "w-", lw=2, zorder=5)
beam_dots, = ax_ani.plot([], [], "wo", ms=3, zorder=6)
title_ani  = ax_ani.set_title("")
ax_ani.set_xlabel("x"); ax_ani.set_ylabel("y")
ax_ani.set_xlim(0, NX); ax_ani.set_ylim(0, NY)

def _update(frame):
    im_ani.set_data(frame["omega"])
    beam_line.set_data(frame["X"][:, 0], frame["X"][:, 1])
    beam_dots.set_data(frame["X"][:, 0], frame["X"][:, 1])
    title_ani.set_text(f"Vorticity + beam  (step {frame['step']})")
    return im_ani, beam_line, beam_dots, title_ani

ani = animation.FuncAnimation(
    fig_ani, _update, frames=frames,
    interval=80,          # ms between frames
    blit=True,
)

ani_path = os.path.join(OUT_DIR, "beam_animation.gif")
ani.save(ani_path, writer="pillow", fps=12)
plt.close(fig_ani)
print(f"Saved {ani_path}")

print("Done.")
