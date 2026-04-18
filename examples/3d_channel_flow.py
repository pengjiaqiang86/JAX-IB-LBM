"""
3D channel flow — demonstrating D3Q19 and 3D EulerianGrid.

This example uses periodic streamwise boundary conditions plus a constant
body force in the x-direction to drive the flow, while bounce-back enforces
no-slip walls on the other faces.
"""

import os

import jax
import jax.numpy as jnp

from src.core import D3Q19, EulerianGrid, FluidState, SimulationParams, funcutils
from src.boundary import PeriodicBC, BounceBackBC
from src.fluid import compute_equilibrium, compute_macroscopic
from src.solvers import make_lbm_step
from src.utils.export import save_trajectory_netcdf

# =============================================================================
# Output directory
# =============================================================================
OUT_DIR = "experiments/output_3d_channel_flow"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Domain
# =============================================================================
NX, NY, NZ = 60, 40, 40
grid       = EulerianGrid(shape=(NZ, NY, NX), dx=1.0)
lattice    = D3Q19
params     = SimulationParams.from_Re(Re=50, u_ref=0.05, L_ref=float(NY), lattice=lattice)
print(params)
body_force = jnp.array([1.0E-4, 0.0, 0.0])

# =============================================================================
# Solid mask & boundary conditions
# =============================================================================
solid_mask = jnp.zeros((NZ, NY, NX), dtype=bool)
solid_mask = solid_mask.at[0,      :, :].set(True)
solid_mask = solid_mask.at[NZ - 1, :, :].set(True)
solid_mask = solid_mask.at[:,      0, :].set(True)
solid_mask = solid_mask.at[:, NY - 1, :].set(True)

bcs = [
    PeriodicBC(axes=(2,)),          # x-direction periodic
    BounceBackBC(solid_mask=solid_mask),
]

# =============================================================================
# Initial state
# =============================================================================
rho0   = jnp.ones((NZ, NY, NX))
u0     = jnp.zeros((NZ, NY, NX, 3))
f0     = compute_equilibrium(rho0, u0, lattice)
state0 = FluidState(f=f0, g=None, t=0)

# =============================================================================
# Rollout
# =============================================================================
INNER_STEPS = 100
OUTER_STEPS = 200

lbm_step = make_lbm_step(lattice, grid, params, bcs, external_force=body_force)

def post_process(state):
    rho, u = compute_macroscopic(state.f, lattice, g=state.g)
    return rho, u

step_fn    = funcutils.repeated(lbm_step, steps=INNER_STEPS)
rollout_fn = jax.jit(
    funcutils.trajectory(step_fn, OUTER_STEPS,
                         post_process=post_process,
                         start_with_input=True)
)

print("Compiling…")
final_state, (rho_hist, u_hist) = rollout_fn(state0)
print(f"Done at t={final_state.t}")

# =============================================================================
# Visualization
# =============================================================================
save_trajectory_netcdf(
    {
        "rho": rho_hist,
        "u": u_hist
    },
    grid=grid, path=os.path.join(OUT_DIR, "output.nc"),
    t_vals=jnp.arange(OUTER_STEPS)
)
