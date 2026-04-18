from src.core.state import FluidState
from src.core.grid import EulerianGrid
from src.core.lattice import Lattice
from src.fluid.macroscopic import compute_macroscopic
from src.immersed_boundary.geometry import PointCloud2D
from src.immersed_boundary.delta import DeltaKernel, PESKIN_4PT
from src.immersed_boundary.solid_model import SolidModel, RigidBody
from src.immersed_boundary.interpolation import interpolation
from src.immersed_boundary.spreading import spreading


def ib_step(
        state:       FluidState,
        body:        PointCloud2D,
        grid:        EulerianGrid,
        lattice:     Lattice,
        solid_model: SolidModel,
        kernel:      DeltaKernel = PESKIN_4PT,
        dt:          float = 1.0,
) -> "FluidState":
    """
    One IB-LBM coupling cycle.

    Parameters
    ----------
    state            : current FluidState  (g will be overwritten)
    body             : current IBGeometry
    grid             : EulerianGrid
    lattice          : Lattice
    elasticity_model : Callable (body) -> LagrangianBody  with updated F
    kernel           : Peskin delta kernel
    dt               : time-step size (LBM units, typically 1.0)

    Returns
    -------
    (state_new, body_new) : updated fluid state (g set) and Lagrangian body
    """
    # 1. Get current fluid velocity
    _, u = compute_macroscopic(state.f, lattice, state.g)   # (*spatial, D)

    # 2. Compute velocity difference
    u_interp = interpolation(u, body, grid, kernel)

    # 3. Compute forcing term based on direct-forcing method, and spread force from solid to fluid
    # TODO, fluid density shape (Ny, Nx), velocity shape (N, 2), they cannot multiply
    # Assume uniform density
    # ib_forcing_solid = state.rho() / 1.0 * (body.V - u_interp)
    ib_forcing_solid = state.rho().mean() / 1.0 * (body.V - u_interp)

    # 4. Spread forcing from solid to fluid
    ib_forcing_fluid = spreading(ib_forcing_solid, body, grid, kernel)

    # 5. Add forcing term to LBM step
    state_new = state.with_g(ib_forcing_fluid)

    # 6. Update solid position / velocity / acceleration
    # body_new = body.update_X().update_V()
    # TODO, currently, only static rigid solid is implemented
    body_new = body

    return state_new, body_new
