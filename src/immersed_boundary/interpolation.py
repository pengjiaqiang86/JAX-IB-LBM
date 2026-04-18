import jax
import jax.numpy as jnp

from src.core.grid import EulerianGrid
from src.immersed_boundary.geometry import PointCloud2D
from src.immersed_boundary.delta import DeltaKernel, PESKIN_4PT

def interpolation(
        u:       jnp.ndarray,
        body:    PointCloud2D,
        grid:    EulerianGrid,
        kernel:  DeltaKernel = PESKIN_4PT
):
    """
    Interpolating velocity from fluid to immersed solid boundry.

    Parameters
    ----------
    u      : (*spatial, D)  Eulerian velocity field
    body   : LagrangianBody  with X in physical coords
    grid   : EulerianGrid
    kernel : delta kernel (default: Peskin 4-point)

    Returns
    -------
    U_L : (N, D)  velocity at each marker
    """
    if grid.ndim == 2:
        Y, X = grid.meshgrid()
        dy, dx = grid.spacing
    else:
        raise ValueError("IB interpolation for 3D body is not implemented.")

    def interp_one_point(ib_point):
        x, y = ib_point[0], ib_point[1]
        kernel_weight = jnp.expand_dims(kernel.phi((x - X)/dx) * kernel.phi((y - Y)/dy), -1)
        interped_u = jnp.sum(u * kernel_weight * dx * dy, axis=(0, 1))
        return interped_u

    # Vmap over the first axis
    interp_all_points = jax.vmap(interp_one_point, in_axes=0)
    interped_u = interp_all_points(body.X)

    return interped_u
