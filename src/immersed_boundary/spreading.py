import jax
import jax.numpy as jnp

from src.core.grid import EulerianGrid
from src.immersed_boundary.geometry import PointCloud2D
from src.immersed_boundary.delta import DeltaKernel, PESKIN_4PT


def spreading(
        forcing: jnp.ndarray,
        body:    PointCloud2D,
        grid:    EulerianGrid,
        kernel:  DeltaKernel = PESKIN_4PT,
) -> jnp.ndarray:
    """
    Spread Lagrangian force density onto the Eulerian grid.

    Parameters
    ----------
    forcing: jnp.ndarray - Solid forcing from direct forcing method. shape (N, 2)
    body   : IBGeometry  —  body.F : (N, D), body.X : (N, D),
                                 body.ds : (N,)
    grid   : EulerianGrid
    kernel : delta kernel (default: Peskin 4-point)

    Returns
    -------
    g : (*spatial, D)   Eulerian body-force density
    """
    if grid.ndim == 2:
        Y, X = grid.meshgrid()
        dy, dx = grid.spacing
    else:
        raise ValueError("IB spreading for 3D body is not implemented.")
    
    def spread_one_point(ib_point, force_k, ds_k):
        x, y = ib_point[0], ib_point[1]
        kernel_weight = kernel.phi((x - X)/dx) * kernel.phi((y - Y)/dy)
        spreaded_f = jnp.einsum("i,jk->jki", force_k, kernel_weight) * ds_k # (Ny, Nx, D)
        return spreaded_f
    
    spread_all_points = jax.vmap(spread_one_point, in_axes=(0, 0, 0))
    spreaded_f = spread_all_points(body.X, forcing, body.ds) # (N, Ny, Nx, D)

    return jnp.sum(spreaded_f, axis=0) # (Ny, Nx, D)
