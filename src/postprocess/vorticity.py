import jax.numpy as jnp


def compute_vorticity(
    u:  jnp.ndarray,    # (*spatial, D)  velocity field
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
) -> jnp.ndarray:
    """
    Compute vorticity using 2nd-order central differences.

    Parameters
    ----------
    u  : (*spatial, D)  velocity field
       2D: (NY, NX, 2),  components (ux, uy)
       3D: (NZ, NY, NX, 3), components (ux, uy, uz)
    dx, dy, dz : grid spacings

    Returns
    -------
    2D: (NY, NX)    scalar vorticity  ω = ∂uy/∂x - ∂ux/∂y
    3D: (NZ, NY, NX, 3)  vorticity vector (ωx, ωy, ωz)
    """
    ndim = u.ndim - 1   # spatial dimensions

    if ndim == 2:
        ux = u[..., 0]   # (NY, NX)
        uy = u[..., 1]   # (NY, NX)
        # ∂uy/∂x:  roll along x-axis (axis 1)
        duy_dx = (jnp.roll(uy, -1, axis=1) - jnp.roll(uy, 1, axis=1)) / (2.0 * dx)
        # ∂ux/∂y:  roll along y-axis (axis 0)
        dux_dy = (jnp.roll(ux, -1, axis=0) - jnp.roll(ux, 1, axis=0)) / (2.0 * dy)
        return duy_dx - dux_dy   # (NY, NX)

    elif ndim == 3:
        ux = u[..., 0]   # (NZ, NY, NX)
        uy = u[..., 1]
        uz = u[..., 2]

        def cd(f, axis, h):
            return (jnp.roll(f, -1, axis=axis) - jnp.roll(f, 1, axis=axis)) / (2.0 * h)

        # ωx = ∂uz/∂y - ∂uy/∂z
        ox = cd(uz, axis=1, h=dy) - cd(uy, axis=0, h=dz)
        # ωy = ∂ux/∂z - ∂uz/∂x
        oy = cd(ux, axis=0, h=dz) - cd(uz, axis=2, h=dx)
        # ωz = ∂uy/∂x - ∂ux/∂y
        oz = cd(uy, axis=2, h=dx) - cd(ux, axis=1, h=dy)

        return jnp.stack([ox, oy, oz], axis=-1)   # (NZ, NY, NX, 3)

    else:
        raise ValueError(f"compute_vorticity expects 2D or 3D velocity, got ndim={ndim}")
