import jax.numpy as jnp

from src.core.lattice import Lattice


def compute_drag_lift(
    f:          jnp.ndarray,    # (*spatial, Q)
    solid_mask: jnp.ndarray,    # (*spatial,)
    lattice:    Lattice,
) -> tuple:
    """
    Compute drag and lift forces on a solid body via momentum exchange.

    Returns (Fx, Fy) — force components on the solid in LBM units.
    """
    # Find boundary (fluid) nodes adjacent to solid nodes
    # For each direction q, if the neighbour in direction c[q] is solid,
    # the momentum exchange is 2 * f[x, q] * c[q]
    Fx = jnp.zeros(())
    Fy = jnp.zeros(())

    for q in range(lattice.Q):
        cx = int(lattice.c[q, 0])
        cy = int(lattice.c[q, 1])
        # shift solid mask in direction c[q] to find boundary fluid cells
        neigh_solid = jnp.roll(jnp.roll(solid_mask, -cy, axis=0), -cx, axis=1)
        fluid_adj   = (~solid_mask) & neigh_solid
        contribution = 2.0 * f[..., q]
        Fx = Fx + jnp.sum(jnp.where(fluid_adj, contribution * cx, 0.0))
        Fy = Fy + jnp.sum(jnp.where(fluid_adj, contribution * cy, 0.0))

    return Fx, Fy


def compute_cfl(
    u:   jnp.ndarray,   # (*spatial, D)
    dx:  float = 1.0,
    dt:  float = 1.0,
) -> float:
    """
    Maximum CFL number over the domain.  Should remain < 1 for stability.
    """
    speed = jnp.sqrt(jnp.sum(u ** 2, axis=-1))
    return float(jnp.max(speed) * dt / dx)


def compute_mass_flux(
    u:   jnp.ndarray,   # (NY, NX, 2)  2D velocity
    rho: jnp.ndarray,   # (NY, NX)
    x_idx: int,         # column index for the measurement plane
    dy:  float = 1.0,
) -> float:
    """Integrate ρ·ux over a vertical slice at x = x_idx."""
    return float(jnp.sum(rho[:, x_idx] * u[:, x_idx, 0]) * dy)
