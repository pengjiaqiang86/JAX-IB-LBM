from typing import Optional, NamedTuple

import jax.numpy as jnp


class FluidState(NamedTuple):
    """
    Immutable JAX pytree holding the full LBM state through every solver step.

    Parameters
    ----------
    f : jnp.ndarray
        Distribution functions, shape (*spatial, Q).
        Spatial axes are (NZ, NY, NX) for 3D or (NY, NX) for 2D.
    g : jnp.ndarray or None
        Eulerian body-force density, shape (*spatial, D).
        Set to None when no body force is applied (pure BGK without forcing).
        When the IB method is active, g carries the spread Lagrangian force.
    t : int
        Current time-step index (used by time-dependent BCs).
    """

    f: jnp.ndarray
    g: Optional[jnp.ndarray]
    t: int

    # ------------------------------------------------------------------
    # Convenience derived quantities (not stored — recomputed on demand)
    # ------------------------------------------------------------------

    def rho(self) -> jnp.ndarray:
        """Fluid density, shape (*spatial,)."""
        return jnp.sum(self.f, axis=-1)

    def velocity(self, c: jnp.ndarray) -> jnp.ndarray:
        """
        Fluid velocity, shape (*spatial, D).

        Parameters
        ----------
        c : (Q, D) lattice velocity array from the Lattice.
        """
        rho = self.rho()
        # einsum: qd, ...q -> ...d
        u = jnp.einsum("qd,...q->...d", c, self.f) / rho[..., None]
        return u

    def with_f(self, f: jnp.ndarray) -> "FluidState":
        """Return a new FluidState with updated f."""
        return FluidState(f=f, g=self.g, t=self.t)

    def with_g(self, g: Optional[jnp.ndarray]) -> "FluidState":
        """Return a new FluidState with updated body force."""
        return FluidState(f=self.f, g=g, t=self.t)

    def advance(self) -> "FluidState":
        """Return a new FluidState with t incremented by 1 and g cleared."""
        return FluidState(f=self.f, g=None, t=self.t + 1)
