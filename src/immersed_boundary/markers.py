"""
LagrangianBody: JAX pytree for an immersed elastic structure.

Each body is a set of N Lagrangian marker points in D-dimensional space.
The pytree carries:
  X   — current positions
  X0  — reference (stress-free) positions
  V   — marker velocities
  F   — computed Lagrangian force density (written by elasticity models)
  ds  — arc-length / area element per marker (used in spreading)

All arrays use physical (not index) coordinates so that the delta-function
spreading works correctly when dx != 1.
"""

from typing import NamedTuple, Optional

import jax.numpy as jnp
import numpy as np


class LagrangianBody(NamedTuple):
    """
    Lagrangian marker point set.

    Shapes
    ------
    X, X0, V, F : (N, D)
    ds          : (N,)
    """
    X:  jnp.ndarray   # (N, D)  current positions (physical coords)
    X0: jnp.ndarray   # (N, D)  reference positions
    V:  jnp.ndarray   # (N, D)  marker velocities
    F:  jnp.ndarray   # (N, D)  Lagrangian force density
    ds: jnp.ndarray   # (N,)    arc-length / area element

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def make_circle(
        center: tuple,
        radius: float,
        n_markers: int,
        dx: float,
    ) -> "LagrangianBody":
        """
        Uniformly spaced markers on a circle.

        Parameters
        ----------
        center    : (cx, cy) in physical coordinates
        radius    : circle radius (physical)
        n_markers : number of markers
        dx        : Eulerian grid spacing (used to set ds)
        """
        theta = jnp.linspace(0, 2 * jnp.pi, n_markers, endpoint=False)
        cx, cy = center
        X = jnp.stack([
            cx + radius * jnp.cos(theta),
            cy + radius * jnp.sin(theta),
        ], axis=1)   # (N, 2)
        ds = jnp.full((n_markers,), 2 * jnp.pi * radius / n_markers)
        return LagrangianBody(
            X=X, X0=X,
            V=jnp.zeros_like(X),
            F=jnp.zeros_like(X),
            ds=ds,
        )

    @staticmethod
    def make_beam(
        x0: tuple,
        x1: tuple,
        n_markers: int,
        dx: float,
    ) -> "LagrangianBody":
        """
        Uniformly spaced markers along a straight beam from x0 to x1.

        Parameters
        ----------
        x0, x1    : end-point coordinates  (D-tuples)
        n_markers : number of markers
        dx        : Eulerian grid spacing
        """
        x0 = jnp.array(x0, dtype=float)
        x1 = jnp.array(x1, dtype=float)
        t  = jnp.linspace(0, 1, n_markers)[:, None]
        X  = x0 + t * (x1 - x0)                       # (N, D)
        length = jnp.linalg.norm(x1 - x0)
        ds = jnp.full((n_markers,), length / (n_markers - 1))
        return LagrangianBody(
            X=X, X0=X,
            V=jnp.zeros_like(X),
            F=jnp.zeros_like(X),
            ds=ds,
        )

    # ------------------------------------------------------------------
    # Convenience updates (return new NamedTuple)
    # ------------------------------------------------------------------

    def with_positions(self, X_new: jnp.ndarray) -> "LagrangianBody":
        return LagrangianBody(X=X_new, X0=self.X0,
                              V=self.V, F=self.F, ds=self.ds)

    def with_velocities(self, V_new: jnp.ndarray) -> "LagrangianBody":
        return LagrangianBody(X=self.X, X0=self.X0,
                              V=V_new, F=self.F, ds=self.ds)

    def with_forces(self, F_new: jnp.ndarray) -> "LagrangianBody":
        return LagrangianBody(X=self.X, X0=self.X0,
                              V=self.V, F=F_new, ds=self.ds)
