from typing import NamedTuple, Optional

import jax.numpy as jnp


class PointCloud2D(NamedTuple):
    """
    2D Lagrangian marker point cloud.
    """

    X:    jnp.ndarray # (N, D=2) current position
    V:    jnp.ndarray # (N, D=2) current velocity
    A:    jnp.ndarray # (N, D=2) current acceleration
    F:    jnp.ndarray # (N, D=2) Force density from fluid
    ds:   jnp.ndarray # (N,)     arc-length / area element

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def make_circle(
        center: tuple,
        radius: float,
        n_markers: int
    ) -> "PointCloud2D":
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

        return PointCloud2D(
            X=X,
            V=jnp.zeros_like(X),
            A=jnp.zeros_like(X),
            F=jnp.zeros_like(X),
            ds=ds,
        )

    def update_X(self, new_X: jnp.ndarray) -> "PointCloud2D":
        assert new_X.shape == self.X.shape, "Array shape mismatch during updating PointCloud2D.X."
        return PointCloud2D(new_X, self.V, self.A, self.F, self.ds)
    
    def update_V(self, new_V: jnp.ndarray) -> "PointCloud2D":
        assert new_V.shape == self.V.shape, "Array shape mismatch during updating PointCloud2D.V."
        return PointCloud2D(self.X, new_V, self.A, self.F, self.ds)
    
    def update_A(self, new_A: jnp.ndarray) -> "PointCloud2D":
        assert new_A.shape == self.A.shape, "Array shape mismatch during updating PointCloud2D.A."
        return PointCloud2D(self.X, self.V, new_A, self.F, self.ds)
    
    def update_F(self, new_F: jnp.ndarray) -> "PointCloud2D":
        assert new_F.shape == self.F.shape, "Array shape mismatch during updating PointCloud2D.F."
        return PointCloud2D(self.X, self.V, self.A, new_F, self.ds)