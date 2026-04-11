"""
Elastic constitutive models for Lagrangian structures.

Each model is a callable  (body: LagrangianBody) -> LagrangianBody
that returns a new body with updated F (Lagrangian force density).

Models
------
linear_spring_tether  — each marker tethered to its reference position
linear_beam           — Euler-Bernoulli bending stiffness along a 1-D curve
neo_hookean_membrane  — in-plane stretching for 2-D membranes (placeholder)
"""

from typing import Callable

import jax.numpy as jnp

from src.immersed_boundary.markers import LagrangianBody


# ---------------------------------------------------------------------------
# Factory functions — return a model callable
# ---------------------------------------------------------------------------

def linear_spring_tether(stiffness: float) -> Callable:
    """
    Tether force: each marker is connected to its reference position X0
    by a linear spring.

        F_k = -stiffness * (X_k - X0_k)

    Good for: rigid body approximation, pinned structures.
    """
    def model(body: LagrangianBody) -> LagrangianBody:
        F = -stiffness * (body.X - body.X0)   # (N, D)
        return body.with_forces(F)
    return model


def linear_beam(
    stiffness: float,
    bending_stiffness: float = 0.0,
) -> Callable:
    """
    Linear elastic beam along a 1-D curve of markers.

    Stretching term:
        F_k += stiffness * (|X_{k+1} - X_k| - ds_k) * t_k
        where t_k is the unit tangent vector.

    Bending term (Euler-Bernoulli, finite-difference curvature):
        F_k += -bending_stiffness * d^2X/ds^2|_k   (central difference)
    """
    def model(body: LagrangianBody) -> LagrangianBody:
        X   = body.X    # (N, D)
        ds  = body.ds   # (N,)
        N   = X.shape[0]

        # -- stretching --
        dX     = jnp.roll(X, -1, axis=0) - X          # (N, D)
        length = jnp.linalg.norm(dX, axis=-1, keepdims=True)  # (N,1)
        t      = dX / (length + 1e-14)                 # unit tangent
        stretch = stiffness * (length - ds[:, None]) * t  # (N, D)

        F_stretch = stretch - jnp.roll(stretch, 1, axis=0)

        # -- bending --
        F_bend = jnp.zeros_like(X)
        if bending_stiffness > 0.0:
            Xp1  = jnp.roll(X, -1, axis=0)
            Xm1  = jnp.roll(X,  1, axis=0)
            curv = (Xp1 - 2.0 * X + Xm1) / (ds[:, None] ** 2)
            F_bend = bending_stiffness * curv

        return body.with_forces(F_stretch + F_bend)
    return model


def neo_hookean_membrane(
    shear_modulus: float,
    bulk_modulus:  float,
) -> Callable:
    """
    Placeholder for a Neo-Hookean 2-D membrane model.
    Full implementation requires computing the deformation gradient F = dX/dX0.
    """
    def model(body: LagrangianBody) -> LagrangianBody:
        raise NotImplementedError(
            "neo_hookean_membrane is not yet implemented. "
            "Use linear_spring_tether or linear_beam for now."
        )
    return model
