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
    n_clamped: int = 0,
    clamp_stiffness: float = 0.0,
) -> Callable:
    """
    Linear elastic beam along an OPEN 1-D curve of markers.

    Stretching (tension along each segment):
        T_k = stiffness * (|X_{k+1} - X_k| - ds_k) * t_k   for k = 0..N-2
        F_k = T_k - T_{k-1}   (net nodal force, free-end BCs: T_{-1}=T_{N-1}=0)

    Bending (Euler-Bernoulli curvature, central differences):
        F_k = bending_stiffness * (X_{k+1} - 2X_k + X_{k-1}) / ds_k^2
        Free-end boundary condition: F_0 = F_{N-1} = 0

    Clamped base (optional):
        The first n_clamped markers are tethered to their reference positions
        X0 with spring stiffness clamp_stiffness.  Use this to model a
        cantilever beam (root fixed, tip free).  Choose clamp_stiffness high
        enough to resist the flow but still within the IB stability bound:
            clamp_stiffness  <~  5 * stiffness   (heuristic; check stability)

    Stability guideline (LBM units):
        max(F_k) * ds / (2*kernel_support)^D  <<  rho * c_s^2 * omega
        For the default 2D setup (support=2, dx=1, rho=1, omega≈1.5):
            stiffness  <~  1.0
            bending_stiffness  <~  1.0
    """
    def model(body: LagrangianBody) -> LagrangianBody:
        X  = body.X    # (N, D)
        ds = body.ds   # (N,)

        # ── Stretching ──────────────────────────────────────────────────
        # dX[k] = X[k+1] - X[k] via roll.  dX[N-1] wraps to X[0]-X[N-1]
        # which is a phantom "closing" segment — must be zeroed for open beam.
        dX     = jnp.roll(X, -1, axis=0) - X               # (N, D)
        length = jnp.linalg.norm(dX, axis=-1, keepdims=True)  # (N, 1)
        t      = dX / (length + 1e-14)                      # unit tangent
        T      = stiffness * (length - ds[:, None]) * t     # segment tension (N, D)

        # Zero the phantom segment [N-1 → 0] so endpoints obey free-end BC.
        T = T.at[-1].set(0.0)

        # Nodal force = divergence of tension: F_k = T_k - T_{k-1}
        # After zeroing T[-1], roll brings 0 to index 0, giving F[0] = T[0]. ✓
        F_stretch = T - jnp.roll(T, 1, axis=0)

        # ── Bending ──────────────────────────────────────────────────────
        F_bend = jnp.zeros_like(X)
        if bending_stiffness > 0.0:
            Xp1  = jnp.roll(X, -1, axis=0)
            Xm1  = jnp.roll(X,  1, axis=0)
            curv = (Xp1 - 2.0 * X + Xm1) / (ds[:, None] ** 2)
            F_bend = bending_stiffness * curv
            # Free-end BC: endpoints have no bending moment.
            # The roll-based curvature at k=0 and k=N-1 uses phantom neighbours
            # (X[N-1] and X[0] respectively) — zero them out.
            F_bend = F_bend.at[0].set(0.0)
            F_bend = F_bend.at[-1].set(0.0)

        F_total = F_stretch + F_bend

        # ── Clamped-base tether ──────────────────────────────────────────
        # Pull the first n_clamped markers back to their reference positions.
        # This models a cantilever root: the node is not free to drift.
        if n_clamped > 0 and clamp_stiffness > 0.0:
            disp = body.X[:n_clamped] - body.X0[:n_clamped]   # (n_clamped, D)
            F_clamp = -clamp_stiffness * disp
            F_total = F_total.at[:n_clamped].add(F_clamp)

        return body.with_forces(F_total)
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
