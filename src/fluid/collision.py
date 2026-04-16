"""
Collision operators.

BGK  — single-relaxation-time (SRT), simplest and most common.
MRT  — multiple-relaxation-time, better stability for low viscosity.

Both operators accept an optional body-force correction array `g_force`
(pre-computed by the Guo forcing scheme in src/forcing/guo.py).
When g_force is None the operators reduce to the standard BGK/MRT.
"""

from typing import Optional

import jax.numpy as jnp

from src.core.lattice import Lattice


# ---------------------------------------------------------------------------
# BGK (SRT)
# ---------------------------------------------------------------------------

def bgk_collision(
    f:        jnp.ndarray,
    feq:      jnp.ndarray,
    omega:    float,
    g_force:  Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    BGK collision with optional body-force source term.

        f_post = f - omega * (f - feq) + (1 - omega/2) * F_source

    Parameters
    ----------
    f        : (*spatial, Q)   pre-collision distributions
    feq      : (*spatial, Q)   equilibrium distributions
    omega    : float           relaxation frequency  1/tau
    g_force  : (*spatial, Q) or None
               Raw Guo forcing term F_q from guo_forcing_term().
               The (1 − ω/2) prefactor is applied here, not in guo.py.
               Pass None to skip forcing.

    Returns
    -------
    f_post : (*spatial, Q)
    """
    f_post = f - omega * (f - feq)
    if g_force is not None:
        f_post = f_post + (1.0 - 0.5 * omega) * g_force
    return f_post


# ---------------------------------------------------------------------------
# MRT
# ---------------------------------------------------------------------------

def mrt_collision(
    f:        jnp.ndarray,
    feq:      jnp.ndarray,
    M:        jnp.ndarray,
    S:        jnp.ndarray,
    g_force:  Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    MRT collision in moment space.

        f_post = f - M^{-1} S (m - m_eq) + (I - S/2) M^{-1} F_hat

    Parameters
    ----------
    f    : (*spatial, Q)
    feq  : (*spatial, Q)
    M    : (Q, Q)  transformation matrix to moment space
    S    : (Q, Q)  diagonal relaxation matrix  diag(s_0, s_1, ..., s_{Q-1})
    g_force : (*spatial, Q) or None  — same convention as bgk_collision

    Returns
    -------
    f_post : (*spatial, Q)
    """
    M_inv = jnp.linalg.inv(M)

    # project to moment space: m[..., q] = sum_q' M[q,q'] f[...,q']
    m    = jnp.einsum("qp,...p->...q", M,     f)
    meq  = jnp.einsum("qp,...p->...q", M,     feq)

    # relax
    m_post = m - jnp.einsum("qp,...p->...q", S, m - meq)

    # project back
    f_post = jnp.einsum("qp,...p->...q", M_inv, m_post)

    if g_force is not None:
        # Guo correction in moment space: (I - S/2) M^{-1} F_hat
        # Here g_force is already in f-space; convert to moment space, apply
        F_hat  = jnp.einsum("qp,...p->...q", M, g_force)
        F_hat_corr = jnp.einsum(
            "qp,...p->...q",
            jnp.eye(M.shape[0]) - 0.5 * S,
            F_hat,
        )
        correction = jnp.einsum("qp,...p->...q", M_inv, F_hat_corr)
        f_post = f_post + correction

    return f_post
