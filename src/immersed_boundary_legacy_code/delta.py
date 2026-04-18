from typing import NamedTuple, Callable

import jax.numpy as jnp


class DeltaKernel(NamedTuple):
    """
    1-D regularised delta function kernel.

    Attributes
    ----------
    support : half-width in grid cells (total stencil = 2*support cells)
    phi     : Callable[[jnp.ndarray], jnp.ndarray]
              Maps r (in grid units, i.e. r = (x - x_k) / dx) to weight.
    """
    support: int
    phi:     Callable


# ---------------------------------------------------------------------------
# 2-point (linear / hat) kernel
# ---------------------------------------------------------------------------
def _phi_2pt(r: jnp.ndarray) -> jnp.ndarray:
    """Linear hat function.  Support: r in [-1, 1]."""
    r_abs = jnp.abs(r)
    return jnp.where(r_abs <= 1.0, 1.0 - r_abs, 0.0)

PESKIN_2PT = DeltaKernel(support=1, phi=_phi_2pt)


# ---------------------------------------------------------------------------
# 4-point Peskin raised-cosine kernel (default)
# ---------------------------------------------------------------------------
def _phi_4pt(r: jnp.ndarray) -> jnp.ndarray:
    """
    Peskin 4-point kernel.  Support: r in [-2, 2].

        φ(r) = (1/8) * (3 - 2|r| + sqrt(1 + 4|r| - 4r^2))   0 <= |r| < 1
             = (1/8) * (5 - 2|r| - sqrt(-7 + 12|r| - 4r^2))  1 <= |r| < 2
             = 0                                               |r| >= 2
    """
    r_abs = jnp.abs(r)
    phi1 = (3.0 - 2.0 * r_abs + jnp.sqrt(1.0 + 4.0 * r_abs - 4.0 * r_abs ** 2)) / 8.0
    phi2 = (5.0 - 2.0 * r_abs - jnp.sqrt(-7.0 + 12.0 * r_abs - 4.0 * r_abs ** 2)) / 8.0
    return jnp.where(
        r_abs < 1.0, phi1,
        jnp.where(r_abs < 2.0, phi2, 0.0),
    )

PESKIN_4PT = DeltaKernel(support=2, phi=_phi_4pt)
