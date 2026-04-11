"""
Lattice descriptors: D2Q9, D3Q19, D3Q27.

All arrays follow the convention:
  c   : (Q, D)  integer velocity vectors
  w   : (Q,)    quadrature weights
  opp : (Q,)    index of the opposite direction  (c[opp[q]] == -c[q])

The same equilibrium / streaming / collision code works for every lattice
because all operations are written in terms of these arrays.
"""

from typing import NamedTuple

import jax.numpy as jnp


class Lattice(NamedTuple):
    """Immutable, JAX-traceable lattice definition."""

    D: int            # spatial dimension
    Q: int            # number of discrete velocities
    c: jnp.ndarray    # (Q, D) lattice velocity vectors (integer values)
    w: jnp.ndarray    # (Q,)   weights
    opp: jnp.ndarray  # (Q,)   opposite-direction index


# ---------------------------------------------------------------------------
# D2Q9
# ---------------------------------------------------------------------------
#  Index layout (standard):
#   6  2  5
#   3  0  1
#   7  4  8
# ---------------------------------------------------------------------------
_c_d2q9 = jnp.array([
    [ 0,  0],   # 0  rest
    [ 1,  0],   # 1  E
    [ 0,  1],   # 2  N
    [-1,  0],   # 3  W
    [ 0, -1],   # 4  S
    [ 1,  1],   # 5  NE
    [-1,  1],   # 6  NW
    [-1, -1],   # 7  SW
    [ 1, -1],   # 8  SE
], dtype=jnp.int32)

_w_d2q9 = jnp.array([
    4/9,
    1/9,  1/9,  1/9,  1/9,
    1/36, 1/36, 1/36, 1/36,
])

_opp_d2q9 = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=jnp.int32)

D2Q9 = Lattice(D=2, Q=9, c=_c_d2q9, w=_w_d2q9, opp=_opp_d2q9)


# ---------------------------------------------------------------------------
# D3Q19
# ---------------------------------------------------------------------------
# Velocities: rest (1) + face-connected (6) + edge-connected (12)
# ---------------------------------------------------------------------------
_c_d3q19 = jnp.array([
    [ 0,  0,  0],   # 0  rest
    [ 1,  0,  0],   # 1
    [-1,  0,  0],   # 2
    [ 0,  1,  0],   # 3
    [ 0, -1,  0],   # 4
    [ 0,  0,  1],   # 5
    [ 0,  0, -1],   # 6
    [ 1,  1,  0],   # 7
    [-1, -1,  0],   # 8
    [ 1, -1,  0],   # 9
    [-1,  1,  0],   # 10
    [ 1,  0,  1],   # 11
    [-1,  0, -1],   # 12
    [ 1,  0, -1],   # 13
    [-1,  0,  1],   # 14
    [ 0,  1,  1],   # 15
    [ 0, -1, -1],   # 16
    [ 0,  1, -1],   # 17
    [ 0, -1,  1],   # 18
], dtype=jnp.int32)

_w_d3q19 = jnp.array([
    1/3,
    1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
    1/36, 1/36, 1/36, 1/36,
    1/36, 1/36, 1/36, 1/36,
    1/36, 1/36, 1/36, 1/36,
])

_opp_d3q19 = jnp.array(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17],
    dtype=jnp.int32,
)

D3Q19 = Lattice(D=3, Q=19, c=_c_d3q19, w=_w_d3q19, opp=_opp_d3q19)


# ---------------------------------------------------------------------------
# D3Q27
# ---------------------------------------------------------------------------
def _build_d3q27():
    import itertools
    import numpy as np

    vels = list(itertools.product([-1, 0, 1], repeat=3))
    c = np.array(vels, dtype=np.int32)  # (27, 3)

    # weights by speed^2: 0->8/27, 1->2/27, 2->1/54, 3->1/216
    speed2 = np.sum(c ** 2, axis=1)
    w_map = {0: 8/27, 1: 2/27, 2: 1/54, 3: 1/216}
    w = np.array([w_map[int(s)] for s in speed2])

    opp = np.array(
        [np.where(np.all(c == -c[q], axis=1))[0][0] for q in range(27)],
        dtype=np.int32,
    )
    return (
        jnp.array(c),
        jnp.array(w),
        jnp.array(opp),
    )


_c_d3q27, _w_d3q27, _opp_d3q27 = _build_d3q27()
D3Q27 = Lattice(D=3, Q=27, c=_c_d3q27, w=_w_d3q27, opp=_opp_d3q27)
