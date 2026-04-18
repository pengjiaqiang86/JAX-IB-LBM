from pathlib import Path
from typing import Optional

import numpy as np
import jax.numpy as jnp

from src.core.state import FluidState


def save_checkpoint(
    path:  str,
    state: FluidState,
    body=None,
    **extra,
) -> None:
    """
    Save a simulation checkpoint to a .npz file.

    Parameters
    ----------
    path  : file path (e.g. "checkpoint_00500.npz")
    state : FluidState
    body  : LagrangianBody or None
    extra : additional arrays to include (e.g. rho=rho_array)
    """
    arrays = {
        "f": np.array(state.f),
        "t": np.array(state.t),
    }
    if state.g is not None:
        arrays["g"] = np.array(state.g)
    if body is not None:
        arrays["body_X"]  = np.array(body.X)
        arrays["body_X0"] = np.array(body.X0)
        arrays["body_V"]  = np.array(body.V)
        arrays["body_F"]  = np.array(body.F)
        arrays["body_ds"] = np.array(body.ds)
    arrays.update({k: np.array(v) for k, v in extra.items()})

    np.savez(path, **arrays)


def load_checkpoint(path: str):
    """
    Load a checkpoint saved by save_checkpoint.

    Returns
    -------
    state : FluidState
    body  : LagrangianBody or None (if not stored)
    extras: dict of any additional arrays
    """
    from src.immersed_boundary.markers import LagrangianBody

    data = np.load(path)
    g    = jnp.array(data["g"]) if "g" in data else None
    state = FluidState(
        f=jnp.array(data["f"]),
        g=g,
        t=int(data["t"]),
    )

    body = None
    if "body_X" in data:
        body = LagrangianBody(
            X=jnp.array(data["body_X"]),
            X0=jnp.array(data["body_X0"]),
            V=jnp.array(data["body_V"]),
            F=jnp.array(data["body_F"]),
            ds=jnp.array(data["body_ds"]),
        )

    known = {"f", "g", "t", "body_X", "body_X0", "body_V", "body_F", "body_ds"}
    extras = {k: jnp.array(data[k]) for k in data.files if k not in known}
    return state, body, extras
