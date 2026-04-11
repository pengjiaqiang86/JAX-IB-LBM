"""
Matplotlib plotting helpers.
"""

from typing import Optional

import numpy as np


def plot_field_2d(
    field:      "np.ndarray",       # (NY, NX)
    solid_mask: Optional["np.ndarray"] = None,
    title:      str   = "",
    label:      str   = "",
    save_path:  Optional[str] = None,
    cmap:       str   = "RdBu_r",
):
    """
    Imshow a 2D scalar field with optional solid contour overlay.

    Parameters
    ----------
    field      : (NY, NX) scalar array
    solid_mask : (NY, NX) bool array or None
    title      : plot title
    label      : colorbar label
    save_path  : if given, save figure to this path
    cmap       : matplotlib colormap name
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    im = ax.imshow(np.asarray(field), origin="lower", cmap=cmap)
    fig.colorbar(im, ax=ax, label=label)
    if solid_mask is not None:
        ax.contour(np.asarray(solid_mask), levels=[0.5], colors="k", linewidths=1)
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_velocity_2d(
    u:          "np.ndarray",        # (NY, NX, 2)
    solid_mask: Optional["np.ndarray"] = None,
    title:      str   = "Velocity magnitude",
    save_path:  Optional[str] = None,
    stride:     int   = 4,
):
    """
    Plot velocity magnitude as imshow + quiver arrows.

    Parameters
    ----------
    u          : (NY, NX, 2)
    solid_mask : (NY, NX) bool or None
    stride     : subsample stride for quiver arrows
    """
    import matplotlib.pyplot as plt

    u = np.asarray(u)
    speed = np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2)

    fig, ax = plt.subplots()
    im = ax.imshow(speed, origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, label="|u|")

    NY, NX = speed.shape
    Y, X = np.mgrid[0:NY:stride, 0:NX:stride]
    ax.quiver(X, Y, u[::stride, ::stride, 0], u[::stride, ::stride, 1],
              scale=5, color="white", alpha=0.6)

    if solid_mask is not None:
        ax.contour(np.asarray(solid_mask), levels=[0.5], colors="k")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
