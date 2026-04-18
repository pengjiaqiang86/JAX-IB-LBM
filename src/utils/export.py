"""
Export Eulerian field snapshots and trajectories to NetCDF and VTK formats.

Both formats open directly in ParaView / VisIt and standard scientific Python
tools (xarray / netCDF4 for .nc; ParaView / pyvista for .vts).

Field conventions
-----------------
``fields`` is a plain dict ``{name: array}`` where each array is either a
JAX or NumPy array.  The module auto-detects scalars vs vectors:

  - Scalar : shape  (*spatial_shape)
  - Vector : shape  (*spatial_shape, D)   where D == grid.ndim (2 or 3)

Vectors are stored with their components named ``{name}_x``, ``{name}_y``
(and ``{name}_z`` for 3D) in NetCDF, and as a proper VTK vector tuple in
VTK so that ParaView can show arrows / glyphs.

Dependencies
------------
  xarray     — for .nc output   (``pip install xarray``)
  netCDF4    — engine for .nc   (``pip install netCDF4``)
  pyevtk     — for .vts output  (``pip install pyevtk``)

A missing dependency raises ``ImportError`` with an install hint rather than
failing silently at import time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np

from src.core.grid import EulerianGrid

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _np(arr) -> np.ndarray:
    """Accept JAX or NumPy array and return a NumPy float64 array."""
    return np.asarray(arr, dtype=np.float64)


def _is_vector(arr: np.ndarray, grid: EulerianGrid) -> bool:
    """True when arr is a vector field: shape (*spatial, D) with D == grid.ndim."""
    return arr.ndim == grid.ndim + 1 and arr.shape[-1] == grid.ndim


def _expand(fields: dict, grid: EulerianGrid) -> dict:
    """
    Return a new dict with vector fields split into scalar components.

    A vector field named ``u`` with D=2 becomes ``u_x`` and ``u_y``.
    Scalar fields are passed through unchanged.
    """
    out: dict = {}
    suffixes = ("x", "y", "z")
    for name, raw in fields.items():
        arr = _np(raw)
        if _is_vector(arr, grid):
            for d in range(grid.ndim):
                out[f"{name}_{suffixes[d]}"] = arr[..., d]
        else:
            out[name] = arr
    return out


def _vtk_order(arr: np.ndarray, grid: EulerianGrid) -> np.ndarray:
    """
    Reorder a (*spatial) array from our (Z, Y, X) storage convention to the
    VTK/pyevtk (X, Y, Z) convention as a Fortran-contiguous float64 array.

    pyevtk requires Fortran (column-major) order; using C-order causes data
    to be mis-interpreted and all frames look identical in ParaView.

    2D (NY, NX)      → (NX, NY, 1)
    3D (NZ, NY, NX)  → (NX, NY, NZ)
    """
    if grid.ndim == 2:
        return np.asfortranarray(arr.T[:, :, np.newaxis], dtype=np.float64)
    # 3D: (NZ, NY, NX) → transpose to (NX, NY, NZ)
    return np.asfortranarray(arr.transpose(2, 1, 0), dtype=np.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_netcdf(
    fields: Dict[str, np.ndarray],
    grid:   EulerianGrid,
    path:   Union[str, os.PathLike],
    *,
    t:      Optional[float] = None,
    attrs:  Optional[dict]  = None,
) -> None:
    """
    Save a single-snapshot dict of Eulerian fields to a NetCDF file.

    Parameters
    ----------
    fields : dict
        ``{name: array}`` mapping.  Scalars have shape ``(*spatial_shape)``;
        vectors ``(*spatial_shape, D)`` are split into ``{name}_x/y[/z]``.
    grid : EulerianGrid
    path : str or Path
        Output file path (should end in ``.nc``).
    t : float, optional
        Simulation time stored as a global attribute ``time``.
    attrs : dict, optional
        Additional global attributes written to the file.

    Examples
    --------
    >>> save_netcdf({"rho": rho, "u": u, "omega": omega}, grid,
    ...             "output/snapshot.nc", t=1000.0)
    """
    try:
        import xarray
    except ImportError as exc:
        raise ImportError(
            "xarray is required for NetCDF export.  "
            "Install it with:  pip install xarray netCDF4"
        ) from exc

    flat = _expand(fields, grid)

    # Build dimension names and coordinate arrays
    if grid.ndim == 2:
        dims   = ("y", "x")
        coords = {
            "y": ("y", np.array(grid.y_coords())),
            "x": ("x", np.array(grid.x_coords())),
        }
    else:
        dims   = ("z", "y", "x")
        coords = {
            "z": ("z", np.array(grid.z_coords())),
            "y": ("y", np.array(grid.y_coords())),
            "x": ("x", np.array(grid.x_coords())),
        }

    data_vars = {name: (dims, arr) for name, arr in flat.items()}

    global_attrs: dict = {}
    if t is not None:
        global_attrs["time"] = float(t)
    if attrs:
        global_attrs.update(attrs)

    ds = xarray.Dataset(data_vars, coords=coords, attrs=global_attrs)
    ds.to_netcdf(path)
    print(f"Saved {path}")


def save_trajectory_netcdf(
    fields:  Dict[str, np.ndarray],
    grid:    EulerianGrid,
    path:    Union[str, os.PathLike],
    *,
    t_vals:  Optional[Sequence[float]] = None,
    attrs:   Optional[dict]            = None,
) -> None:
    """
    Save a trajectory (time series) to a single NetCDF file.

    Parameters
    ----------
    fields : dict
        ``{name: array}`` mapping where each array has a leading time axis:
        shape ``(T, *spatial_shape)`` for scalars or
        ``(T, *spatial_shape, D)`` for vectors.
    grid : EulerianGrid
    path : str or Path
    t_vals : sequence of float, optional
        Time values for each frame.  Defaults to ``[0, 1, 2, …]``.
    attrs : dict, optional

    Examples
    --------
    >>> # u_hist has shape (OUTER_STEPS, NY, NX, 2)
    >>> save_trajectory_netcdf(
    ...     {"u": u_hist, "omega": omega_hist},
    ...     grid, "output/trajectory.nc",
    ...     t_vals=np.arange(OUTER_STEPS) * INNER_STEPS,
    ... )
    """
    try:
        import xarray
    except ImportError as exc:
        raise ImportError(
            "xarray is required for NetCDF export.  "
            "Install it with:  pip install xarray netCDF4"
        ) from exc

    # Detect T from the first field
    first = next(iter(fields.values()))
    T = _np(first).shape[0]

    if t_vals is None:
        t_vals_arr = np.arange(T, dtype=float)
    else:
        t_vals_arr = np.asarray(t_vals, dtype=float)

    # Split vector fields: (T, *spatial, D) → per-component (T, *spatial)
    flat: dict = {}
    suffixes = ("x", "y", "z")
    for name, raw in fields.items():
        arr = _np(raw)          # (T, *spatial[, D])
        spatial_arr = arr[0]    # (*spatial[, D]) — check shape of one frame
        if _is_vector(spatial_arr, grid):
            for d in range(grid.ndim):
                flat[f"{name}_{suffixes[d]}"] = arr[..., d]
        else:
            flat[name] = arr

    # Dimensions: (time, *spatial)
    if grid.ndim == 2:
        spatial_dims = ("y", "x")
        coords = {
            "time": ("time", t_vals_arr),
            "y":    ("y",    np.array(grid.y_coords())),
            "x":    ("x",    np.array(grid.x_coords())),
        }
    else:
        spatial_dims = ("z", "y", "x")
        coords = {
            "time": ("time", t_vals_arr),
            "z":    ("z",    np.array(grid.z_coords())),
            "y":    ("y",    np.array(grid.y_coords())),
            "x":    ("x",    np.array(grid.x_coords())),
        }

    dims = ("time",) + spatial_dims
    data_vars = {name: (dims, arr) for name, arr in flat.items()}

    global_attrs: dict = dict(attrs) if attrs else {}
    ds = xarray.Dataset(data_vars, coords=coords, attrs=global_attrs)

    # CF-convention attributes so ParaView's vtkNetCDFCFReader recognises the
    # "time" dimension as the time axis.  Without "units" the reader treats all
    # dimensions as spatial → "More than 3 dims without time" error for 3D grids.
    ds["time"].attrs["units"] = "steps"
    ds["time"].attrs["axis"]  = "T"

    ds.to_netcdf(path)
    print(f"Saved {path}  ({T} time steps)")


def save_vtk(
    fields:      Dict[str, np.ndarray],
    grid:        EulerianGrid,
    path:        Union[str, os.PathLike],
    *,
    z_thickness: Optional[float] = None,
) -> None:
    """
    Save a single-snapshot dict of Eulerian fields to a VTK XML structured
    grid file (``.vts``).  Readable by ParaView, VisIt, and pyvista.

    Parameters
    ----------
    fields : dict
        ``{name: array}`` mapping.  Scalars have shape ``(*spatial_shape)``;
        vectors ``(*spatial_shape, D)`` are stored as VTK vector fields
        (ParaView will offer arrow / glyph visualisation automatically).
    grid : EulerianGrid
    path : str or Path
        Output path.  The ``.vts`` extension is appended by pyevtk if absent.
    z_thickness : float, optional
        **2D grids only.** Depth of the extruded slab in the z-direction.
        Defaults to ``grid.dx`` (one cell thick — a flat slab).  Increase
        to e.g. ``10 * grid.dx`` if you want to see the slab in ParaView's
        3D view without it disappearing into a plane.

    Examples
    --------
    >>> save_vtk({"rho": rho, "u": u, "omega": omega}, grid,
    ...          "output/snapshot")                  # writes output/snapshot.vts
    >>> save_vtk(fields, grid, "output/snapshot", z_thickness=5.0)
    """
    try:
        from pyevtk.hl import gridToVTK
    except ImportError as exc:
        raise ImportError(
            "pyevtk is required for VTK export.  "
            "Install it with:  pip install pyevtk"
        ) from exc

    path = str(Path(path).with_suffix(""))   # pyevtk appends .vts itself

    # Node (corner) coordinates — pyevtk needs n+1 points for n cells
    x_nodes = np.arange(grid.NX + 1, dtype=np.float64) * grid.dx
    y_nodes = np.arange(grid.NY + 1, dtype=np.float64) * grid.dy
    if grid.ndim == 2:
        _dz = float(z_thickness) if z_thickness is not None else float(grid.dx)
        z_nodes = np.array([0.0, _dz], dtype=np.float64)
    else:
        z_nodes = np.arange(grid.NZ + 1, dtype=np.float64) * grid.dz

    cell_data: dict = {}
    for name, raw in fields.items():
        arr = _np(raw)
        if _is_vector(arr, grid):
            # Store as VTK vector: tuple of (x, y, z) component arrays
            # pyevtk expects each component in VTK (X, Y, Z) cell order
            components = [_vtk_order(arr[..., d], grid) for d in range(grid.ndim)]
            if grid.ndim == 2:
                # Pad with zero z-component for 3D VTK vectors
                components.append(np.zeros_like(components[0]))
            cell_data[name] = tuple(components)
        else:
            cell_data[name] = _vtk_order(arr, grid)

    gridToVTK(path, x_nodes, y_nodes, z_nodes, cellData=cell_data)
    print(f"Saved {path}.vts")


def save_vtk_series(
    fields_list: Sequence[Dict[str, np.ndarray]],
    grid:        EulerianGrid,
    out_dir:     Union[str, os.PathLike],
    prefix:      str = "frame",
    *,
    t_vals:      Optional[Sequence[float]] = None,
    z_thickness: Optional[float]           = None,
) -> None:
    """
    Save a sequence of snapshots as numbered VTK files plus a ``.pvd`` index.

    The ``.pvd`` file lets ParaView load the entire time series at once with
    correct time labels (File → Open → select the ``.pvd``).

    Parameters
    ----------
    fields_list : list of dicts  *or*  dict of arrays
        **List form** — one ``{name: array}`` dict per time frame, same
        structure as :func:`save_vtk`.

        **Trajectory form** — a single ``{name: array}`` dict where each
        array has a leading time axis: shape ``(T, *spatial)`` for scalars
        or ``(T, *spatial, D)`` for vectors.  T is inferred from that axis.
        This matches the output layout of ``funcutils.trajectory``.
    grid : EulerianGrid
    out_dir : str or Path
        Directory for output files (created if needed).
    prefix : str
        Filename prefix for each frame, e.g. ``"frame"`` → ``frame_0000.vts``.
    t_vals : sequence of float, optional
        Time values for the ``.pvd`` index.  Defaults to ``[0, 1, 2, …]``.
    z_thickness : float, optional
        **2D grids only.**  Forwarded to :func:`save_vtk`.  Defaults to
        ``grid.dx`` (one-cell-thick slab).

    Examples
    --------
    >>> # Trajectory form — pass the history dict directly
    >>> save_vtk_series({"u": np.array(u_hist), "omega": np.array(omega_hist)},
    ...                 grid, "output/vtk_series",
    ...                 t_vals=np.arange(OUTER_STEPS) * INNER_STEPS)
    >>>
    >>> # List form — build per-frame dicts manually
    >>> frames_dicts = [{"u": np.array(u_hist[i]), "omega": np.array(omega_hist[i])}
    ...                 for i in range(OUTER_STEPS)]
    >>> save_vtk_series(frames_dicts, grid, "output/vtk_series",
    ...                 t_vals=np.arange(OUTER_STEPS) * INNER_STEPS)
    """
    try:
        from pyevtk.hl import gridToVTK  # noqa: F401 — check import early
    except ImportError as exc:
        raise ImportError(
            "pyevtk is required for VTK export.  "
            "Install it with:  pip install pyevtk"
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accept either a list-of-dicts or a trajectory dict {name: (T, *spatial[, D])}
    if isinstance(fields_list, dict):
        first = _np(next(iter(fields_list.values())))
        T = first.shape[0]
        frames: Sequence[Dict[str, np.ndarray]] = [
            {name: _np(arr)[i] for name, arr in fields_list.items()}
            for i in range(T)
        ]
    else:
        frames = fields_list
        T = len(frames)

    if t_vals is None:
        t_vals_arr = np.arange(T, dtype=float)
    else:
        t_vals_arr = np.asarray(t_vals, dtype=float)

    print("Start exporting to VTK frames ...")

    vts_paths = []
    for i, snap in enumerate(frames):
        vts_stem = out_dir / f"{prefix}_{i:04d}"
        save_vtk(snap, grid, vts_stem, z_thickness=z_thickness)
        vts_paths.append(f"{prefix}_{i:04d}.vts")

    # Write .pvd index so ParaView can open the series as an animation
    pvd_path = out_dir / f"{prefix}.pvd"
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1">',
             '  <Collection>']
    for vts, t in zip(vts_paths, t_vals_arr):
        lines.append(f'    <DataSet timestep="{t}" group="" part="0" file="{vts}"/>')
    lines += ['  </Collection>', '</VTKFile>']
    pvd_path.write_text("\n".join(lines))

    print(f"Saved {T} VTK frames + index {pvd_path}")
