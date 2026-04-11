from src.postprocess.vorticity import compute_vorticity
from src.postprocess.streamfunction import solve_streamfunction
from src.postprocess.diagnostics import compute_drag_lift, compute_cfl

__all__ = [
    "compute_vorticity",
    "solve_streamfunction",
    "compute_drag_lift",
    "compute_cfl",
]
