"""
JAX Immersed Boundary Lattice-Boltzmann Method
===============================================
A dimension-agnostic (2D/3D) IB-LBM framework built on JAX.

Quick start
-----------
>>> from src.core import D2Q9, EulerianGrid, SimulationParams
>>> from src.boundary import DirichletVelocityBC, NeumannBC, BounceBackBC
>>> from src.solvers import make_lbm_step
"""

from src.core.lattice import D2Q9, D3Q19, D3Q27, Lattice
from src.core.grid import EulerianGrid
from src.core.state import FluidState
from src.core.params import SimulationParams

__all__ = [
    "D2Q9", "D3Q19", "D3Q27", "Lattice",
    "EulerianGrid",
    "FluidState",
    "SimulationParams",
]
