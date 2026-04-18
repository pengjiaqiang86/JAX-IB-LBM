# JAX-IB-LBM

A GPU-accelerated Lattice Boltzmann Method (LBM) solver with Immersed Boundary (IB) coupling, built on [JAX](https://github.com/google/jax).

## Overview

JAX-IB-LBM implements the weakly-compressible LBM with BGK collision on D2Q9 and D3Q19 lattices, extended with a direct-forcing Immersed Boundary method for simulating flow around complex and moving geometries. The solver is fully JIT-compiled and runs unchanged on CPU, GPU, and TPU.

## Features

- **Hardware-agnostic** — single codebase runs on CPU, CUDA GPU, and TPU via JAX's XLA backend
- **JIT-compiled rollouts** — `funcutils.repeated` + `funcutils.trajectory` wrap `jax.lax.scan` for zero-Python-overhead time integration. (from [JAX-CFD](https://github.com/google/jax-cfd) under Apache License 2.0)
- **Automatic differentiation** — JAX AD passes through the entire solver; gradients with respect to initial conditions, boundary data, or geometry parameters are available out of the box
- **Immersed Boundary coupling** — Peskin regularised delta function (2-pt and 4-pt kernels) for interpolation and spreading; direct-forcing scheme enforces no-slip at Lagrangian markers
- **Boundary conditions** — Dirichlet velocity inlet, zero-gradient Neumann outlet, bounce-back (solid mask), and periodic
- **Guo body-force scheme** — consistent forcing for both IB coupling and external body forces (gravity, pressure gradient)
- **Post-processing** — vorticity, streamfunction (Poisson solver), drag/lift (momentum exchange), CFL; export to NetCDF (`.nc`) and VTK structured grid (`.vts` / `.pvd`) for ParaView

## Installation

```bash
git clone https://github.com/pengjiaqiang86/JAX-IB-LBM.git
cd JAX-IB-LBM
pip install -r requirements.txt
```

For GPU support install the CUDA-enabled jaxlib wheel before running the above:

```bash
pip install --upgrade "jax[cuda12]"
```

## Project Structure

```
src/
├── core/           lattice descriptors (D2Q9, D3Q19), grid, state, params, funcutils
├── fluid/          streaming, BGK/MRT collision, equilibrium, macroscopic quantities
├── forcing/        Guo (2002) body-force source term
├── boundary/       Dirichlet, Neumann, bounce-back, periodic BCs
├── solvers/        make_lbm_step, make_fsi_step (FSI coupling loop)
├── immersed_boundary/
│   ├── geometry.py         PointCloud2D — Lagrangian marker geometry
│   ├── delta.py            Peskin 2-pt and 4-pt regularised delta kernels
│   ├── interpolation.py    Eulerian → Lagrangian velocity interpolation
│   ├── spreading.py        Lagrangian → Eulerian force spreading
│   ├── ib_step.py          one IB-LBM coupling cycle
│   └── solid_model.py      SolidModel base class
├── postprocess/    vorticity, streamfunction, drag/lift, CFL
└── utils/          export to NetCDF and VTK (ParaView-compatible)

examples/
├── 2d_lid_driven_cavity.py
├── 2d_poiseuille_flow.py
├── 2d_taylor_green_vortex.py
├── 2d_static_circular_cylinder.py          bounce-back solid mask
├── 2d_ib_static_circular_cylinder.py       IB direct-forcing (no solid mask)
├── 2d_elastically_mounted_circular_cylinder.py
├── 2d_flow_past_tethered_circular_membrane.py
├── 2d_square_cylinder.py
└── 3d_channel_flow.py
```

## Examples

### 2D Lid-Driven Cavity

![2d-Lid-driven_cavity](https://github.com/pengjiaqiang86/JAX-IB-LBM/blob/main/imgs/2d_lid_driven_cavity.gif)

### 2D Taylor-Green Vortex

![Taylor-Green vortex — vorticity and analytical comparison](https://github.com/pengjiaqiang86/JAX-IB-LBM/blob/main/imgs/2d_taylor_green_vortex.png)

### 3D Laminar Channel Flow

![3D channel flow — velocity profile](https://github.com/pengjiaqiang86/JAX-IB-LBM/blob/main/imgs/3d_laminar_channel_flow.png)

### 2D Static Circular Cylinder — Bounce-Back

![2D-static_circular_cylinder](https://github.com/pengjiaqiang86/JAX-IB-LBM/blob/main/imgs/2d_static_circular_cylinder.gif)

### 2D Static Circular Cylinder — Immersed Boundary

![2D-static_circular_cylinder-ib](https://github.com/pengjiaqiang86/JAX-IB-LBM/blob/main/imgs/2d_static_circular_cylinder_ib.gif)

## Roadmap

... If I have free time ...

## Contributing

Contributions are welcome. Please open an issue to discuss the change before submitting a pull request.

## License

MIT
