# Introduction

`AstraToolbox.jl` is an independent, unofficial Julia port and interface for the [ASTRA Toolbox](https://github.com/astra-toolbox/astra-toolbox).

!!! note "Disclaimer"
    This project is not affiliated with, endorsed by, or maintained by imec-Vision Lab, the University of Antwerp, CWI, or the upstream ASTRA Toolbox contributors. The original ASTRA Toolbox is open-source software distributed under the GNU General Public License v3.0 or later. We gratefully acknowledge the upstream authors and institutions for developing and open-sourcing the toolbox.

The primary goal of this package is to provide a seamless, high-performance interface to the ASTRA Toolbox's native C++/CUDA backend while following Julia's idioms and conventions.

Unlike the Python or MATLAB interfaces, `AstraToolbox.jl` does not use an internal ID management system for data objects. Instead, it operates directly on standard Julia `Array{Float32}` buffers via zero-copy pointer borrowing, providing a stateless, type-safe, and natural experience for Julia users.

---

## Upstream Credits & Attribution

This package is built upon the foundational work of the ASTRA Toolbox authors and contributors. We express our deep appreciation and gratitude to:
- **imec-Vision Lab**, University of Antwerp, Belgium
- **CWI**, Centrum Wiskunde & Informatica, Amsterdam, The Netherlands
- Official Website: [https://www.astra-toolbox.com/](https://www.astra-toolbox.com/)
- GitHub Repository: [https://github.com/astra-toolbox/astra-toolbox](https://github.com/astra-toolbox/astra-toolbox)

If you use `AstraToolbox.jl` in scientific publications, please cite the upstream ASTRA Toolbox papers listed in [THIRD_PARTY_NOTICES.md](https://github.com/hashimoto-koh/AstraToolbox.jl/blob/master/THIRD_PARTY_NOTICES.md).

---

## Key Features

- **Stateless High-Level API**: Direct `project`, `backproject`, and `reconstruct` functions without opaque handle dictionaries.
- **Zero-Allocation In-Place Calls**: `project!` and `backproject!` allow memory reuse inside iterative loops.
- **2D & 3D Geometries**:
  - Parallel-beam, Fan-flat, and Cone-beam models.
  - Arbitrary ray/detector vector geometries (`ParallelVec2D`, `FanFlatVec2D`, `ParallelVec3D`, `ConeVec3D`, `CylConeVec3D`).
  - Sparse matrix geometries (`SparseMatrix2DGeometry`) with `SparseMatrixCSC` interop.
  - Arbitrary pixel/voxel sizes, centers, and ROI zooming (`VolumeGeometry2D`, `VolumeGeometry3D`, `create_vol_geom`).
- **CPU & GPU (CUDA) Engines**:
  - Analytic algorithms: FBP (CPU & CUDA 2D), FDK (CUDA 3D).
  - Iterative algorithms: SIRT, SART, CGLS, ART, and EM with constraints and projection masks.
  - Advanced discrete tomography: **DART** (`reconstruct_dart`, `dart_mask`, `dart_smoothing`).
  - Memory-constrained large-scale imaging: **Composite FP/BP** (`do_composite_FP`, `do_composite_BP`) for GPU VRAM tiling.
- **LinearAlgebra Compatibility**: `AstraProjectorMap` acts as an `AbstractMatrix{Float32}` operator compatible with `LinearAlgebra` and Julia Krylov solvers (`Krylov.jl`, `IterativeSolvers.jl`).
- **Interactive Demonstrations**: 9 Jupyter notebooks in [`demos/`](demos.md) featuring beginner-friendly walkthroughs and **Makie** visualizations.

---

## Documentation Overview

- [Quick Start](quickstart.md): Minimal, reproducible 2D and 3D examples.
- [Installation](installation.md): Building from source and CUDA toolchain setup.
- [Interactive Demonstrations](demos.md): Guide to the 9 Jupyter notebooks in `demos/`.
- [2D Geometries](geometries_2d.md) & [3D Geometries](geometries_3d.md): Defining detector configurations and volume spaces.
- [Projectors](projectors.md): Forward projection, backprojection, and available interpolation kernels.
- [Algorithms](algorithms.md): Analytic and iterative reconstruction methods, options, and parameters.
- [Utilities](utilities.md): Geometric alignment, Shepp-Logan phantom generators, Poisson noise, and DART.
- [API Reference](api.md): Complete index of all exported types, functions, and docstrings.
