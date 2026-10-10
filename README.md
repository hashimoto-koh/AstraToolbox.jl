# AstraToolbox.jl

[![CI](https://github.com/hashimoto-koh/AstraToolbox.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/hashimoto-koh/AstraToolbox.jl/actions/workflows/ci.yml)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://hashimoto-koh.github.io/AstraToolbox.jl/dev/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Julia](https://img.shields.io/badge/julia-v1.10+-purple.svg)](https://julialang.org)

AstraToolbox.jl is an unofficial Julia port of
[ASTRA Toolbox](https://github.com/astra-toolbox/astra-toolbox).

This project is **not** affiliated with, endorsed by, or maintained by imec-Vision Lab,
the University of Antwerp, CWI (Centrum Wiskunde & Informatica), or the upstream ASTRA Toolbox contributors.

The ASTRA Toolbox is originally developed by imec-Vision Lab (University of Antwerp) and CWI
and is distributed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).

This repository exists because the original `ASTRA Toolbox` project is an outstanding, highly performant, and widely used platform in tomographic reconstruction research. `AstraToolbox.jl` is developed with explicit respect for that foundational work, and with sincere gratitude to the original authors and maintainers for making the ASTRA Toolbox open source and advancing accessible tomographic reconstruction.

`AstraToolbox.jl` provides a high-performance, Julia-native binding for the ASTRA Toolbox. Unlike wrapper packages that call the Python binding via `PyCall` or `PythonCall`, `AstraToolbox.jl` communicates directly with the compiled C++/CUDA core library through a zero-overhead C ABI bridge. It exposes a stateless, idiomatic Julia interface operating seamlessly on standard Julia `Array{Float32}` buffers with zero unnecessary memory copies.

---

## Key Highlights

- **Stateless & Julia-Idiomatic**: No internal ID dictionaries, object handles, or explicit cleanup calls. Pass standard Julia arrays directly to `project` and `reconstruct`.
- **Zero-Copy Memory Borrowing**: Native C++ kernels read directly from and write directly into Julia-allocated memory buffers (`Float32`).
- **Comprehensive 2D & 3D Geometries**:
  - 2D: Parallel, Fan-flat, Arbitrary Ray/Detector Vector (`ParallelVec2D`, `FanFlatVec2D`), and Sparse Matrix.
  - 3D: Parallel, Cone-beam, 12-parameter Vector (`ConeVec3D`, `ParallelVec3D`), and Helical/Cylindrical (`CylConeVec3D`).
  - Flexible Volume Geometries with arbitrary pixel/voxel size, window offsets, and ROI zooming (`VolumeGeometry2D`, `VolumeGeometry3D`, `create_vol_geom`).
- **Full CPU & GPU (CUDA) Support**:
  - Analytic algorithms: Filtered Backprojection (FBP, 2D) and Feldkamp-Davis-Kress (FDK, 3D GPU).
  - Iterative algorithms: SIRT, SART, CGLS, ART, and EM (2D CPU/GPU and 3D GPU).
  - Advanced discrete tomography: **DART** (Discrete Algebraic Reconstruction Technique) for 2D and 3D with automatic segmentation and boundary refinement.
  - Multi-part tiling: **Composite FP/BP** (`do_composite_FP`, `do_composite_BP`) for reconstructing volumes that exceed GPU VRAM.
- **LinearAlgebra Integration (`AstraProjectorMap`)**:
  - First-class `AbstractMatrix{Float32}` operator representing the forward projector $A$ and adjoint backprojector $A^T$.
  - Matrix-free multiplication (`mul!`), 5-argument `mul!`, and direct compatibility with Julia iterative solvers (`Krylov.jl`, `IterativeSolvers.jl`).
- **Interactive Demonstrations**: 9 ready-to-run Jupyter notebooks in [`demos/`](./demos/) featuring rich visualizations powered by **Makie**.

---

## Upstream Policy

This repository does **not** vendor the upstream `ASTRA Toolbox` source tree in Git.

Instead, `deps/build.jl` fetches the upstream C++/CUDA sources at build time from a pinned upstream Git commit (`bd0323e9b4b243ee90bca41341927b05e6e816c8`), applies minimal integration shims and patches from `deps/wrapper/`, and compiles the shared library `libastra.so` locally on your system.

This policy maintains a transparent, reproducible build story and a clear boundary between upstream ASTRA Toolbox and the Julia package.

---

## Installation & Build

### Prerequisites

Building `AstraToolbox.jl` requires native C++ compilation tools and autotools:

- **C++ Host Compiler**: A C++11 compliant compiler (`g++` or `clang++`).
- **Build Utilities**: `make`, `autoconf`, `automake`, and `libtool`.
- **Git**: Required by `deps/build.jl` to clone the pinned upstream ASTRA Toolbox repository sources.
- **NVIDIA GPU & CUDA Toolkit (Optional, for GPU acceleration)**:
  - CUDA-capable GPU (Compute Capability 3.5 or higher recommended).
  - NVIDIA CUDA Toolkit (`nvcc`) version 10.0 or higher.
  - `nvcc` should be accessible in your `PATH` (or specified via `ASTRATOOLBOXJL_CUDA`, `CUDA_ROOT`, or `CUDA_HOME`).

> [!IMPORTANT]
> Because `AstraToolbox.jl` builds the native ASTRA C++/CUDA library (`libastra.so`) directly on your machine via `deps/build.jl`, the build tools and compiler must be installed before running `Pkg.build("AstraToolbox")`.

On Ubuntu / Debian:
```bash
sudo apt-get update
sudo apt-get install -y build-essential autoconf automake libtool git

# Optional: for CUDA GPU acceleration
sudo apt-get install -y nvidia-cuda-toolkit
```

### Standard Installation & Build

In Julia's package manager (`Pkg` mode, press `]` in the REPL):
```julia
pkg> add https://github.com/hashimoto-koh/AstraToolbox.jl.git
pkg> build AstraToolbox
```

Or programmatically in standard Julia code:
```julia
using Pkg
Pkg.add(url="https://github.com/hashimoto-koh/AstraToolbox.jl.git")
Pkg.build("AstraToolbox")
```

For development (local checkout):
```bash
julia --startup-file=no --project=. -e 'import Pkg; Pkg.develop(path=pwd()); Pkg.build("AstraToolbox")'
```

### Build Mechanics

Running `Pkg.build("AstraToolbox")` automatically performs the following steps:
1. Clones upstream `astra-toolbox/astra-toolbox` into `deps/work/astra-toolbox` at pinned commit `bd0323e` (if not already cached).
2. Applies necessary upstream compatibility patches (e.g. 2D DART boundary helper fix).
3. Installs Julia C ABI wrapper files (`julia_c_api.h` and `julia_c_api.cpp`) into upstream source tree from `deps/wrapper/`.
4. Patches upstream Autotools `Makefile.in` to include the wrapper object `src/julia_c_api.lo`.
5. Detects the CUDA toolkit root (`ASTRATOOLBOXJL_CUDA`, `CUDA_ROOT`, `CUDA_HOME`, `/usr/local/cuda`, or `which nvcc`).
6. Configures upstream via `./autogen.sh` and `./configure --with-cuda=$cuda_root --with-python=no`.
7. Compiles `libastra.la` using parallel make jobs (`min(Sys.CPU_THREADS, 8)`).
8. Copies the compiled `libastra.so` into `deps/usr/lib/libastra.so` and generates `deps/deps.jl`.

### Build Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `ASTRATOOLBOXJL_CUDA` | Custom path to CUDA installation root directory | Detected via `CUDA_ROOT`, `CUDA_HOME`, `/usr/local/cuda`, or `nvcc` |
| `CUDA_ROOT` / `CUDA_HOME` | Standard CUDA directory fallback | — |
| `ASTRATOOLBOXJL_ASTRA_SOURCE` | Custom upstream repository URL or local path | `https://github.com/astra-toolbox/astra-toolbox.git` |
| `ASTRATOOLBOXJL_ASTRA_REF` | Custom Git commit, tag, or branch to check out | `bd0323e9b4b243ee90bca41341927b05e6e816c8` |
| `ASTRA_JULIA_LIB` | Runtime override path to an existing precompiled `libastra.so` | Empty (uses `deps/usr/lib/libastra.so`) |

### Library Status & GPU Inspection

Inside Julia, you can verify the status of the native library and CUDA availability:

```julia
using AstraToolbox

# Path to the loaded native shared library
println("Loaded library: ", AstraToolbox.library_path())

# Check if CUDA GPU acceleration is enabled and compiled
if has_cuda()
    println("CUDA is enabled!")
    println("Available GPU count: ", get_gpu_count())
    println("Current GPU device:  ", get_gpu_index())
    println("GPU device details:\n", get_gpu_info())
else
    println("Running in CPU-only mode.")
end

# Check support for specific ASTRA features
println("Projectors scaled as line integrals: ", has_feature("projectors_scaled_as_line_integrals"))
```

To manually override or relocate the native shared library at runtime:
```julia
using AstraToolbox
AstraToolbox.set_library!("/custom/path/to/libastra.so")
```

---

## Quick Start Examples

### 1. 2D CT Forward Projection & FBP Reconstruction

```julia
using AstraToolbox

# Create a 256x256 Shepp-Logan phantom
volume = shepp_logan(256)

# Set up a parallel-beam geometry with 180 projection angles over [0, π)
angles = Float32.(range(0, pi, length = 180))
geometry = Parallel2DGeometry(angles; detector_count = 256, detector_width = 1.0f0)

# Forward projection (sinogram simulation)
sinogram = project(volume, geometry; backend = cpu())

# Filtered Backprojection (FBP) reconstruction
rec_fbp = reconstruct(sinogram, geometry, FBPMethod(); volume_size = (256, 256), backend = cpu())
```

### 2. 2D Iterative Reconstruction with Non-negativity (SIRT)

```julia
# SIRT with 50 iterations and non-negativity constraint
method = SIRTMethod(iterations = 50, min_constraint = 0.0f0)
rec_sirt = reconstruct(sinogram, geometry, method; volume_size = (256, 256), backend = cpu())
```

### 3. 3D Cone-Beam CT & GPU-Accelerated FDK

```julia
# Generate a 3D Shepp-Logan phantom
vol_3d = shepp_logan_3d((128, 128, 128))

# Circular cone-beam geometry (360 degrees, source-to-origin = 500 mm, detector-to-origin = 250 mm)
angles_3d = Float32.(range(0, 2pi, length = 180))
cone_geom = Cone3DGeometry(
    angles_3d, 128, 128, 500.0f0, 250.0f0;
    detector_spacing_u = 1.0f0, detector_spacing_v = 1.0f0
)

# High-speed GPU forward projection
sinogram_3d = project(vol_3d, cone_geom; backend = cuda())

# 3D FDK reconstruction on GPU
rec_fdk = reconstruct(sinogram_3d, cone_geom, FDK3DMethod(); volume_size = (128, 128, 128), backend = cuda())
```

### 4. Matrix-Free Linear Operator (`AstraProjectorMap`)

`AstraProjectorMap` wraps the forward projection and backprojection into an `AbstractMatrix{Float32}` operator $A$:

```julia
using AstraToolbox
using LinearAlgebra

A = AstraProjectorMap(geometry, (256, 256))

# A * x computes forward projection (vec(volume) -> vec(sinogram))
y = A * vec(volume)

# A' * y computes backprojection (vec(sinogram) -> vec(volume))
x_bp = A' * y

# In-place operations (zero-allocation)
y_buf = zeros(Float32, size(A, 1))
mul!(y_buf, A, vec(volume))
```

---

## Interactive Demonstrations (`demos/`)

A curated set of 9 beginner-friendly Jupyter notebooks is provided in the [`demos/`](./demos/) directory. All notebooks feature detailed markdown explanations, in-depth parameter notes, and rich interactive visualizations built with **Makie** (`CairoMakie`).

| Notebook | Topic & Features | Visualizations |
|:---|:---|:---|
| [`01_2D_parallel_basic.ipynb`](./demos/01_2D_parallel_basic.ipynb) | 2D parallel-beam CT: FP, BP, FBP vs iterative SIRT / CGLS | Sinogram & comparative reconstruction grid |
| [`02_projector_and_adjoint.ipynb`](./demos/02_projector_and_adjoint.ipynb) | In-place zero-allocation, adjointness test $\langle Ax, y \rangle \approx \langle x, A^T y \rangle$, power iteration for $\|A\|_2$ | Singular value convergence history |
| [`03_custom_iterative_reconstruction.ipynb`](./demos/03_custom_iterative_reconstruction.ipynb) | Poisson noise simulation, custom Landweber and accelerated FISTA with non-negativity projection | Residual loss curve & artifact analysis |
| [`04_3D_conebeam_FDK.ipynb`](./demos/04_3D_conebeam_FDK.ipynb) | 3D cone-beam geometry, 3D Shepp-Logan, GPU projection, FDK reconstruction | Orthogonal 3-slice view (Axial, Coronal, Sagittal) |
| [`05_3D_geometry_and_cone_vec.ipynb`](./demos/05_3D_geometry_and_cone_vec.ipynb) | 12-parameter vector geometry (`geom_2vec`), center-of-rotation (COR) shift & detector tilt calibration | 3D source/detector trajectories & COR correction |
| [`06_conebeam_laminography.ipynb`](./demos/06_conebeam_laminography.ipynb) | Computed laminography with tilted axis, missing-cone artifacts, 3D SIRT layer separation | Multi-layer PCB depth slice separation |
| [`07_sampling_and_accuracy.ipynb`](./demos/07_sampling_and_accuracy.ipynb) | Nyquist-Shannon angular sampling, projection kernels (`line`, `linear`, `strip`, `distance_driven`), ROI zoom | Aliasing streak comparison & ROI detail zoom |
| [`08_limited_angle_and_sparse_view.ipynb`](./demos/08_limited_angle_and_sparse_view.ipynb) | Limited-angle ($\pm 60^\circ$) & sparse-view (15 angles) CT, DART discrete tomography | FBP vs SIRT vs DART error heatmaps |
| [`09_large_scale_and_gpu.ipynb`](./demos/09_large_scale_and_gpu.ipynb) | Pre-allocated buffers, GPU voxel throughput benchmark, multi-volume tiling (`do_composite_FP/BP`) | Mega-Voxels/s scaling & tiled projection |

To run the notebooks:
```bash
cd demos/
jupyter lab
```

---

## Upstream Credits & Attribution

`AstraToolbox.jl` is an independent open-source port and wrapper developed with deep gratitude to the original creators and contributors of the **ASTRA Toolbox**:
- **imec-Vision Lab**, University of Antwerp, Belgium
- **CWI**, Centrum Wiskunde & Informatica, Amsterdam, The Netherlands
- Project homepage: [https://www.astra-toolbox.com/](https://www.astra-toolbox.com/)
- GitHub repository: [https://github.com/astra-toolbox/astra-toolbox](https://github.com/astra-toolbox/astra-toolbox)

If you use `AstraToolbox.jl` in academic work, please cite the upstream ASTRA Toolbox papers as detailed in [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

---

## Related Julia Packages

The following Julia packages and organizations may be useful for users
interested in tomography, image reconstruction, phantoms, sinograms, and
computational imaging. This list is not exhaustive.

- [JuliaImageRecon](https://github.com/JuliaImageRecon) — Julia packages for image reconstruction and related tools.
- [Sinograms.jl](https://github.com/JuliaImageRecon/Sinograms.jl) — tools for sinograms, tomography, Radon transforms, and CT/CBCT geometry.
- [ImagePhantoms.jl](https://github.com/JuliaImageRecon/ImagePhantoms.jl) — software phantoms and analytical Radon/X-ray transforms.
- [MIRT.jl](https://github.com/JeffFessler/MIRT.jl) — Julia version of the Michigan Image Reconstruction Toolbox.
- [RadonKA.jl](https://github.com/roflmaostc/RadonKA.jl) — Radon transforms with multithreading and CUDA support.

Contributions to this list are welcome via issues or pull requests.

---

## License

This package is licensed under the [GNU General Public License v3.0 or later (GPL-3.0-or-later)](./LICENSE), in accordance with the upstream ASTRA Toolbox licensing terms. Third-party attribution, academic citations, and redistribution obligations are documented in [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
