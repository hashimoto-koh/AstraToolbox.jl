# Interactive Demonstrations (`demos/`)

`AstraToolbox.jl` includes a comprehensive suite of 9 interactive Jupyter demonstration notebooks in the [`demos/`](https://github.com/hashimoto-koh/AstraToolbox.jl/tree/master/demos) directory.

Designed specifically with beginner and intermediate Julia users in mind, every notebook features:
- **Sectional structure**: Clear progression from mathematical problem formulation to implementation and evaluation.
- **Rich inline commentary**: Thoroughly documented code cells explaining every calculation step.
- **Preceding markdown cells**: Explaining the function purpose, expected array shapes, parameter units, and both explicit and implicit default values.
- **Makie visualization**: Beautiful, publication-quality inline plots and heatmaps using `CairoMakie`.

---

## Running the Demonstrations

The demo notebooks are located in the `demos/` directory at the root of the repository.

### Prerequisites

To execute the notebooks and render the visualizations, install `IJulia` and `CairoMakie` in your Julia environment:

```julia
using Pkg
Pkg.add(["IJulia", "CairoMakie"])
```

### Launching Jupyter

From the project root:

```bash
cd demos/
jupyter lab
```

Or from the Julia REPL:

```julia
using IJulia
notebook(dir = normpath(joinpath(pathof(AstraToolbox), "..", "..", "demos")))
```

---

## Notebook Catalog

### [01. 2D Parallel-Beam CT Basics](https://github.com/hashimoto-koh/AstraToolbox.jl/blob/master/demos/01_2D_parallel_basic.ipynb)
- **File**: `demos/01_2D_parallel_basic.ipynb`
- **Core Topics**:
  - Setting up 2D parallel-beam geometry (`Parallel2DGeometry`) and volume geometry (`VolumeGeometry2D`).
  - Generating analytical Shepp-Logan phantoms with `shepp_logan`.
  - Simulating sinograms with forward projection (`project`).
  - Backprojection operator properties (`backproject`).
  - Analytic reconstruction via Filtered Backprojection (`FBPMethod`).
  - Iterative algebraic reconstruction via SIRT (`SIRTMethod`) and CGLS (`CGLSMethod`).
- **Visualizations**: Phantom, sinogram, and a side-by-side reconstruction comparison grid in Makie.

---

### [02. Projector Operator Properties & Adjointness](https://github.com/hashimoto-koh/AstraToolbox.jl/blob/master/demos/02_projector_and_adjoint.ipynb)
- **File**: `demos/02_projector_and_adjoint.ipynb`
- **Core Topics**:
  - Memory allocation analysis and in-place forward projection (`project!`) and backprojection (`backproject!`).
  - The mathematical adjoint test verifying $\langle A x, y \rangle \approx \langle x, A^T y \rangle$ to double-precision tolerance.
  - Using `AstraProjectorMap` as an `AbstractMatrix{Float32}` operator.
  - Computing the operator norm $\|A\|_2$ and maximum singular value via the Power Iteration method.
- **Visualizations**: Singular value / spectral radius convergence curve over power iteration steps.

---

### [03. Custom Iterative Reconstruction & Regularization](https://github.com/hashimoto-koh/AstraToolbox.jl/blob/master/demos/03_custom_iterative_reconstruction.ipynb)
- **File**: `demos/03_custom_iterative_reconstruction.ipynb`
- **Core Topics**:
  - Beer-Lambert law and Poisson photon counting noise simulation with `add_noise_to_sino`.
  - Implementing custom gradient descent (Landweber iteration) using Julia's native broadcast loops.
  - Implementing accelerated gradient descent (FISTA) with non-negativity projection $\Pi_{\ge 0}(x) = \max(x, 0)$.
  - Tracking objective function $\|A x - y\|_2^2$ reduction.
- **Visualizations**: Noisy sinograms, residual convergence history curves, and comparative reconstructions.

---

### [04. 3D Cone-Beam CT & FDK Reconstruction](https://github.com/hashimoto-koh/AstraToolbox.jl/blob/master/demos/04_3D_conebeam_FDK.ipynb)
- **File**: `demos/04_3D_conebeam_FDK.ipynb`
- **Core Topics**:
  - 3D circular cone-beam geometry specification with `Cone3DGeometry`.
  - Analytical 3D head phantom generation with `shepp_logan_3d`.
  - High-speed GPU forward projection on CUDA devices.
  - 3D Feldkamp-Davis-Kress (`FDK3DMethod`) volume reconstruction.
- **Visualizations**: Interactive multi-planar reformatting (MPR) showing orthogonal Axial, Coronal, and Sagittal cross-sections.

---

### [05. 3D Vector Geometry & Calibration](https://github.com/hashimoto-koh/AstraToolbox.jl/blob/master/demos/05_3D_geometry_and_cone_vec.ipynb)
- **File**: `demos/05_3D_geometry_and_cone_vec.ipynb`
- **Core Topics**:
  - Converting circular geometries to 12-parameter vector geometries (`geom_2vec`, `ConeVec3DGeometry`).
  - Visualizing 3D ray trajectories and source-detector orbits in 3D coordinate space.
  - Simulating and correcting center-of-rotation (COR) misalignment and detector shifts with `geom_postalignment`.
  - Comparison of uncorrected double-edge blurring vs. calibrated sharp FDK reconstruction.
- **Visualizations**: 3D geometric orbit plots in Makie and reconstructed slice comparison showing artifact suppression.

---

### [06. Computed Laminography & Layer Separation](https://github.com/hashimoto-koh/AstraToolbox.jl/blob/master/demos/06_conebeam_laminography.ipynb)
- **File**: `demos/06_conebeam_laminography.ipynb`
- **Core Topics**:
  - Principles of Computed Laminography (CL) with an inclined rotation axis for flat specimens (e.g. printed circuit boards).
  - Simulating multi-layer phantom structures with micro-traces and solder pads.
  - Understanding the "missing cone" Fourier sampling deficiency.
  - Iterative 3D reconstruction (`SIRT3DMethod`) to eliminate cross-talk between adjacent layers.
- **Visualizations**: Multi-layer ground truth structure and depth-resolved layer separation slices.

---

### [07. Sampling Theory & Projector Kernels](https://github.com/hashimoto-koh/AstraToolbox.jl/blob/master/demos/07_sampling_and_accuracy.ipynb)
- **File**: `demos/07_sampling_and_accuracy.ipynb`
- **Core Topics**:
  - Nyquist-Shannon sampling limits in computed tomography (angular view count vs. detector pitch).
  - Comparing 4 distinct 2D CPU projection kernels:
    - `:line` (Siddon ray tracer)
    - `:linear` (Joseph interpolation)
    - `:strip` (finite beam width)
    - `:distance_driven` (overlapping footprint)
  - Region-of-Interest (ROI) zooming using `VolumeGeometry2D` with shifted window boundaries.
- **Visualizations**: View-sampling streak artifacts, kernel difference maps, and full-field vs. ROI zoomed reconstructions.

---

### [08. Limited-Angle, Sparse-View CT & DART](https://github.com/hashimoto-koh/AstraToolbox.jl/blob/master/demos/08_limited_angle_and_sparse_view.ipynb)
- **File**: `demos/08_limited_angle_and_sparse_view.ipynb`
- **Core Topics**:
  - Ill-posed inverse problems: limited angular aperture ($\pm 60^\circ$) and sparse view (15 angles).
  - Failure modes of standard FBP under incomplete data.
  - Discrete Algebraic Reconstruction Technique (DART) via `reconstruct_dart`.
  - Prior knowledge utilization: segmenting discrete gray-level attenuation values with automatic boundary voxel updates.
- **Visualizations**: Sinograms with missing wedges, reconstruction comparisons (FBP vs. SIRT vs. DART), and error heatmaps.

---

### [09. Large-Scale Tomography & GPU Memory Management](https://github.com/hashimoto-koh/AstraToolbox.jl/blob/master/demos/09_large_scale_and_gpu.ipynb)
- **File**: `demos/09_large_scale_and_gpu.ipynb`
- **Core Topics**:
  - Best practices for zero-allocation iterative loops in Julia.
  - Measuring raw GPU voxel throughput (Mega-Voxels / second) across varying 3D volume resolutions ($64^3$ to $256^3$).
  - Overcoming GPU VRAM limitations: multi-part tiling and projection synthesis with `do_composite_FP` and `do_composite_BP`.
- **Visualizations**: GPU forward and backprojection throughput curves, and composite projection verification of split volume slices.
