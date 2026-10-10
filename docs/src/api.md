# API Reference

This page provides the complete, categorized reference of types, operators, reconstruction algorithms, and utilities in `AstraToolbox.jl`.

Each item is documented below with its signatures, parameters, mathematical context, and usage examples.

---

## Backend & Execution Environment

APIs for selecting computational backends (CPU vs. CUDA), querying and configuring GPU devices, and managing native shared libraries.

### Backend Selection
```@docs
AbstractBackend
CPUBackend
CUDABackend
cpu
cuda
```

### GPU Management & System Diagnostics
```@docs
has_cuda
has_feature
get_gpu_count
get_gpu_info
get_gpu_index
set_gpu_index
```

### Native Library Management
```@docs
set_library!
library_path
last_error
```

---

## Geometries & Coordinate Systems

Types and utilities for specifying 2D and 3D tomographic scanning trajectories, projection geometries, and reconstruction volume grids.

### 2D Projection Geometries
```@docs
AbstractGeometry
AbstractGeometry2D
Parallel2DGeometry
ParallelVec2DGeometry
FanFlat2DGeometry
FanFlatVec2DGeometry
SparseMatrix2DGeometry
Sparse2DGeometry
```

### 3D Projection Geometries
```@docs
AbstractGeometry3D
Parallel3DGeometry
ParallelVec3DGeometry
Cone3DGeometry
ConeVec3DGeometry
CylConeVec3DGeometry
```

### Volume Geometries & Spatial Queries
```@docs
AbstractVolumeGeometry
VolumeGeometry2D
VolumeGeometry3D
create_vol_geom
pixel_size
voxel_size
center
move_vol_geom
sinogram_size
geom_size
```

---

## Projection & Backprojection Operators

Stateless forward and backward projection functions, configuration options, and linear operator wrappers for integration with iterative solvers.

### Projection & Backprojection API
```@docs
project
project!
backproject
backproject!
```

### Options Specification
```@docs
AbstractProjectionOptions
ProjectionOptions2D
ProjectionOptions3D
AbstractBackprojectionOptions
BackprojectionOptions2D
BackprojectionOptions3D
```

### Linear Operator Wrapper
```@docs
AstraProjectorMap
```

---

## Reconstruction Algorithms

High-level reconstruction interface and parameter structures for analytical and iterative solvers.

### Generic Reconstruction API & Options
```@docs
reconstruct
reconstruct!
AbstractMethod
AbstractReconstructionOptions
ReconstructionOptions2D
ReconstructionOptions3D
```

### 2D Reconstruction Methods
```@docs
FBPMethod
SIRTMethod
SARTMethod
CGLSMethod
ARTMethod
EMMethod
```

### 3D Reconstruction Methods
```@docs
FDK3DMethod
SIRT3DMethod
CGLS3DMethod
```

---

## Reconstruction Filters

Frequency-domain windowing filter constants used in filtered backprojection (`FBPMethod` and `FDK3DMethod`).

```@docs
FILTER_NONE
FILTER_RAMLAK
FILTER_SHEPPLOGAN
FILTER_COSINE
FILTER_HAMMING
FILTER_HANN
FILTER_TUKEY
FILTER_LANCZOS
FILTER_TRIANGULAR
FILTER_GAUSSIAN
FILTER_BARTLETTHANN
FILTER_BLACKMAN
FILTER_NUTTALL
FILTER_BLACKMANHARRIS
FILTER_BLACKMANNUTTALL
FILTER_FLATTOP
FILTER_KAISER
FILTER_PARZEN
FILTER_PROJECTION
FILTER_SINOGRAM
FILTER_RPROJECTION
FILTER_RSINOGRAM
FILTER_ERROR
```

---

## Discrete Algebraic Reconstruction Technique (DART)

Workflows and building blocks for DART reconstruction incorporating prior knowledge of discrete gray levels.

```@docs
reconstruct_dart
dart_segment
dart_segment!
dart_mask
dart_mask!
dart_smoothing
dart_smoothing!
```

---

## Composite Geometries

Split-volume forward projection, backprojection, and accumulation routines for processing memory-constrained large-scale 3D datasets.

```@docs
CompositeVolumePart
CompositeProjectionPart
composite_project
composite_project!
composite_backproject
composite_backproject!
do_composite_FP
do_composite_FP!
do_composite_BP
do_composite_BP!
accumulate_FP!
accumulate_BP!
```

---

## Geometry Utilities & Simulation Tools

Geometric transformations, center-of-rotation alignment, ray tracing, circular clipping, standard phantom generation, and Poisson/Gaussian noise simulation.

### Vector Conversion & Ray Tracing
```@docs
geom_2vec
geom_postalignment
project_point
projected_bbox
clip_circle
clip_circle!
```

### Phantoms & Noise Simulation
```@docs
shepp_logan
shepp_logan!
shepp_logan_3d
add_noise_to_sino
add_noise_to_sino!
```
