# ASTRA utility scripts

This page covers additional utility functions in `AstraToolbox.jl`.

## Noise Generation

### `add_noise_to_sino`

This function adds Poisson noise to a sinogram. It's equivalent to the `add_noise_to_sino` function in the original Python interface.

```julia
sino_noise = add_noise_to_sino(sino, I0; seed=123)
```

- `sino::AbstractArray`: Input sinogram.
- `I0::Real`: Background intensity. Higher values mean lower noise levels.
- `seed::Union{Nothing, Integer}`: Optional random seed for reproducibility.
- `rng::AbstractRNG`: Optional random number generator.

There is also an in-place version: `add_noise_to_sino!(sino, I0; seed=...)`.

## Shepp-Logan Phantom Generators

`AstraToolbox.jl` provides pure Julia implementations of the 2D and 3D Shepp-Logan phantoms, matching upstream analytical ellipse/ellipsoid rasterization without external dependencies.

### 2D Phantom: `shepp_logan`, `shepp_logan!`

```julia
phantom = shepp_logan(256; modified = true)
# Or specify dimensions / geometry
phantom = shepp_logan((256, 256))
phantom = shepp_logan(geometry; volume_size = (256, 256))

# In-place version
buf = zeros(Float32, 256, 256)
shepp_logan!(buf; modified = true)
```

- `size`: Dimension `(nx, ny)` or integer `n` for `(n, n)`.
- `modified`: If `true` (default), uses the High-contrast Modified Shepp-Logan phantom. If `false`, uses the original Kak & Slaney Shepp-Logan phantom.

### 3D Phantom: `shepp_logan_3d`

```julia
phantom3d = shepp_logan_3d(128; modified = true)
# Or specify dimensions / geometry
phantom3d = shepp_logan_3d((128, 128, 128))
phantom3d = shepp_logan(geom3d; volume_size = (128, 128, 128))

# In-place version
buf3d = zeros(Float32, 128, 128, 128)
shepp_logan!(buf3d; modified = true)
```

## Geometry Conversion & Calibration

Vector geometries allow defining arbitrary source, detector, and trajectory orientations.

### `geom_2vec`

Converts a standard projection geometry into its vector-based equivalent:

```julia
vec_geom2d = geom_2vec(parallel_geom)  # Parallel2DGeometry -> ParallelVec2DGeometry
vec_fan2d  = geom_2vec(fanflat_geom)   # FanFlat2DGeometry -> FanFlatVec2DGeometry
vec_par3d  = geom_2vec(parallel3d_geom) # Parallel3DGeometry -> ParallelVec3DGeometry
vec_cone3d = geom_2vec(cone3d_geom)     # Cone3DGeometry -> ConeVec3DGeometry
```

If the input is already a vector geometry, an identical copy is returned.

### `geom_postalignment`

Applies a detector shift (measured in detector pixel units) to model rotation axis offset or detector misalignment:

```julia
# 2D geometry: shift detector center along the detector direction
aligned_geom2d = geom_postalignment(parallel_geom, 2.5)

# 3D geometry: shift detector center horizontally and vertically (factor_u, factor_v)
aligned_geom3d = geom_postalignment(cone3d_geom, (1.0, -0.5))
```

## Field-of-View (FOV) Circular Masking: `clip_circle`, `clip_circle!`

Zero out pixels (in 2D) or voxels (in 3D cylindrical FOV) that lie outside a circular boundary (equivalent to `astra.clipCircle`):

```julia
# 2D masking
img_clipped = clip_circle(img)
clip_circle!(img) # in-place

# 3D cylindrical masking (along Z-axis)
vol_clipped = clip_circle(volume)
clip_circle!(volume; radius = 100.0, center = (128.5, 128.5))
```

## Matrix-Free Linear Projector Operator: `AstraProjectorMap`

`AstraProjectorMap` provides a matrix-free linear operator (equivalent to `astra.OpTomo` in Python) that represents the forward projection matrix ``A``. Its adjoint ``A'`` represents the backprojection operator ``A^T``.

It implements Julia's standard `AbstractMatrix{Float32}` interface and `LinearAlgebra.mul!`, making it directly compatible with iterative solvers in `LinearAlgebra`, `IterativeSolvers.jl`, and `Krylov.jl`.

```julia
using AstraToolbox
using LinearAlgebra

# Define geometry and operator
geom = Parallel2DGeometry(range(0, pi, length=180); detector_count=256)
A = AstraProjectorMap(geom, (256, 256))

# Forward projection: A * x
vol = shepp_logan(256)
sino = A * vol               # Returns 256x180 Array{Float32, 2}
sino_vec = A * vec(vol)      # Returns 1D Vector{Float32}

# In-place forward projection
mul!(y_buf, A, vec(vol))

# Backprojection (Adjoint): A' * y
bp = A' * sino               # Returns 256x256 Array{Float32, 2}
bp_vec = A' * sino_vec       # Returns 1D Vector{Float32}

# In-place backprojection
mul!(x_buf, A', sino_vec)

# 3D Operators
geom3d = Cone3DGeometry(range(0, 2pi, length=180), 256, 256, 500, 500)
A3d = AstraProjectorMap(geom3d, (256, 256, 256); backend=cuda())
sino3d = A3d * vol3d
bp3d   = A3d' * sino3d
```


