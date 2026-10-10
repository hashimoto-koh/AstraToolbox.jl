# Quick Start

This page provides minimal examples for common tasks in `AstraToolbox.jl`.

## 2D Forward Projection (CPU)

```julia
using AstraToolbox

# 2x2 volume
volume = reshape(Float32[1, 2, 3, 4], 2, 2)

# Single projection at 0 radians
angles = Float32[0.0]

# Parallel 2D geometry
geometry = Parallel2DGeometry(angles; detector_count = 2, detector_width = 1.0f0)

# Forward project using CPU backend
sinogram = project(volume, geometry; backend = cpu())

# Result is a 2x1 Float32 array
println(sinogram)
```

## 2D Reconstruction (FBP, CPU)

```julia
using AstraToolbox

# Geometries at 0 and pi/2 radians
angles = Float32[0.0, pi / 2]
geometry = Parallel2DGeometry(angles; detector_count = 2, detector_width = 1.0f0)

# 2x2 sinogram (u=2, angles=2)
sinogram = reshape(Float32[4, 6, 3, 7], 2, 2)

# FBP reconstruction on CPU
method = FBPMethod()
reconstruction = reconstruct(sinogram, geometry, method; volume_size = (2, 2), backend = cpu())

println(reconstruction)
```

## Using CUDA Backend

To use the GPU backend, simply replace `backend = cpu()` with `backend = cuda()`.

```julia
# Forward project on GPU
sinogram_gpu = project(volume, geometry; backend = cuda())

# Reconstruction on GPU
reconstruction_gpu = reconstruct(sinogram, geometry, method; volume_size = (2, 2), backend = cuda())
```

Note: `cuda()` accepts an optional `gpu_index` parameter (default is `-1`, which uses the default device).
