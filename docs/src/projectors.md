# 2D and 3D Projectors

In `AstraToolbox.jl`, projection and backprojection are performed using the `project` and `backproject` functions.

This stateless approach differs from the original ASTRA Toolbox, where you often need to create and manage `Projector` objects explicitly.

## Stateless Projection API

You don't need to create a `Projector` object. Instead, you provide the geometry and the data directly to the projection functions.

### `project` and `project!`

Forward projection computes the sinogram from a volume.

```julia
# Non-bang version (allocates a new sinogram)
sinogram = project(volume, geometry; backend = cpu())

# Bang version (updates a pre-allocated sinogram)
project!(sinogram, volume, geometry; backend = cuda())
```

### `backproject` and `backproject!`

Backprojection computes the adjoint operation, mapping a sinogram back onto a volume.

```julia
# Non-bang version (allocates a new volume)
volume = backproject(sinogram, geometry; volume_size = (Nx, Ny), backend = cpu())

# Bang version (updates a pre-allocated volume)
backproject!(volume, sinogram, geometry; backend = cuda())
```

## Projection Options

For fine-grained control over the projection process, you can use `ProjectionOptions2D`, `ProjectionOptions3D`, `BackprojectionOptions2D`, or `BackprojectionOptions3D`.

These options allow you to specify things like the GPU index or other backend-specific parameters.

```julia
options = ProjectionOptions2D(gpu_index = 0)
sinogram = project(volume, geometry; backend = cuda(), options = options)
```

## Internal Projector Types

While the user-facing API is stateless, ASTRA internally uses different projector types (e.g., `line`, `linear`, `strip`). `AstraToolbox.jl` currently uses the most common defaults for each geometry and backend:

- **2D CPU**: Line projector.
- **2D CUDA**: Linear projector.
- **3D CUDA**: Standard CUDA projector for parallel/cone beams.

## Detailed Information

For more information on the underlying projector models, see the official ASTRA Toolbox documentation:
- [2D Projectors](https://www.astra-toolbox.com/docs/proj2d.html)
- [3D Projectors](https://www.astra-toolbox.com/docs/proj3d.html)
