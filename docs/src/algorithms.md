# Algorithms

`AstraToolbox.jl` provides access to several analytical and iterative reconstruction algorithms.

## Analytical Algorithms

### `FBPMethod` (2D)

Filtered Back-Projection for 2D parallel and fan-flat geometries.

- `filter_type::Int`: Type of filter (e.g., `FILTER_RAMLAK`, `FILTER_SHEPPLOGAN`).
- `custom_filter::Vector{Float32}`: An optional custom filter in the frequency domain.

### `FDK3DMethod` (3D)

Feldkamp-Davis-Kress algorithm for 3D cone-beam reconstruction.

- `filter_type::Int`: Filter type.
- `custom_filter::Vector{Float32}`: Optional custom filter.

## Iterative Algorithms

Iterative algorithms are available for both 2D and 3D.

### 2D Iterative Algorithms

- **`SIRTMethod`**: Simultaneous Iterative Reconstruction Technique.
- **`SARTMethod`**: Simultaneous Algebraic Reconstruction Technique.
- **`CGLSMethod`**: Conjugate Gradient Least Squares.
- **`EMMethod`**: Expectation Maximization.

Common parameters:
- `iterations::Int`: Number of iterations to perform.

### 3D Iterative Algorithms

- **`SIRT3DMethod`**: 3D SIRT.
- **`CGLS3DMethod`**: 3D CGLS.

## Using Algorithms

To perform a reconstruction, use the `reconstruct` or `reconstruct!` functions.

```julia
method = SIRTMethod(iterations = 100)
reconstruction = reconstruct(sinogram, geometry, method; volume_size = (256, 256), backend = cuda())
```

## Reconstruction Options

You can fine-tune the reconstruction using `ReconstructionOptions2D` or `ReconstructionOptions3D`. Options include:
- `min_constraint`, `max_constraint`: Bound constraints on the reconstruction values.
- `reconstruction_mask`: Spatial mask for the reconstruction.
- `sinogram_mask`: Mask for the projection data.

## Detailed Information

For more information on each algorithm and its parameters, see the official ASTRA Toolbox documentation:
- [2D Algorithms](https://www.astra-toolbox.com/docs/algos2d.html)
- [3D Algorithms](https://www.astra-toolbox.com/docs/algos3d.html)
