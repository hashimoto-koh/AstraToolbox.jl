# Algorithms

`AstraToolbox.jl` provides access to several analytical and iterative reconstruction algorithms.

## Analytical Algorithms

### `FBPMethod` (2D)

Filtered Back-Projection for 2D parallel and fan-flat geometries.

- `filter_type::Integer`: Window filter type (e.g., `FILTER_RAMLAK`, `FILTER_SHEPPLOGAN`, `FILTER_PROJECTION`, defaults to `FILTER_RAMLAK`).
- `filter_d::Real`: Filter scaling parameter (defaults to `1.0f0`).
- `filter_parameter::Real`: Parameter for configurable window filters such as Tukey or Kaiser (defaults to `-1.0f0`).
- `custom_filter`: Custom frequency-domain filter array (Matrix/Vector of `Float32`) when using `FILTER_PROJECTION` or `FILTER_SINOGRAM`.
- `short_scan::Bool`: If `true`, applies Parker weighting for fan-beam short-scan geometries covering less than $360^\circ$ (defaults to `false`).

### `FDK3DMethod` (3D)

Feldkamp-Davis-Kress algorithm for 3D cone-beam reconstruction.

- `filter_type::Integer`: Window filter type (defaults to `FILTER_RAMLAK`).
- `filter_d::Real`: Filter scaling parameter (defaults to `1.0f0`).
- `filter_parameter::Real`: Filter parameter for adjustable windows (defaults to `-1.0f0`).
- `custom_filter`: Optional custom frequency-domain filter array.
- `short_scan::Bool`: If `true`, applies Parker weighting for short-scan cone-beam geometries (defaults to `false`).
- `voxel_supersampling::Integer`: Voxel supersampling rate along each dimension for backprojection (defaults to `1`).

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
