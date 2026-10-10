# 2D Geometries

`AstraToolbox.jl` supports several 2D geometries for defining the setup of your tomographic system.

## Parallel Beam Geometries

### `Parallel2DGeometry`

This geometry defines a standard parallel beam setup.

- `angles::Vector{Float32}`: The rotation angles (in radians).
- `detector_count::Int32`: Number of detector pixels.
- `detector_width::Float32`: Width of a single detector pixel.

### `ParallelVec2DGeometry`

A more flexible "vector" geometry for parallel beams, where each projection can have its own source and detector configuration.

- `vectors::Matrix{Float32}`: A matrix of size $6 \times N_{angles}$. Each column contains $(s_x, s_y, d_x, d_y, u_x, u_y)$, where $(s_x, s_y)$ is the source position, $(d_x, d_y)$ is the detector center, and $(u_x, u_y)$ is the detector pixel vector.
- `detector_count::Int32`: Number of detector pixels.

## Fan Beam Geometries

### `FanFlat2DGeometry`

This geometry defines a fan-beam setup with a flat detector.

- `detector_width::Float32`: Width of a single detector pixel.
- `source_origin::Float32`: Distance from the source to the origin.
- `origin_detector::Float32`: Distance from the origin to the detector center.
- `angles::Vector{Float32}`: The rotation angles (in radians).
- `detector_count::Int32`: Number of detector pixels.

### `FanFlatVec2DGeometry`

A vector version of the fan-flat geometry.

- `vectors::Matrix{Float32}`: A matrix of size $6 \times N_{angles}$. Each column contains $(s_x, s_y, d_x, d_y, u_x, u_y)$, where $(s_x, s_y)$ is the source position, $(d_x, d_y)$ is the detector center, and $(u_x, u_y)$ is the detector pixel vector.
- `detector_count::Int32`: Number of detector pixels.

## Creating Geometries

Geometries are created by calling their respective constructors. For example:

```julia
# 2D parallel beam geometry with 180 projections
angles = range(0, pi, length=180)
geometry = Parallel2DGeometry(angles; detector_count=512, detector_width=1.0)
```

## Detailed Information

For more detailed information on the mathematical models and coordinate systems used by each geometry, refer to the [official ASTRA Toolbox documentation on 2D geometries](https://www.astra-toolbox.com/docs/geom2d.html).
