# 3D Geometries

`AstraToolbox.jl` supports several 3D geometries for defining the setup of your tomographic system.

## Parallel Beam Geometries

### `Parallel3DGeometry`

This geometry defines a standard parallel beam setup in 3D.

- `angles::Vector{Float32}`: The rotation angles (in radians).
- `detector_count_u::Integer`: Number of detector pixels in the horizontal direction.
- `detector_count_v::Integer`: Number of detector pixels in the vertical direction.
- `detector_spacing_u::Real`: Horizontal pixel spacing / width (defaults to `1.0f0`).
- `detector_spacing_v::Real`: Vertical pixel spacing / width (defaults to `1.0f0`).

### `ParallelVec3DGeometry`

A flexible vector geometry for parallel beams in 3D.

- `vectors::Matrix{Float32}`: A matrix of size $12 \times N_{angles}$. Each column contains $(s_x, s_y, s_z, d_x, d_y, d_z, u_x, u_y, u_z, v_x, v_y, v_z)$, where $(s_x, s_y, s_z)$ is the source position, $(d_x, d_y, d_z)$ is the detector center, $(u_x, u_y, u_z)$ is the horizontal detector pixel vector, and $(v_x, v_y, v_z)$ is the vertical detector pixel vector.
- `detector_count_u::Int32`: Number of detector pixels in the horizontal direction.
- `detector_count_v::Int32`: Number of detector pixels in the vertical direction.

## Cone Beam Geometries

### `Cone3DGeometry`

This geometry defines a cone-beam setup with a flat detector.

- `angles::Vector{Float32}`: Rotation angles (in radians).
- `detector_count_u::Integer`: Horizontal pixel count.
- `detector_count_v::Integer`: Vertical pixel count.
- `origin_source_distance::Real`: Distance from the origin to the X-ray source.
- `origin_detector_distance::Real`: Distance from the origin to the detector plane center.
- `detector_spacing_u::Real`: Horizontal pixel spacing / width (defaults to `1.0f0`).
- `detector_spacing_v::Real`: Vertical pixel spacing / width (defaults to `1.0f0`).

### `ConeVec3DGeometry`

A vector version of the cone-beam geometry.

- `vectors::Matrix{Float32}`: A matrix of size $12 \times N_{angles}$ defining the geometry for each angle.
- `detector_count_u::Int32`: Horizontal pixel count.
- `detector_count_v::Int32`: Vertical pixel count.

### `CylConeVec3DGeometry`

A vector geometry for a cone-beam setup with a cylindrical detector.

- `vectors::Matrix{Float32}`: A matrix of size $13 \times N_{angles}$.
- `detector_count_u::Int32`: Horizontal pixel count.
- `detector_count_v::Int32`: Vertical pixel count.

## Creating Geometries

Geometries are created by calling their respective constructors. For example:

```julia
# 3D cone beam geometry
angles = range(0, 2*pi, length=360)
geometry = Cone3DGeometry(
    angles;
    detector_count_u = 256,
    detector_count_v = 256,
    origin_source_distance = 1000.0,
    origin_detector_distance = 500.0,
    detector_spacing_u = 1.0,
    detector_spacing_v = 1.0
)
```

## Detailed Information

For more detailed information on the mathematical models and coordinate systems used by each geometry, refer to the [official ASTRA Toolbox documentation on 3D geometries](https://www.astra-toolbox.com/docs/geom3d.html).
