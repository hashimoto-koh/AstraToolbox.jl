# 3D Geometries

`AstraToolbox.jl` supports several 3D geometries for defining the setup of your tomographic system.

## Parallel Beam Geometries

### `Parallel3DGeometry`

This geometry defines a standard parallel beam setup in 3D.

- `detector_width_u::Float32`: Width of a single detector pixel in the horizontal direction.
- `detector_width_v::Float32`: Width of a single detector pixel in the vertical direction.
- `detector_count_u::Int32`: Number of detector pixels in the horizontal direction.
- `detector_count_v::Int32`: Number of detector pixels in the vertical direction.
- `angles::Vector{Float32}`: The rotation angles (in radians).

### `ParallelVec3DGeometry`

A flexible vector geometry for parallel beams in 3D.

- `vectors::Matrix{Float32}`: A matrix of size $12 \times N_{angles}$. Each column contains $(s_x, s_y, s_z, d_x, d_y, d_z, u_x, u_y, u_z, v_x, v_y, v_z)$, where $(s_x, s_y, s_z)$ is the source position, $(d_x, d_y, d_z)$ is the detector center, $(u_x, u_y, u_z)$ is the horizontal detector pixel vector, and $(v_x, v_y, v_z)$ is the vertical detector pixel vector.
- `detector_count_u::Int32`: Number of detector pixels in the horizontal direction.
- `detector_count_v::Int32`: Number of detector pixels in the vertical direction.

## Cone Beam Geometries

### `Cone3DGeometry`

This geometry defines a cone-beam setup with a flat detector.

- `detector_width_u::Float32`: Horizontal pixel width.
- `detector_width_v::Float32`: Vertical pixel width.
- `detector_count_u::Int32`: Horizontal pixel count.
- `detector_count_v::Int32`: Vertical pixel count.
- `angles::Vector{Float32}`: Rotation angles (in radians).
- `source_origin::Float32`: Distance from the source to the origin.
- `origin_detector::Float32`: Distance from the origin to the detector center.

### `ConeVec3DGeometry`

A vector version of the cone-beam geometry.

- `vectors::Matrix{Float32}`: A matrix of size $12 \times N_{angles}$ defining the geometry for each angle.
- `detector_count_u::Int32`: Horizontal pixel count.
- `detector_count_v::Int32`: Vertical pixel count.

### `CylConeVec3DGeometry`

A vector geometry for a cone-beam setup with a cylindrical detector.

- `vectors::Matrix{Float32}`: A matrix of size $12 \times N_{angles}$.
- `detector_count_u::Int32`: Horizontal pixel count.
- `detector_count_v::Int32`: Vertical pixel count.

## Creating Geometries

Geometries are created by calling their respective constructors. For example:

```julia
# 3D cone beam geometry
angles = range(0, 2*pi, length=360)
geometry = Cone3DGeometry(
    angles;
    detector_width_u=1.0,
    detector_width_v=1.0,
    detector_count_u=256,
    detector_count_v=256,
    source_origin=1000.0,
    origin_detector=500.0
)
```

## Detailed Information

For more detailed information on the mathematical models and coordinate systems used by each geometry, refer to the [official ASTRA Toolbox documentation on 3D geometries](https://www.astra-toolbox.com/docs/geom3d.html).
