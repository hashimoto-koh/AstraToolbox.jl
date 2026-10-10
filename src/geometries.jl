"""
    Parallel2DGeometry(angles; detector_count, detector_width=1.0)

Create a 2D parallel-beam geometry.

# Arguments
- `angles`: Rotation angles in radians.
- `detector_count`: Number of detector pixels.
- `detector_width`: Width of a single detector pixel. Defaults to 1.0.

# Returns
- `Parallel2DGeometry`: 2D parallel-beam geometry object.

# Examples
```julia
geom = Parallel2DGeometry(range(0, pi, length=180); detector_count=256)
```
"""
function Parallel2DGeometry(
    angles::AbstractVector{<:Real};
    detector_count::Integer,
    detector_width::Real = 1.0f0,
)
    detector_count > 0 || throw(ArgumentError("detector_count must be > 0"))
    length(angles) > 0 || throw(ArgumentError("angles must be non-empty"))
    detector_width > 0 || throw(ArgumentError("detector_width must be > 0"))
    return Parallel2DGeometry(Float32.(angles), Int(detector_count), Float32(detector_width))
end

"""
    ParallelVec2DGeometry(vectors; detector_count)

Create a 2D parallel-beam vector geometry.

# Arguments
- `vectors`: 6xN matrix of vectors defining source and detector positions.
- `detector_count`: Number of detector pixels.

# Returns
- `ParallelVec2DGeometry`: 2D parallel-beam vector geometry object.

# Examples
```julia
vectors = zeros(Float32, 6, 180) # ... fill vectors ...
geom = ParallelVec2DGeometry(vectors; detector_count=256)
```
"""
function ParallelVec2DGeometry(
    vectors::AbstractMatrix{<:Real};
    detector_count::Integer,
)
    detector_count > 0 || throw(ArgumentError("detector_count must be > 0"))
    size(vectors, 1) == 6 || throw(ArgumentError("size(vectors, 1) must be 6"))
    size(vectors, 2) > 0 || throw(ArgumentError("vectors must have at least one angle"))
    return ParallelVec2DGeometry(Float32.(vectors), Int(detector_count))
end

"""
    FanFlat2DGeometry(angles, detector_count, origin_source_distance, origin_detector_distance; detector_width=1.0)

Create a 2D fan-beam flat-detector geometry.

# Arguments
- `angles`: Rotation angles in radians.
- `detector_count`: Number of detector pixels.
- `origin_source_distance`: Distance from origin to source.
- `origin_detector_distance`: Distance from origin to detector center.
- `detector_width`: Width of a single detector pixel. Defaults to 1.0.

# Returns
- `FanFlat2DGeometry`: 2D fan-beam flat-detector geometry object.

# Examples
```julia
geom = FanFlat2DGeometry(range(0, 2pi, length=360), 256, 500, 500)
```
"""
function FanFlat2DGeometry(
    angles::AbstractVector{<:Real},
    detector_count::Integer,
    origin_source_distance::Real,
    origin_detector_distance::Real;
    detector_width::Real = 1.0f0,
)
    detector_count > 0 || throw(ArgumentError("detector_count must be > 0"))
    length(angles) > 0 || throw(ArgumentError("angles must be non-empty"))
    detector_width > 0 || throw(ArgumentError("detector_width must be > 0"))
    origin_source_distance > 0 || throw(ArgumentError("origin_source_distance must be > 0"))
    origin_detector_distance > 0 || throw(ArgumentError("origin_detector_distance must be > 0"))
    return FanFlat2DGeometry(
        Float32.(angles),
        Int(detector_count),
        Float32(detector_width),
        Float32(origin_source_distance),
        Float32(origin_detector_distance),
    )
end

function FanFlat2DGeometry(
    angles::AbstractVector{<:Real};
    detector_count::Integer,
    origin_source_distance::Real,
    origin_detector_distance::Real,
    detector_width::Real = 1.0f0,
)
    return FanFlat2DGeometry(
        angles,
        detector_count,
        origin_source_distance,
        origin_detector_distance;
        detector_width = detector_width,
    )
end

"""
    FanFlatVec2DGeometry(vectors; detector_count)

Create a 2D fan-beam flat-detector vector geometry.

# Arguments
- `vectors`: 6xN matrix of vectors defining source and detector positions.
- `detector_count`: Number of detector pixels.

# Returns
- `FanFlatVec2DGeometry`: 2D fan-beam vector geometry object.

# Examples
```julia
vectors = zeros(Float32, 6, 360) # ... fill vectors ...
geom = FanFlatVec2DGeometry(vectors; detector_count=256)
```
"""
function FanFlatVec2DGeometry(
    vectors::AbstractMatrix{<:Real};
    detector_count::Integer,
)
    detector_count > 0 || throw(ArgumentError("detector_count must be > 0"))
    size(vectors, 1) == 6 || throw(ArgumentError("size(vectors, 1) must be 6"))
    size(vectors, 2) > 0 || throw(ArgumentError("vectors must have at least one angle"))
    return FanFlatVec2DGeometry(Float32.(vectors), Int(detector_count))
end

"""
    SparseMatrix2DGeometry(matrix; detector_count, angle_count=size(matrix, 1) ÷ detector_count, volume_size=nothing)

Create a 2D sparse matrix projection geometry from a sparse matrix (e.g., `SparseMatrixCSC`).

# Arguments
- `matrix`: A sparse projection matrix (or abstract matrix) mapping a flattened volume of size `(volume_width, volume_height)` (length `N = volume_width * volume_height`)
  to a flattened sinogram of size `(detector_count, angle_count)` (length `M = detector_count * angle_count`).
- `detector_count`: Number of detector pixels per projection angle.
- `angle_count`: Number of projection angles. Defaults to `size(matrix, 1) ÷ detector_count`.
- `volume_size`: Optional `(volume_width, volume_height)` tuple. Defaults to `(s, s)` if `N` is a square, or `(N, 1)`.

# Returns
- `SparseMatrix2DGeometry`: 2D sparse matrix geometry object.

# Examples
```julia
using SparseArrays
A = sprand(Float32, 32 * 16, 32 * 32, 0.05)
geom = SparseMatrix2DGeometry(A; detector_count=32, angle_count=16, volume_size=(32, 32))
```
"""
function SparseMatrix2DGeometry(
    matrix::SparseMatrixCSC{<:Real};
    detector_count::Integer,
    angle_count::Integer = size(matrix, 1) ÷ detector_count,
    volume_size::Union{Nothing, Tuple{Integer, Integer}} = nothing,
)
    detector_count > 0 || throw(ArgumentError("detector_count must be > 0"))
    angle_count > 0 || throw(ArgumentError("angle_count must be > 0"))
    size(matrix, 1) == angle_count * detector_count ||
        throw(ArgumentError("size(matrix, 1) ($(size(matrix, 1))) must equal angle_count * detector_count ($angle_count * $detector_count = $(angle_count * detector_count))"))

    M, N = size(matrix)
    vol_w, vol_h = if volume_size !== nothing
        w, h = volume_size
        w * h == N || throw(ArgumentError("volume_size ($w, $h) product does not match size(matrix, 2) ($N)"))
        Int(w), Int(h)
    else
        s = isqrt(N)
        if s * s == N
            s, s
        else
            N, 1
        end
    end

    mat_f32 = SparseMatrixCSC{Float32, Int}(matrix)
    mat_T = copy(transpose(mat_f32))
    row_starts = UInt64.(mat_T.colptr .- 1)
    col_indices = UInt32.(mat_T.rowval .- 1)
    values = Vector{Float32}(mat_T.nzval)

    return SparseMatrix2DGeometry(
        Int(angle_count),
        Int(detector_count),
        vol_w,
        vol_h,
        mat_f32,
        row_starts,
        col_indices,
        values,
    )
end

function SparseMatrix2DGeometry(
    matrix::AbstractMatrix{<:Real};
    detector_count::Integer,
    angle_count::Integer = size(matrix, 1) ÷ detector_count,
    volume_size::Union{Nothing, Tuple{Integer, Integer}} = nothing,
)
    return SparseMatrix2DGeometry(
        sparse(matrix);
        detector_count = detector_count,
        angle_count = angle_count,
        volume_size = volume_size,
    )
end

Sparse2DGeometry(args...; kwargs...) = SparseMatrix2DGeometry(args...; kwargs...)

"""
    Parallel3DGeometry(angles; detector_count_u, detector_count_v, detector_spacing_u=1.0, detector_spacing_v=1.0)

Create a 3D parallel-beam geometry.

# Arguments
- `angles`: Rotation angles in radians.
- `detector_count_u`: Number of detector pixels (horizontal).
- `detector_count_v`: Number of detector pixels (vertical).
- `detector_spacing_u`: Horizontal pixel spacing. Defaults to 1.0.
- `detector_spacing_v`: Vertical pixel spacing. Defaults to 1.0.

# Returns
- `Parallel3DGeometry`: 3D parallel-beam geometry object.

# Examples
```julia
geom = Parallel3DGeometry(range(0, pi, length=180); detector_count_u=256, detector_count_v=256)
```
"""
function Parallel3DGeometry(
    angles::AbstractVector{<:Real};
    detector_count_u::Integer,
    detector_count_v::Integer,
    detector_spacing_u::Real = 1.0f0,
    detector_spacing_v::Real = 1.0f0,
)
    detector_count_u > 0 || throw(ArgumentError("detector_count_u must be > 0"))
    detector_count_v > 0 || throw(ArgumentError("detector_count_v must be > 0"))
    length(angles) > 0 || throw(ArgumentError("angles must be non-empty"))
    detector_spacing_u > 0 || throw(ArgumentError("detector_spacing_u must be > 0"))
    detector_spacing_v > 0 || throw(ArgumentError("detector_spacing_v must be > 0"))
    return Parallel3DGeometry(
        Float32.(angles),
        Int(detector_count_u),
        Int(detector_count_v),
        Float32(detector_spacing_u),
        Float32(detector_spacing_v),
    )
end

"""
    ParallelVec3DGeometry(vectors; detector_count_u, detector_count_v)

Create a 3D parallel-beam vector geometry.

# Arguments
- `vectors`: 12xN matrix defining geometry.
- `detector_count_u`: Horizontal detector pixel count.
- `detector_count_v`: Vertical detector pixel count.

# Returns
- `ParallelVec3DGeometry`: 3D parallel-beam vector geometry object.

# Examples
```julia
vectors = zeros(Float32, 12, 180) # ... fill vectors ...
geom = ParallelVec3DGeometry(vectors; detector_count_u=256, detector_count_v=256)
```
"""
function ParallelVec3DGeometry(
    vectors::AbstractMatrix{<:Real};
    detector_count_u::Integer,
    detector_count_v::Integer,
)
    detector_count_u > 0 || throw(ArgumentError("detector_count_u must be > 0"))
    detector_count_v > 0 || throw(ArgumentError("detector_count_v must be > 0"))
    size(vectors, 1) == 12 || throw(ArgumentError("size(vectors, 1) must be 12"))
    size(vectors, 2) > 0 || throw(ArgumentError("vectors must have at least one angle"))
    return ParallelVec3DGeometry(Float32.(vectors), Int(detector_count_u), Int(detector_count_v))
end

"""
    Cone3DGeometry(angles, detector_count_u, detector_count_v, origin_source_distance, origin_detector_distance; detector_spacing_u=1.0, detector_spacing_v=1.0)

Create a 3D cone-beam geometry.

# Arguments
- `angles`: Rotation angles in radians.
- `detector_count_u`: Horizontal detector pixel count.
- `detector_count_v`: Vertical detector pixel count.
- `origin_source_distance`: Distance from origin to source.
- `origin_detector_distance`: Distance from origin to detector center.
- `detector_spacing_u`: Horizontal pixel spacing. Defaults to 1.0.
- `detector_spacing_v`: Vertical pixel spacing. Defaults to 1.0.

# Returns
- `Cone3DGeometry`: 3D cone-beam geometry object.

# Examples
```julia
geom = Cone3DGeometry(range(0, 2pi, length=360), 256, 256, 500, 500)
```
"""
function Cone3DGeometry(
    angles::AbstractVector{<:Real},
    detector_count_u::Integer,
    detector_count_v::Integer,
    origin_source_distance::Real,
    origin_detector_distance::Real;
    detector_spacing_u::Real = 1.0f0,
    detector_spacing_v::Real = 1.0f0,
)
    detector_count_u > 0 || throw(ArgumentError("detector_count_u must be > 0"))
    detector_count_v > 0 || throw(ArgumentError("detector_count_v must be > 0"))
    length(angles) > 0 || throw(ArgumentError("angles must be non-empty"))
    detector_spacing_u > 0 || throw(ArgumentError("detector_spacing_u must be > 0"))
    detector_spacing_v > 0 || throw(ArgumentError("detector_spacing_v must be > 0"))
    origin_source_distance > 0 || throw(ArgumentError("origin_source_distance must be > 0"))
    origin_detector_distance > 0 || throw(ArgumentError("origin_detector_distance must be > 0"))
    return Cone3DGeometry(
        Float32.(angles),
        Int(detector_count_u),
        Int(detector_count_v),
        Float32(detector_spacing_u),
        Float32(detector_spacing_v),
        Float32(origin_source_distance),
        Float32(origin_detector_distance),
    )
end

function Cone3DGeometry(
    angles::AbstractVector{<:Real};
    detector_count_u::Integer,
    detector_count_v::Integer,
    origin_source_distance::Real,
    origin_detector_distance::Real,
    detector_spacing_u::Real = 1.0f0,
    detector_spacing_v::Real = 1.0f0,
)
    return Cone3DGeometry(
        angles,
        detector_count_u,
        detector_count_v,
        origin_source_distance,
        origin_detector_distance;
        detector_spacing_u = detector_spacing_u,
        detector_spacing_v = detector_spacing_v,
    )
end

"""
    ConeVec3DGeometry(vectors; detector_count_u, detector_count_v)

Create a 3D cone-beam vector geometry.

# Arguments
- `vectors`: 12xN matrix defining geometry.
- `detector_count_u`: Horizontal detector pixel count.
- `detector_count_v`: Vertical detector pixel count.

# Returns
- `ConeVec3DGeometry`: 3D cone-beam vector geometry object.

# Examples
```julia
vectors = zeros(Float32, 12, 360) # ... fill vectors ...
geom = ConeVec3DGeometry(vectors; detector_count_u=256, detector_count_v=256)
```
"""
function ConeVec3DGeometry(
    vectors::AbstractMatrix{<:Real};
    detector_count_u::Integer,
    detector_count_v::Integer,
)
    detector_count_u > 0 || throw(ArgumentError("detector_count_u must be > 0"))
    detector_count_v > 0 || throw(ArgumentError("detector_count_v must be > 0"))
    size(vectors, 1) == 12 || throw(ArgumentError("size(vectors, 1) must be 12"))
    size(vectors, 2) > 0 || throw(ArgumentError("vectors must have at least one angle"))
    return ConeVec3DGeometry(Float32.(vectors), Int(detector_count_u), Int(detector_count_v))
end

"""
    CylConeVec3DGeometry(vectors; detector_count_u, detector_count_v)

Create a 3D cone-beam vector geometry for cylindrical detectors.

# Arguments
- `vectors`: 13xN matrix defining geometry.
- `detector_count_u`: Horizontal detector pixel count.
- `detector_count_v`: Vertical detector pixel count.

# Returns
- `CylConeVec3DGeometry`: 3D cylindrical detector vector geometry object.

# Examples
```julia
vectors = zeros(Float32, 13, 360) # ... fill vectors ...
geom = CylConeVec3DGeometry(vectors; detector_count_u=256, detector_count_v=256)
```
"""
function CylConeVec3DGeometry(
    vectors::AbstractMatrix{<:Real};
    detector_count_u::Integer,
    detector_count_v::Integer,
)
    detector_count_u > 0 || throw(ArgumentError("detector_count_u must be > 0"))
    detector_count_v > 0 || throw(ArgumentError("detector_count_v must be > 0"))
    size(vectors, 1) == 13 || throw(ArgumentError("size(vectors, 1) must be 13"))
    size(vectors, 2) > 0 || throw(ArgumentError("vectors must have at least one angle"))
    return CylConeVec3DGeometry(Float32.(vectors), Int(detector_count_u), Int(detector_count_v))
end

"""
    VolumeGeometry2D(width, height; window=nothing, pixel_size=nothing, center=(0.0, 0.0))
    VolumeGeometry2D((width, height); window=nothing, pixel_size=nothing, center=(0.0, 0.0))

Create a 2D volume geometry defining grid dimensions and spatial coordinates.

# Arguments
- `width`: Number of grid columns (X dimension).
- `height`: Number of grid rows (Y dimension).

# Keywords
- `window`: Optional 4-element collection `(min_x, max_x, min_y, max_y)`.
- `pixel_size`: Optional scalar or 2-element collection `(ps_x, ps_y)`.
- `center`: Optional 2-element collection `(cx, cy)` representing the center coordinates. Defaults to `(0.0, 0.0)`.

# Returns
- `VolumeGeometry2D`: 2D volume geometry object.

# Examples
```julia
# Standard centered volume with pixel size 1.0
vg = VolumeGeometry2D(256, 256)

# Shifted ROI window
vg_roi = VolumeGeometry2D(256, 256; window=(-10.0, 10.0, -5.0, 15.0))

# Physical scale with custom pixel size and center offset
vg_scaled = VolumeGeometry2D((256, 256); pixel_size=0.25, center=(0.0, 5.0))
```
"""
function VolumeGeometry2D(
    width::Integer,
    height::Integer;
    window = nothing,
    pixel_size = nothing,
    center = (0.0f0, 0.0f0),
)
    w = Int(width)
    h = Int(height)
    w > 0 && h > 0 || throw(ArgumentError("volume dimensions width and height must be positive"))

    if window !== nothing && pixel_size !== nothing
        throw(ArgumentError("cannot specify both `window` and `pixel_size`"))
    end

    if window !== nothing
        length(window) == 4 || throw(ArgumentError("window must be a 4-element collection (min_x, max_x, min_y, max_y)"))
        min_x, max_x, min_y, max_y = Float32.(window)
        min_x < max_x || throw(ArgumentError("window_min_x must be strictly less than window_max_x"))
        min_y < max_y || throw(ArgumentError("window_min_y must be strictly less than window_max_y"))
        return VolumeGeometry2D(w, h, min_x, min_y, max_x, max_y)
    elseif pixel_size !== nothing
        ps_x, ps_y = if pixel_size isa Tuple || pixel_size isa AbstractVector
            length(pixel_size) == 2 || throw(ArgumentError("pixel_size tuple must have 2 elements (ps_x, ps_y)"))
            Float32(pixel_size[1]), Float32(pixel_size[2])
        else
            Float32(pixel_size), Float32(pixel_size)
        end
        ps_x > 0 && ps_y > 0 || throw(ArgumentError("pixel_size must be positive"))
        cx, cy = Float32(center[1]), Float32(center[2])
        min_x = cx - (w * ps_x) / 2.0f0
        max_x = cx + (w * ps_x) / 2.0f0
        min_y = cy - (h * ps_y) / 2.0f0
        max_y = cy + (h * ps_y) / 2.0f0
        return VolumeGeometry2D(w, h, min_x, min_y, max_x, max_y)
    else
        min_x = -Float32(w) / 2.0f0
        max_x = Float32(w) / 2.0f0
        min_y = -Float32(h) / 2.0f0
        max_y = Float32(h) / 2.0f0
        return VolumeGeometry2D(w, h, min_x, min_y, max_x, max_y)
    end
end

VolumeGeometry2D(dims::Tuple{Integer,Integer}; kwargs...) = VolumeGeometry2D(dims[1], dims[2]; kwargs...)

"""
    VolumeGeometry3D(width, height, depth; window=nothing, voxel_size=nothing, center=(0.0, 0.0, 0.0))
    VolumeGeometry3D((width, height, depth); window=nothing, voxel_size=nothing, center=(0.0, 0.0, 0.0))

Create a 3D volume geometry defining grid dimensions and spatial coordinates.

# Arguments
- `width`: Number of grid columns (X dimension).
- `height`: Number of grid rows (Y dimension).
- `depth`: Number of grid slices (Z dimension).

# Keywords
- `window`: Optional 6-element collection `(min_x, max_x, min_y, max_y, min_z, max_z)`.
- `voxel_size`: Optional scalar or 3-element collection `(vs_x, vs_y, vs_z)`.
- `center`: Optional 3-element collection `(cx, cy, cz)`. Defaults to `(0.0, 0.0, 0.0)`.

# Returns
- `VolumeGeometry3D`: 3D volume geometry object.

# Examples
```julia
# Standard centered 3D volume
vg = VolumeGeometry3D(128, 128, 64)

# Bounding box window
vg_roi = VolumeGeometry3D(128, 128, 64; window=(-10.0, 10.0, -10.0, 10.0, -5.0, 5.0))
```
"""
function VolumeGeometry3D(
    width::Integer,
    height::Integer,
    depth::Integer;
    window = nothing,
    voxel_size = nothing,
    center = (0.0f0, 0.0f0, 0.0f0),
)
    w = Int(width)
    h = Int(height)
    d = Int(depth)
    w > 0 && h > 0 && d > 0 || throw(ArgumentError("volume dimensions width, height, and depth must be positive"))

    if window !== nothing && voxel_size !== nothing
        throw(ArgumentError("cannot specify both `window` and `voxel_size`"))
    end

    if window !== nothing
        length(window) == 6 || throw(ArgumentError("window must be a 6-element collection (min_x, max_x, min_y, max_y, min_z, max_z)"))
        min_x, max_x, min_y, max_y, min_z, max_z = Float32.(window)
        min_x < max_x || throw(ArgumentError("window_min_x must be strictly less than window_max_x"))
        min_y < max_y || throw(ArgumentError("window_min_y must be strictly less than window_max_y"))
        min_z < max_z || throw(ArgumentError("window_min_z must be strictly less than window_max_z"))
        return VolumeGeometry3D(w, h, d, min_x, min_y, min_z, max_x, max_y, max_z)
    elseif voxel_size !== nothing
        vs_x, vs_y, vs_z = if voxel_size isa Tuple || voxel_size isa AbstractVector
            length(voxel_size) == 3 || throw(ArgumentError("voxel_size tuple must have 3 elements (vs_x, vs_y, vs_z)"))
            Float32(voxel_size[1]), Float32(voxel_size[2]), Float32(voxel_size[3])
        else
            Float32(voxel_size), Float32(voxel_size), Float32(voxel_size)
        end
        vs_x > 0 && vs_y > 0 && vs_z > 0 || throw(ArgumentError("voxel_size must be positive"))
        cx, cy, cz = Float32(center[1]), Float32(center[2]), Float32(center[3])
        min_x = cx - (w * vs_x) / 2.0f0
        max_x = cx + (w * vs_x) / 2.0f0
        min_y = cy - (h * vs_y) / 2.0f0
        max_y = cy + (h * vs_y) / 2.0f0
        min_z = cz - (d * vs_z) / 2.0f0
        max_z = cz + (d * vs_z) / 2.0f0
        return VolumeGeometry3D(w, h, d, min_x, min_y, min_z, max_x, max_y, max_z)
    else
        min_x = -Float32(w) / 2.0f0
        max_x = Float32(w) / 2.0f0
        min_y = -Float32(h) / 2.0f0
        max_y = Float32(h) / 2.0f0
        min_z = -Float32(d) / 2.0f0
        max_z = Float32(d) / 2.0f0
        return VolumeGeometry3D(w, h, d, min_x, min_y, min_z, max_x, max_y, max_z)
    end
end

VolumeGeometry3D(dims::Tuple{Integer,Integer,Integer}; kwargs...) = VolumeGeometry3D(dims[1], dims[2], dims[3]; kwargs...)

"""
    create_vol_geom(args...)

Compatibility constructor matching upstream ASTRA `astra.create_vol_geom`.

# Arguments
- `args...`: Volume dimensions `N` (for square `N x N`), `(width, height)`, `(width, height, depth)`, or with explicit window bounds `(width, height, [depth], min_x, max_x, min_y, max_y, [min_z, max_z])`.

# Returns
- `VolumeGeometry2D` or `VolumeGeometry3D`.

# Examples
```julia
vg2d = create_vol_geom(256, 256)
vg3d = create_vol_geom(128, 128, 128)
```
"""
create_vol_geom(N::Integer) = VolumeGeometry2D(N, N)
create_vol_geom(dims::Tuple{Integer,Integer}) = VolumeGeometry2D(dims[1], dims[2])
create_vol_geom(dims::Tuple{Integer,Integer,Integer}) = VolumeGeometry3D(dims[1], dims[2], dims[3])
create_vol_geom(width::Integer, height::Integer) = VolumeGeometry2D(width, height)
create_vol_geom(width::Integer, height::Integer, depth::Integer) = VolumeGeometry3D(width, height, depth)
create_vol_geom(width::Integer, height::Integer, min_x, max_x, min_y, max_y) = VolumeGeometry2D(width, height; window=(min_x, max_x, min_y, max_y))
create_vol_geom(width::Integer, height::Integer, depth::Integer, min_x, max_x, min_y, max_y, min_z, max_z) = VolumeGeometry3D(width, height, depth; window=(min_x, max_x, min_y, max_y, min_z, max_z))

Base.size(vg::VolumeGeometry2D) = (vg.width, vg.height)
Base.size(vg::VolumeGeometry3D) = (vg.width, vg.height, vg.depth)

"""
    pixel_size(vg::VolumeGeometry2D) -> Tuple{Float32, Float32}

Get the physical pixel dimensions `(pixel_width, pixel_height)` from a `VolumeGeometry2D`.

# Arguments
- `vg`: `VolumeGeometry2D` object.

# Returns
- `Tuple{Float32, Float32}`: Pixel size in X and Y directions.

# Examples
```julia
vg = VolumeGeometry2D(256, 256; window=(-10.0, 10.0, -10.0, 10.0))
ps = pixel_size(vg)
```
"""
pixel_size(vg::VolumeGeometry2D) = ((vg.window_max_x - vg.window_min_x) / Float32(vg.width), (vg.window_max_y - vg.window_min_y) / Float32(vg.height))

"""
    voxel_size(vg::VolumeGeometry3D) -> Tuple{Float32, Float32, Float32}

Get the physical voxel dimensions `(voxel_x, voxel_y, voxel_z)` from a `VolumeGeometry3D`.

# Arguments
- `vg`: `VolumeGeometry3D` object.

# Returns
- `Tuple{Float32, Float32, Float32}`: Voxel size in X, Y, and Z directions.

# Examples
```julia
vg = VolumeGeometry3D(128, 128, 128; window=(-10.0, 10.0, -10.0, 10.0, -5.0, 5.0))
vs = voxel_size(vg)
```
"""
voxel_size(vg::VolumeGeometry3D) = ((vg.window_max_x - vg.window_min_x) / Float32(vg.width), (vg.window_max_y - vg.window_min_y) / Float32(vg.height), (vg.window_max_z - vg.window_min_z) / Float32(vg.depth))

"""
    center(vg::VolumeGeometry2D) -> NTuple{2, Float32}
    center(vg::VolumeGeometry3D) -> NTuple{3, Float32}

Compute the physical center coordinates of the reconstruction window defined by a volume geometry.

# Arguments
- `vg`: `VolumeGeometry2D` or `VolumeGeometry3D`.

# Returns
- `NTuple{2, Float32}` or `NTuple{3, Float32}`: Coordinates `(cx, cy)` or `(cx, cy, cz)` of the window center.

# Examples
```julia
vg = VolumeGeometry2D(256, 256; center=(10.0, -5.0))
c = center(vg)
```
"""
center(vg::VolumeGeometry2D) = ((vg.window_min_x + vg.window_max_x) / 2.0f0, (vg.window_min_y + vg.window_max_y) / 2.0f0)
center(vg::VolumeGeometry3D) = ((vg.window_min_x + vg.window_max_x) / 2.0f0, (vg.window_min_y + vg.window_max_y) / 2.0f0, (vg.window_min_z + vg.window_max_z) / 2.0f0)

"""
    move_vol_geom(vg::VolumeGeometry2D, pos; is_relative::Bool = false) -> VolumeGeometry2D
    move_vol_geom(vg::VolumeGeometry3D, pos; is_relative::Bool = false) -> VolumeGeometry3D

Shift the reconstruction window of a volume geometry. Matches upstream `astra.functions.move_vol_geom`.

# Arguments
- `vg`: `VolumeGeometry2D` or `VolumeGeometry3D`.
- `pos`: Coordinate tuple `(x, y)` or `(x, y, z)`.
- `is_relative`: If `false` (default), the center of the window is set to `pos`. If `true`, `pos` is added as an offset to current coordinates.

# Returns
- A new `VolumeGeometry2D` or `VolumeGeometry3D` with shifted window.

# Examples
```julia
vg = VolumeGeometry2D(256, 256)
vg_shifted = move_vol_geom(vg, (5.0, -2.0); is_relative=true)
```
"""
function move_vol_geom(vg::VolumeGeometry2D, pos::Union{Tuple{Real,Real}, AbstractVector{<:Real}}; is_relative::Bool = false)
    dx = Float32(pos[1])
    dy = Float32(pos[2])
    if !is_relative
        half_w = (vg.window_max_x - vg.window_min_x) / 2.0f0
        half_h = (vg.window_max_y - vg.window_min_y) / 2.0f0
        return VolumeGeometry2D(vg.width, vg.height, dx - half_w, dy - half_h, dx + half_w, dy + half_h)
    else
        return VolumeGeometry2D(vg.width, vg.height, vg.window_min_x + dx, vg.window_min_y + dy, vg.window_max_x + dx, vg.window_max_y + dy)
    end
end
move_vol_geom(vg::VolumeGeometry2D, dx::Real, dy::Real; kwargs...) = move_vol_geom(vg, (dx, dy); kwargs...)

function move_vol_geom(vg::VolumeGeometry3D, pos::Union{Tuple{Real,Real,Real}, AbstractVector{<:Real}}; is_relative::Bool = false)
    dx = Float32(pos[1])
    dy = Float32(pos[2])
    dz = Float32(pos[3])
    if !is_relative
        half_w = (vg.window_max_x - vg.window_min_x) / 2.0f0
        half_h = (vg.window_max_y - vg.window_min_y) / 2.0f0
        half_d = (vg.window_max_z - vg.window_min_z) / 2.0f0
        return VolumeGeometry3D(vg.width, vg.height, vg.depth, dx - half_w, dy - half_h, dz - half_d, dx + half_w, dy + half_h, dz + half_d)
    else
        return VolumeGeometry3D(vg.width, vg.height, vg.depth, vg.window_min_x + dx, vg.window_min_y + dy, vg.window_min_z + dz, vg.window_max_x + dx, vg.window_max_y + dy, vg.window_max_z + dz)
    end
end
move_vol_geom(vg::VolumeGeometry3D, dx::Real, dy::Real, dz::Real; kwargs...) = move_vol_geom(vg, (dx, dy, dz); kwargs...)

# Base.size for 2D projection geometries
Base.size(geom::Parallel2DGeometry) = (geom.detector_count, length(geom.angles))
Base.size(geom::FanFlat2DGeometry) = (geom.detector_count, length(geom.angles))
Base.size(geom::ParallelVec2DGeometry) = (geom.detector_count, size(geom.vectors, 2))
Base.size(geom::FanFlatVec2DGeometry) = (geom.detector_count, size(geom.vectors, 2))
Base.size(geom::SparseMatrix2DGeometry) = (geom.detector_count, geom.angle_count)

# Base.size for 3D projection geometries
Base.size(geom::Parallel3DGeometry) = (geom.detector_count_u, length(geom.angles), geom.detector_count_v)
Base.size(geom::Cone3DGeometry) = (geom.detector_count_u, length(geom.angles), geom.detector_count_v)
Base.size(geom::ParallelVec3DGeometry) = (geom.detector_count_u, size(geom.vectors, 2), geom.detector_count_v)
Base.size(geom::ConeVec3DGeometry) = (geom.detector_count_u, size(geom.vectors, 2), geom.detector_count_v)
Base.size(geom::CylConeVec3DGeometry) = (geom.detector_count_u, size(geom.vectors, 2), geom.detector_count_v)
Base.size(geom::AbstractGeometry, dim::Integer) = size(geom)[dim]

"""
    sinogram_size(geom::AbstractGeometry) -> Tuple{Int, ...}
    sinogram_size(geom::AbstractGeometry, dim::Integer) -> Int

Returns the dimensions of a sinogram array compatible with `geom`.

# Arguments
- `geom`: Projection geometry object (`Parallel2DGeometry`, `Cone3DGeometry`, etc.).
- `dim`: Optional integer axis index.

# Returns
- Dimensions tuple or integer size along dimension `dim`.

# Examples
```julia
geom = Parallel2DGeometry(range(0, pi, length=180); detector_count=256)
sz = sinogram_size(geom) # (256, 180)
```
"""
sinogram_size(geom::AbstractGeometry) = size(geom)
sinogram_size(geom::AbstractGeometry, dim::Integer) = size(geom, dim)

"""
    geom_size(geom::AbstractGeometry) -> Tuple{Int, ...}
    geom_size(geom::AbstractGeometry, dim::Integer) -> Int

Returns the dimensions of an array (volume or sinogram) defined by `geom`.
Matches upstream `astra.functions.geom_size` using Julia's column-major shape conventions:
- 2D volume: `(width, height)`
- 3D volume: `(width, height, depth)`
- 2D sinogram: `(detector_count, angle_count)`
- 3D sinogram: `(detector_count_u, angle_count, detector_count_v)`

# Arguments
- `geom`: Geometry object (`AbstractGeometry` or `AbstractVolumeGeometry`).
- `dim`: Optional integer axis index.

# Returns
- Dimension tuple or integer size along dimension `dim`.

# Examples
```julia
geom = Cone3DGeometry(range(0, 2pi, length=120), 128, 128, 800.0, 400.0)
sz = geom_size(geom) # (128, 120, 128)
```
"""
geom_size(geom::AbstractGeometry) = size(geom)
geom_size(geom::AbstractGeometry, dim::Integer) = size(geom, dim)

function _resolve_volume_geometry(
    volume_or_size::Union{Tuple, AbstractArray},
    volume_geometry::Union{Nothing, AbstractVolumeGeometry},
    window,
)
    if volume_geometry !== nothing
        if window !== nothing
            throw(ArgumentError("cannot specify both `volume_geometry` and `window`"))
        end
        return volume_geometry
    end
    dims = volume_or_size isa Tuple ? volume_or_size : size(volume_or_size)
    if length(dims) == 2
        return VolumeGeometry2D(dims; window = window)
    elseif length(dims) == 3
        return VolumeGeometry3D(dims; window = window)
    else
        throw(ArgumentError("unsupported volume dimensions: $(length(dims))"))
    end
end
