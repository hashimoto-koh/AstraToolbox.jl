# Geometry conversion and calibration utilities

"""
    geom_2vec(geom::AbstractGeometry) -> AbstractGeometry

Convert a standard projection geometry to its corresponding vector-based projection geometry.
If `geom` is already a vector geometry, a copy is returned.

# Arguments
- `geom`: Input projection geometry (`Parallel2DGeometry`, `FanFlat2DGeometry`, `Parallel3DGeometry`, `Cone3DGeometry`, or a vector geometry).

# Returns
- Vector-based projection geometry (`ParallelVec2DGeometry`, `FanFlatVec2DGeometry`, `ParallelVec3DGeometry`, `ConeVec3DGeometry`, or `CylConeVec3DGeometry`).

# Examples
```julia
geom = Parallel2DGeometry(range(0, pi, length=180); detector_count=256)
geom_vec = geom_2vec(geom)
```
"""
function geom_2vec(geom::Parallel2DGeometry)
    n = length(geom.angles)
    vectors = Matrix{Float32}(undef, 6, n)
    w = geom.detector_width
    @inbounds for i in 1:n
        θ = geom.angles[i]
        sθ, cθ = sincos(θ)
        # Ray direction
        vectors[1, i] = sθ
        vectors[2, i] = -cθ
        # Detector center
        vectors[3, i] = 0.0f0
        vectors[4, i] = 0.0f0
        # Vector from detector pixel 0 to 1
        vectors[5, i] = cθ * w
        vectors[6, i] = sθ * w
    end
    return ParallelVec2DGeometry(vectors, geom.detector_count)
end

function geom_2vec(geom::FanFlat2DGeometry)
    n = length(geom.angles)
    vectors = Matrix{Float32}(undef, 6, n)
    w = geom.detector_width
    dso = geom.origin_source_distance
    dod = geom.origin_detector_distance
    @inbounds for i in 1:n
        θ = geom.angles[i]
        sθ, cθ = sincos(θ)
        # Source position
        vectors[1, i] = sθ * dso
        vectors[2, i] = -cθ * dso
        # Detector center
        vectors[3, i] = -sθ * dod
        vectors[4, i] = cθ * dod
        # Vector from detector pixel 0 to 1
        vectors[5, i] = cθ * w
        vectors[6, i] = sθ * w
    end
    return FanFlatVec2DGeometry(vectors, geom.detector_count)
end

function geom_2vec(geom::Parallel3DGeometry)
    n = length(geom.angles)
    vectors = Matrix{Float32}(undef, 12, n)
    su = geom.detector_spacing_u
    sv = geom.detector_spacing_v
    @inbounds for i in 1:n
        θ = geom.angles[i]
        sθ, cθ = sincos(θ)
        # Ray direction
        vectors[1, i] = sθ
        vectors[2, i] = -cθ
        vectors[3, i] = 0.0f0
        # Detector center
        vectors[4, i] = 0.0f0
        vectors[5, i] = 0.0f0
        vectors[6, i] = 0.0f0
        # Vector from detector pixel (0,0) to (0,1) [horizontal u]
        vectors[7, i] = cθ * su
        vectors[8, i] = sθ * su
        vectors[9, i] = 0.0f0
        # Vector from detector pixel (0,0) to (1,0) [vertical v]
        vectors[10, i] = 0.0f0
        vectors[11, i] = 0.0f0
        vectors[12, i] = sv
    end
    return ParallelVec3DGeometry(vectors, geom.detector_count_u, geom.detector_count_v)
end

function geom_2vec(geom::Cone3DGeometry)
    n = length(geom.angles)
    vectors = Matrix{Float32}(undef, 12, n)
    dso = geom.origin_source_distance
    dod = geom.origin_detector_distance
    su = geom.detector_spacing_u
    sv = geom.detector_spacing_v
    @inbounds for i in 1:n
        θ = geom.angles[i]
        sθ, cθ = sincos(θ)
        # Source position
        vectors[1, i] = sθ * dso
        vectors[2, i] = -cθ * dso
        vectors[3, i] = 0.0f0
        # Detector center
        vectors[4, i] = -sθ * dod
        vectors[5, i] = cθ * dod
        vectors[6, i] = 0.0f0
        # Vector from detector pixel (0,0) to (0,1) [horizontal u]
        vectors[7, i] = cθ * su
        vectors[8, i] = sθ * su
        vectors[9, i] = 0.0f0
        # Vector from detector pixel (0,0) to (1,0) [vertical v]
        vectors[10, i] = 0.0f0
        vectors[11, i] = 0.0f0
        vectors[12, i] = sv
    end
    return ConeVec3DGeometry(vectors, geom.detector_count_u, geom.detector_count_v)
end

geom_2vec(geom::ParallelVec2DGeometry) = ParallelVec2DGeometry(copy(geom.vectors), geom.detector_count)
geom_2vec(geom::FanFlatVec2DGeometry) = FanFlatVec2DGeometry(copy(geom.vectors), geom.detector_count)
geom_2vec(geom::ParallelVec3DGeometry) = ParallelVec3DGeometry(copy(geom.vectors), geom.detector_count_u, geom.detector_count_v)
geom_2vec(geom::ConeVec3DGeometry) = ConeVec3DGeometry(copy(geom.vectors), geom.detector_count_u, geom.detector_count_v)
geom_2vec(geom::CylConeVec3DGeometry) = CylConeVec3DGeometry(copy(geom.vectors), geom.detector_count_u, geom.detector_count_v)

function geom_2vec(geom::AbstractGeometry)
    throw(ArgumentError("No suitable vector geometry found for type: $(typeof(geom))"))
end

"""
    geom_postalignment(proj_geom::AbstractGeometry, factor) -> AbstractGeometry

Apply postalignment to a projection geometry. Can be used to model detector offset or rotation axis offset.

# Arguments
- `proj_geom`: Input projection geometry (basic or vector-based). If a basic geometry is passed, it is first converted to vector format.
- `factor`:
  - For 2D geometries: A single number specifying detector pixel shift.
  - For 3D geometries: A single number (horizontal shift) or a 2-element tuple/vector `(factor_u, factor_v)` specifying horizontal and vertical pixel shifts.

# Returns
- A modified vector-based projection geometry (`ParallelVec2DGeometry`, `FanFlatVec2DGeometry`, `ParallelVec3DGeometry`, or `ConeVec3DGeometry`).

# Examples
```julia
geom = Parallel2DGeometry(range(0, pi, length=180); detector_count=256)
geom_shifted = geom_postalignment(geom, 2.5) # Shift detector by +2.5 pixels
```
"""
function geom_postalignment(geom::Union{Parallel2DGeometry, FanFlat2DGeometry}, factor::Real)
    return geom_postalignment(geom_2vec(geom), factor)
end

function geom_postalignment(geom::ParallelVec2DGeometry, factor::Real)
    V = copy(geom.vectors)
    f = Float32(factor)
    @views V[3:4, :] .+= f .* V[5:6, :]
    return ParallelVec2DGeometry(V, geom.detector_count)
end

function geom_postalignment(geom::FanFlatVec2DGeometry, factor::Real)
    V = copy(geom.vectors)
    f = Float32(factor)
    @views V[3:4, :] .+= f .* V[5:6, :]
    return FanFlatVec2DGeometry(V, geom.detector_count)
end

function geom_postalignment(geom::Union{Parallel3DGeometry, Cone3DGeometry}, factor)
    return geom_postalignment(geom_2vec(geom), factor)
end

function geom_postalignment(geom::ParallelVec3DGeometry, factor::Union{Real, Tuple{Real, Real}, AbstractVector{<:Real}})
    fu = Float32(factor isa Real ? factor : factor[1])
    fv = Float32(factor isa Real ? 0.0f0 : (length(factor) > 1 ? factor[2] : 0.0f0))
    V = copy(geom.vectors)
    @views V[4:6, :] .+= fu .* V[7:9, :] .+ fv .* V[10:12, :]
    return ParallelVec3DGeometry(V, geom.detector_count_u, geom.detector_count_v)
end

function geom_postalignment(geom::ConeVec3DGeometry, factor::Union{Real, Tuple{Real, Real}, AbstractVector{<:Real}})
    fu = Float32(factor isa Real ? factor : factor[1])
    fv = Float32(factor isa Real ? 0.0f0 : (length(factor) > 1 ? factor[2] : 0.0f0))
    V = copy(geom.vectors)
    @views V[4:6, :] .+= fu .* V[7:9, :] .+ fv .* V[10:12, :]
    return ConeVec3DGeometry(V, geom.detector_count_u, geom.detector_count_v)
end

function geom_postalignment(geom::AbstractGeometry, factor)
    throw(ArgumentError("$(typeof(geom)) is not suitable for postalignment"))
end

"""
    clip_circle!(image::AbstractMatrix; radius=nothing, center=nothing) -> image
    clip_circle!(volume::AbstractArray{<:Real, 3}; radius=nothing, center=nothing) -> volume
    clip_circle(data; radius=nothing, center=nothing) -> new_data

Zero out pixels (in 2D) or voxels (in 3D cylindrical FOV) that lie outside a circular boundary.
Matches upstream `astra.clipCircle`.

# Arguments
- `data`: 2D Matrix or 3D Array of real numbers.
- `radius`: Radius of the circular mask. Defaults to `min(nx, ny) / 2`.
- `center`: Coordinates `(cx, cy)` of the center in 1-based indexing. Defaults to `((nx + 1) / 2, (ny + 1) / 2)`.

# Returns
- The modified array (in-place for `clip_circle!`, new copy for `clip_circle`).

# Examples
```julia
img = ones(Float32, 256, 256)
clip_circle!(img)
```
"""
function clip_circle!(image::AbstractMatrix{T}; radius = nothing, center = nothing) where {T<:Real}
    nx, ny = size(image)
    r = radius === nothing ? (min(nx, ny) / 2.0f0) : Float32(radius)
    r2 = r * r
    cx, cy = center === nothing ? ((nx + 1.0f0) / 2.0f0, (ny + 1.0f0) / 2.0f0) : (Float32(center[1]), Float32(center[2]))

    @inbounds for j in 1:ny
        dy = Float32(j) - cy
        dy2 = dy * dy
        for i in 1:nx
            dx = Float32(i) - cx
            if dx * dx + dy2 > r2
                image[i, j] = zero(T)
            end
        end
    end
    return image
end

function clip_circle!(volume::AbstractArray{T, 3}; radius = nothing, center = nothing) where {T<:Real}
    nx, ny, nz = size(volume)
    r = radius === nothing ? (min(nx, ny) / 2.0f0) : Float32(radius)
    r2 = r * r
    cx, cy = center === nothing ? ((nx + 1.0f0) / 2.0f0, (ny + 1.0f0) / 2.0f0) : (Float32(center[1]), Float32(center[2]))

    @inbounds for k in 1:nz
        for j in 1:ny
            dy = Float32(j) - cy
            dy2 = dy * dy
            for i in 1:nx
                dx = Float32(i) - cx
                if dx * dx + dy2 > r2
                    volume[i, j, k] = zero(T)
                end
            end
        end
    end
    return volume
end

"""
    clip_circle(data::AbstractArray{<:Real}; radius=nothing, center=nothing) -> AbstractArray

Non-destructive version of [`clip_circle!`](@ref). Returns a copy of `data` with values outside the circular or cylindrical mask set to zero.

# Arguments
- `data`: 2D Matrix or 3D Array of real numbers.
- `radius`: Radius of the circular mask. Defaults to `min(nx, ny) / 2`.
- `center`: Coordinates `(cx, cy)` of the center in 1-based indexing. Defaults to `((nx + 1) / 2, (ny + 1) / 2)`.

# Returns
- A newly allocated array of the same type and size as `data` with masked values.

# Examples
```julia
img = ones(Float32, 256, 256)
masked = clip_circle(img)
```
"""
function clip_circle(data::AbstractArray{<:Real}; radius = nothing, center = nothing)
    out = copy(data)
    clip_circle!(out; radius = radius, center = center)
    return out
end

function _make_c_projection_part(geom::AbstractGeometry3D)
    proj_count = Int32(0)
    det_row_count = Int32(0)
    det_col_count = Int32(0)
    det_width = 1.0f0
    det_height = 1.0f0
    src_origin = 0.0f0
    origin_det = 0.0f0
    cyl_radius = 0.0f0
    angles_ptr = Ptr{Float32}(C_NULL)
    vectors_ptr = Ptr{Float32}(C_NULL)
    backing = Any[]

    if geom isa Parallel3DGeometry
        geom_type = Int32(0)
        proj_count = Int32(length(geom.angles))
        det_row_count = Int32(geom.detector_count_v)
        det_col_count = Int32(geom.detector_count_u)
        det_width = Float32(geom.detector_spacing_u)
        det_height = Float32(geom.detector_spacing_v)
        ang = geom.angles isa Vector{Float32} ? geom.angles : Vector{Float32}(geom.angles)
        push!(backing, ang)
        angles_ptr = pointer(ang)
    elseif geom isa ParallelVec3DGeometry
        geom_type = Int32(1)
        proj_count = Int32(size(geom.vectors, 2))
        det_row_count = Int32(geom.detector_count_v)
        det_col_count = Int32(geom.detector_count_u)
        vecs = geom.vectors isa Matrix{Float32} ? geom.vectors : Matrix{Float32}(geom.vectors)
        push!(backing, vecs)
        vectors_ptr = pointer(vecs)
    elseif geom isa Cone3DGeometry
        geom_type = Int32(2)
        proj_count = Int32(length(geom.angles))
        det_row_count = Int32(geom.detector_count_v)
        det_col_count = Int32(geom.detector_count_u)
        det_width = Float32(geom.detector_spacing_u)
        det_height = Float32(geom.detector_spacing_v)
        src_origin = Float32(geom.origin_source_distance)
        origin_det = Float32(geom.origin_detector_distance)
        ang = geom.angles isa Vector{Float32} ? geom.angles : Vector{Float32}(geom.angles)
        push!(backing, ang)
        angles_ptr = pointer(ang)
    elseif geom isa ConeVec3DGeometry
        geom_type = Int32(3)
        proj_count = Int32(size(geom.vectors, 2))
        det_row_count = Int32(geom.detector_count_v)
        det_col_count = Int32(geom.detector_count_u)
        vecs = geom.vectors isa Matrix{Float32} ? geom.vectors : Matrix{Float32}(geom.vectors)
        push!(backing, vecs)
        vectors_ptr = pointer(vecs)
    elseif geom isa CylConeVec3DGeometry
        geom_type = Int32(4)
        proj_count = Int32(size(geom.vectors, 2))
        det_row_count = Int32(geom.detector_count_v)
        det_col_count = Int32(geom.detector_count_u)
        vecs = geom.vectors isa Matrix{Float32} ? geom.vectors : Matrix{Float32}(geom.vectors)
        push!(backing, vecs)
        vectors_ptr = pointer(vecs)
    else
        throw(ArgumentError("unsupported 3D projection geometry type: $(typeof(geom))"))
    end

    c_part = LowLevel.ProjectionPart3DC(
        Ptr{Float32}(C_NULL),
        geom_type,
        proj_count,
        det_row_count,
        det_col_count,
        det_width,
        det_height,
        src_origin,
        origin_det,
        cyl_radius,
        angles_ptr,
        vectors_ptr,
    )
    return c_part, backing
end

"""
    project_point(geom::AbstractGeometry3D, x::Real, y::Real, z::Real, angle_index::Integer; zero_based::Bool = false) -> Tuple{Float64, Float64}

Calculate the projected detector coordinates `(u, v)` (columns/horizontal, rows/vertical) of a 3D point `(x, y, z)`
for the projection at `angle_index`.

# Arguments
- `geom`: 3D projection geometry object (`Cone3DGeometry`, `ConeVec3DGeometry`, etc.).
- `x, y, z`: 3D world coordinates of the point.
- `angle_index`: Index of the projection angle.
- `zero_based`: If `false` (default Julia convention), `angle_index` and returned `(u, v)` are 1-based. If `true`, they are 0-based.

# Returns
- `Tuple{Float64, Float64}`: Projected detector coordinates `(u, v)`.

# Examples
```julia
geom = Cone3DGeometry(range(0, 2pi, length=120), 128, 128, 800.0, 400.0)
u, v = project_point(geom, 0.0, 0.0, 0.0, 1)
```
"""
function project_point(geom::AbstractGeometry3D, x::Real, y::Real, z::Real, angle_index::Integer; zero_based::Bool = false)
    c_part, backing = _make_c_projection_part(geom)
    c_angle = zero_based ? Int32(angle_index) : Int32(angle_index - 1)
    local u::Float64, v::Float64
    GC.@preserve backing begin
        u, v = LowLevel.project_point_3d(c_part, x, y, z, c_angle)
    end
    if !zero_based
        return (u + 1.0, v + 1.0)
    else
        return (u, v)
    end
end

"""
    projected_bbox(geom::AbstractGeometry3D, min_x::Real, max_x::Real, min_y::Real, max_y::Real, min_z::Real, max_z::Real; angle = nothing, zero_based::Bool = false)
    projected_bbox(geom::AbstractGeometry3D, vg::VolumeGeometry3D; angle = nothing, zero_based::Bool = false)
    projected_bbox(geom::AbstractGeometry3D, bbox::Tuple; angle = nothing, zero_based::Bool = false)

Compute the bounding box of detector coordinates `(min_u, max_u, min_v, max_v)` containing the projection
of the specified 3D volume bounding box.

# Arguments
- `geom`: 3D projection geometry.
- `min_x, max_x, min_y, max_y, min_z, max_z`: Spatial boundaries of the 3D volume.
- `angle`: If specified, computes the projected bounding box for that single angle index. If `nothing` (default), computes the envelope across all projection angles.
- `zero_based`: If `false` (default Julia convention), angle indices and returned detector coordinates are 1-based. If `true`, they are 0-based.

# Returns
- `NamedTuple`: `(min_u = ..., max_u = ..., min_v = ..., max_v = ...)`.

# Examples
```julia
geom = Cone3DGeometry(range(0, 2pi, length=120), 128, 128, 800.0, 400.0)
bbox = projected_bbox(geom, -50.0, 50.0, -50.0, 50.0, -25.0, 25.0)
```
"""
function projected_bbox(
    geom::AbstractGeometry3D,
    min_x::Real, max_x::Real,
    min_y::Real, max_y::Real,
    min_z::Real, max_z::Real;
    angle = nothing,
    zero_based::Bool = false,
)
    c_part, backing = _make_c_projection_part(geom)
    c_angle = angle === nothing ? Int32(-1) : (zero_based ? Int32(angle) : Int32(angle - 1))
    local res
    GC.@preserve backing begin
        res = LowLevel.projected_bbox_3d(c_part, min_x, max_x, min_y, max_y, min_z, max_z, c_angle)
    end
    if !zero_based
        return (min_u = res.min_u + 1.0, max_u = res.max_u + 1.0, min_v = res.min_v + 1.0, max_v = res.max_v + 1.0)
    else
        return res
    end
end

projected_bbox(geom::AbstractGeometry3D, vg::VolumeGeometry3D; kwargs...) =
    projected_bbox(geom, vg.window_min_x, vg.window_max_x, vg.window_min_y, vg.window_max_y, vg.window_min_z, vg.window_max_z; kwargs...)

projected_bbox(geom::AbstractGeometry3D, bbox::Tuple{Real,Real,Real,Real,Real,Real}; kwargs...) =
    projected_bbox(geom, bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], bbox[6]; kwargs...)

