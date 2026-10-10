"""
    Composite Geometry and Multi-Part Projection Module (3D)

Provides multi-volume, multi-resolution, multi-projection, and out-of-core VRAM
splitting forward and backward projections based on upstream ASTRA's `CCompositeGeometryManager`.
"""

"""
    CompositeVolumePart{A<:AbstractArray{Float32,3}}

Represents a 3D volume component in a composite projection or reconstruction task,
associating a 3D data array with its spatial `VolumeGeometry3D`.
"""
struct CompositeVolumePart{A<:AbstractArray{Float32,3}}
    data::A
    geometry::VolumeGeometry3D

    function CompositeVolumePart(data::A, geometry::VolumeGeometry3D) where {A<:AbstractArray{Float32,3}}
        size(data) == (geometry.width, geometry.height, geometry.depth) || throw(
            DimensionMismatch(
                "size(data) ($(size(data))) does not match VolumeGeometry3D dimensions $((geometry.width, geometry.height, geometry.depth))",
            ),
        )
        return new{A}(data, geometry)
    end
end

CompositeVolumePart(data::AbstractArray{Float32,3}) =
    CompositeVolumePart(data, default_volume_geometry(size(data)...))

CompositeVolumePart(pair::Pair{<:AbstractArray{Float32,3},VolumeGeometry3D}) =
    CompositeVolumePart(pair.first, pair.second)

CompositeVolumePart(tuple::Tuple{<:AbstractArray{Float32,3},VolumeGeometry3D}) =
    CompositeVolumePart(tuple[1], tuple[2])

"""
    CompositeProjectionPart{A<:AbstractArray{Float32,3}, G<:AbstractGeometry3D}

Represents a 3D projection (sinogram) component in a composite projection or reconstruction task,
associating a 3D sinogram array `(detector_count_u, angle_count, detector_count_v)` with its projection geometry.
"""
struct CompositeProjectionPart{A<:AbstractArray{Float32,3},G<:AbstractGeometry3D}
    data::A
    geometry::G

    function CompositeProjectionPart(data::A, geometry::G) where {A<:AbstractArray{Float32,3},G<:AbstractGeometry3D}
        det_u = _detector_count_u(geometry)
        det_v = _detector_count_v(geometry)
        n_proj = _projection_count(geometry)
        size(data) == (det_u, n_proj, det_v) || throw(
            DimensionMismatch(
                "size(sinogram) ($(size(data))) does not match projection geometry dimensions $((det_u, n_proj, det_v))",
            ),
        )
        return new{A,G}(data, geometry)
    end
end

CompositeProjectionPart(pair::Pair{<:AbstractArray{Float32,3},<:AbstractGeometry3D}) =
    CompositeProjectionPart(pair.first, pair.second)

CompositeProjectionPart(tuple::Tuple{<:AbstractArray{Float32,3},<:AbstractGeometry3D}) =
    CompositeProjectionPart(tuple[1], tuple[2])

# Internal helpers for geometry dimension queries
_detector_count_u(g::Union{Parallel3DGeometry,ParallelVec3DGeometry,Cone3DGeometry,ConeVec3DGeometry,CylConeVec3DGeometry}) =
    g.detector_count_u
_detector_count_v(g::Union{Parallel3DGeometry,ParallelVec3DGeometry,Cone3DGeometry,ConeVec3DGeometry,CylConeVec3DGeometry}) =
    g.detector_count_v

_projection_count(g::Union{Parallel3DGeometry,Cone3DGeometry}) = length(g.angles)
_projection_count(g::Union{ParallelVec3DGeometry,ConeVec3DGeometry,CylConeVec3DGeometry}) = size(g.vectors, 2)

_to_composite_volume(v::CompositeVolumePart) = v
_to_composite_volume(p::Pair{<:AbstractArray{Float32,3},VolumeGeometry3D}) = CompositeVolumePart(p)
_to_composite_volume(t::Tuple{<:AbstractArray{Float32,3},VolumeGeometry3D}) = CompositeVolumePart(t)
_to_composite_volume(a::AbstractArray{Float32,3}) = CompositeVolumePart(a)

_to_composite_projection(p::CompositeProjectionPart) = p
_to_composite_projection(pair::Pair{<:AbstractArray{Float32,3},<:AbstractGeometry3D}) = CompositeProjectionPart(pair)
_to_composite_projection(t::Tuple{<:AbstractArray{Float32,3},<:AbstractGeometry3D}) = CompositeProjectionPart(t)

function _ensure_volume_list(v)
    if v isa CompositeVolumePart || v isa Pair || v isa Tuple || v isa AbstractArray{Float32,3}
        return CompositeVolumePart[_to_composite_volume(v)]
    elseif v isa AbstractVector
        return CompositeVolumePart[_to_composite_volume(x) for x in v]
    else
        throw(ArgumentError("unsupported volume part type: $(typeof(v))"))
    end
end

function _ensure_projection_list(p)
    if p isa CompositeProjectionPart || p isa Pair || p isa Tuple
        return CompositeProjectionPart[_to_composite_projection(p)]
    elseif p isa AbstractVector
        return CompositeProjectionPart[_to_composite_projection(x) for x in p]
    else
        throw(ArgumentError("unsupported projection part type: $(typeof(p))"))
    end
end

function _build_c_volume_parts(vol_parts::Vector{CompositeVolumePart})
    c_vols = Vector{LowLevel.VolumePart3DC}(undef, length(vol_parts))
    backing_arrays = Vector{Array{Float32,3}}(undef, length(vol_parts))
    for (i, vp) in enumerate(vol_parts)
        arr = vp.data isa Array{Float32,3} ? vp.data : Array{Float32,3}(vp.data)
        backing_arrays[i] = arr
        geom = vp.geometry
        c_vols[i] = LowLevel.VolumePart3DC(
            pointer(arr),
            Int32(geom.width),
            Int32(geom.height),
            Int32(geom.depth),
            Float32(geom.window_min_x),
            Float32(geom.window_max_x),
            Float32(geom.window_min_y),
            Float32(geom.window_max_y),
            Float32(geom.window_min_z),
            Float32(geom.window_max_z),
        )
    end
    return c_vols, backing_arrays
end

function _build_c_projection_parts(proj_parts::Vector{CompositeProjectionPart})
    c_projs = Vector{LowLevel.ProjectionPart3DC}(undef, length(proj_parts))
    backing_arrays = Vector{Array{Float32,3}}(undef, length(proj_parts))
    backing_angles = Vector{Any}(undef, length(proj_parts))
    backing_vectors = Vector{Any}(undef, length(proj_parts))

    for (i, pp) in enumerate(proj_parts)
        arr = pp.data isa Array{Float32,3} ? pp.data : Array{Float32,3}(pp.data)
        backing_arrays[i] = arr
        geom = pp.geometry

        geom_type = Int32(0)
        proj_count = Int32(_projection_count(geom))
        det_row_count = Int32(_detector_count_v(geom))
        det_col_count = Int32(_detector_count_u(geom))
        det_width = 1.0f0
        det_height = 1.0f0
        src_origin = 0.0f0
        origin_det = 0.0f0
        cyl_radius = 0.0f0
        angles_ptr = Ptr{Float32}(C_NULL)
        vectors_ptr = Ptr{Float32}(C_NULL)

        if geom isa Parallel3DGeometry
            geom_type = Int32(0)
            det_width = Float32(geom.detector_spacing_u)
            det_height = Float32(geom.detector_spacing_v)
            ang = geom.angles isa Vector{Float32} ? geom.angles : Vector{Float32}(geom.angles)
            backing_angles[i] = ang
            angles_ptr = pointer(ang)
        elseif geom isa ParallelVec3DGeometry
            geom_type = Int32(1)
            vecs = geom.vectors isa Matrix{Float32} ? geom.vectors : Matrix{Float32}(geom.vectors)
            backing_vectors[i] = vecs
            vectors_ptr = pointer(vecs)
        elseif geom isa Cone3DGeometry
            geom_type = Int32(2)
            det_width = Float32(geom.detector_spacing_u)
            det_height = Float32(geom.detector_spacing_v)
            src_origin = Float32(geom.origin_source_distance)
            origin_det = Float32(geom.origin_detector_distance)
            ang = geom.angles isa Vector{Float32} ? geom.angles : Vector{Float32}(geom.angles)
            backing_angles[i] = ang
            angles_ptr = pointer(ang)
        elseif geom isa ConeVec3DGeometry
            geom_type = Int32(3)
            vecs = geom.vectors isa Matrix{Float32} ? geom.vectors : Matrix{Float32}(geom.vectors)
            backing_vectors[i] = vecs
            vectors_ptr = pointer(vecs)
        elseif geom isa CylConeVec3DGeometry
            geom_type = Int32(4)
            cyl_radius = Float32(geom.cylinder_radius)
            vecs = geom.vectors isa Matrix{Float32} ? geom.vectors : Matrix{Float32}(geom.vectors)
            backing_vectors[i] = vecs
            vectors_ptr = pointer(vecs)
        else
            throw(ArgumentError("unsupported 3D projection geometry type for composite projection: $(typeof(geom))"))
        end

        c_projs[i] = LowLevel.ProjectionPart3DC(
            pointer(arr),
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
    end
    return c_projs, backing_arrays, backing_angles, backing_vectors
end

# -------------------------------------------------------------------------
# Composite Forward Projection (FP)
# -------------------------------------------------------------------------

"""
    do_composite_FP!(projections, volumes; mode=:set, backend=cuda()) -> projections

Compute composite 3D forward projection of one or more volume parts into one or more
projection parts using upstream ASTRA's `CCompositeGeometryManager`.

Each volume part and projection part can possess distinct resolutions, bounds, and geometric specifications.
When GPU memory is limited, volumes are automatically split into slices on GPU to avoid OOM.

# Arguments
- `projections`: Destination projection part or collection of projection parts. Each item can be:
  - `CompositeProjectionPart(data, geometry)`
  - A Pair `sinogram => geometry`
  - A Tuple `(sinogram, geometry)`
- `volumes`: Source volume part or collection of volume parts. Each item can be:
  - `CompositeVolumePart(data, geometry)`
  - A Pair `volume => geometry`
  - A Tuple `(volume, geometry)`
  - An `AbstractArray{Float32, 3}` (uses default centered `VolumeGeometry3D`)
- `mode`: `:set` (overwrite projections with projected values) or `:add` (accumulate projected values into existing projections).
- `backend`: `cuda()` or `cuda(gpu_index=...)`.

# Returns
- `projections`: The destination projections (mutated in-place).

# Examples
```julia
geom = Cone3DGeometry(range(0, 2pi, length=60), 128, 128, 800.0, 400.0)
vol = shepp_logan_3d(128)
sino = zeros(Float32, 128, 60, 128)
do_composite_FP!(sino => geom, vol; backend=cuda())
```
"""
function do_composite_FP!(
    projections,
    volumes;
    mode::Symbol = :set,
    backend::CUDABackend = cuda(),
)
    mode in (:set, :add) || throw(ArgumentError("mode must be :set or :add, got :$mode"))
    mode_int = mode === :set ? Int32(1) : Int32(0)

    vol_list = _ensure_volume_list(volumes)
    proj_list = _ensure_projection_list(projections)

    c_vols, backing_vols = _build_c_volume_parts(vol_list)
    c_projs, backing_projs, backing_angles, backing_vectors = _build_c_projection_parts(proj_list)

    GC.@preserve backing_vols backing_projs backing_angles backing_vectors begin
        LowLevel.cuda_composite_fp3d_f32!(
            c_vols,
            c_projs;
            mode = mode_int,
            gpu_index = backend.gpu_index,
        )
    end

    # Sync back in case of non-Array AbstractArrays
    for (i, pp) in enumerate(proj_list)
        if pp.data !== backing_projs[i]
            copyto!(pp.data, backing_projs[i])
        end
    end

    return projections
end

"""
    do_composite_FP(volumes, projections; mode=:set, backend=cuda())

Out-of-place version of `do_composite_FP!`.
If `projections` contains geometries without data (e.g. `AbstractGeometry3D` or `Vector{<:AbstractGeometry3D}`),
destination arrays will be allocated automatically.

# Arguments
- `volumes`: Volume part or collection of volume parts.
- `projections`: Projection geometry or collection of projection geometries / parts.
- `mode`: `:set` or `:add`. Defaults to `:set`.
- `backend`: `cuda()`.

# Returns
- A newly allocated 3D projection array or vector of arrays.

# Examples
```julia
geom = Cone3DGeometry(range(0, 2pi, length=60), 128, 128, 800.0, 400.0)
vol = shepp_logan_3d(128)
sino = do_composite_FP(vol, geom; backend=cuda())
```
"""
function do_composite_FP(
    volumes,
    projections;
    mode::Symbol = :set,
    backend::CUDABackend = cuda(),
)
    # Check if projections are bare geometries
    if projections isa AbstractGeometry3D
        geom = projections
        u = _detector_count_u(geom)
        v = _detector_count_v(geom)
        n = _projection_count(geom)
        data = zeros(Float32, u, n, v)
        part = CompositeProjectionPart(data, geom)
        do_composite_FP!(part, volumes; mode = mode, backend = backend)
        return data
    elseif projections isa AbstractVector && !isempty(projections) && first(projections) isa AbstractGeometry3D
        parts = CompositeProjectionPart[]
        datas = Array{Float32,3}[]
        for geom in projections
            u = _detector_count_u(geom)
            v = _detector_count_v(geom)
            n = _projection_count(geom)
            data = zeros(Float32, u, n, v)
            push!(datas, data)
            push!(parts, CompositeProjectionPart(data, geom))
        end
        do_composite_FP!(parts, volumes; mode = mode, backend = backend)
        return datas
    else
        # Projections already include arrays
        do_composite_FP!(projections, volumes; mode = mode, backend = backend)
        return projections
    end
end

# -------------------------------------------------------------------------
# Composite Back Projection (BP)
# -------------------------------------------------------------------------

"""
    do_composite_BP!(volumes, projections; mode=:set, backend=cuda()) -> volumes

Compute composite 3D backprojection from one or more projection parts into one or more
volume parts using upstream ASTRA's `CCompositeGeometryManager`.

# Arguments
- `volumes`: Destination volume part or collection of volume parts. Each item can be:
  - `CompositeVolumePart(data, geometry)`
  - A Pair `volume => geometry`
  - A Tuple `(volume, geometry)`
  - An `AbstractArray{Float32, 3}` (uses default centered `VolumeGeometry3D`)
- `projections`: Source projection part or collection of projection parts. Each item can be:
  - `CompositeProjectionPart(data, geometry)`
  - A Pair `sinogram => geometry`
  - A Tuple `(sinogram, geometry)`
- `mode`: `:set` (overwrite volume data) or `:add` (accumulate into volume data).
- `backend`: `cuda()` or `cuda(gpu_index=...)`.

# Returns
- `volumes`: The destination volume(s) (mutated in-place).

# Examples
```julia
geom = Cone3DGeometry(range(0, 2pi, length=60), 128, 128, 800.0, 400.0)
vol = zeros(Float32, 128, 128, 128)
do_composite_BP!(vol => VolumeGeometry3D(128, 128, 128), sino => geom; backend=cuda())
```
"""
function do_composite_BP!(
    volumes,
    projections;
    mode::Symbol = :set,
    backend::CUDABackend = cuda(),
)
    mode in (:set, :add) || throw(ArgumentError("mode must be :set or :add, got :$mode"))
    mode_int = mode === :set ? Int32(1) : Int32(0)

    vol_list = _ensure_volume_list(volumes)
    proj_list = _ensure_projection_list(projections)

    c_vols, backing_vols = _build_c_volume_parts(vol_list)
    c_projs, backing_projs, backing_angles, backing_vectors = _build_c_projection_parts(proj_list)

    GC.@preserve backing_vols backing_projs backing_angles backing_vectors begin
        LowLevel.cuda_composite_bp3d_f32!(
            c_vols,
            c_projs;
            mode = mode_int,
            gpu_index = backend.gpu_index,
        )
    end

    # Sync back in case of non-Array AbstractArrays
    for (i, vp) in enumerate(vol_list)
        if vp.data !== backing_vols[i]
            copyto!(vp.data, backing_vols[i])
        end
    end

    return volumes
end

"""
    do_composite_BP(volumes, projections; mode=:set, backend=cuda())

Out-of-place version of `do_composite_BP!`.
If `volumes` contains geometries without data (e.g. `VolumeGeometry3D` or `Vector{VolumeGeometry3D}`),
destination volume arrays will be allocated automatically.

# Arguments
- `volumes`: Target `VolumeGeometry3D` or collection of geometries / parts.
- `projections`: Projection data or collection of projection parts.
- `mode`: `:set` or `:add`. Defaults to `:set`.
- `backend`: `cuda()`.

# Returns
- A newly allocated 3D volume array or vector of arrays.

# Examples
```julia
geom = Cone3DGeometry(range(0, 2pi, length=60), 128, 128, 800.0, 400.0)
vg = VolumeGeometry3D(128, 128, 128)
vol = do_composite_BP(vg, sino => geom; backend=cuda())
```
"""
function do_composite_BP(
    volumes,
    projections;
    mode::Symbol = :set,
    backend::CUDABackend = cuda(),
)
    if volumes isa VolumeGeometry3D
        geom = volumes
        data = zeros(Float32, geom.width, geom.height, geom.depth)
        part = CompositeVolumePart(data, geom)
        do_composite_BP!(part, projections; mode = mode, backend = backend)
        return data
    elseif volumes isa AbstractVector && !isempty(volumes) && first(volumes) isa VolumeGeometry3D
        parts = CompositeVolumePart[]
        datas = Array{Float32,3}[]
        for geom in volumes
            data = zeros(Float32, geom.width, geom.height, geom.depth)
            push!(datas, data)
            push!(parts, CompositeVolumePart(data, geom))
        end
        do_composite_BP!(parts, projections; mode = mode, backend = backend)
        return datas
    else
        do_composite_BP!(volumes, projections; mode = mode, backend = backend)
        return volumes
    end
end

# -------------------------------------------------------------------------
# Convenience Accumulation & Aliases
# -------------------------------------------------------------------------

"""
    accumulate_FP!(projections, volumes; backend=cuda())

Convenience function to accumulate composite forward projection into `projections` (`mode=:add`).

# Arguments
- `projections`: Destination projection parts (mutated in-place).
- `volumes`: Source volume parts.
- `backend`: `cuda()`.

# Returns
- `projections`: Updated projection buffer.
"""
accumulate_FP!(projections, volumes; backend::CUDABackend = cuda()) =
    do_composite_FP!(projections, volumes; mode = :add, backend = backend)

"""
    accumulate_BP!(volumes, projections; backend=cuda())

Convenience function to accumulate composite backprojection into `volumes` (`mode=:add`).

# Arguments
- `volumes`: Destination volume parts (mutated in-place).
- `projections`: Source projection parts.
- `backend`: `cuda()`.

# Returns
- `volumes`: Updated volume buffer.
"""
accumulate_BP!(volumes, projections; backend::CUDABackend = cuda()) =
    do_composite_BP!(volumes, projections; mode = :add, backend = backend)

const composite_project! = do_composite_FP!
const composite_project = do_composite_FP
const composite_backproject! = do_composite_BP!
const composite_backproject = do_composite_BP
