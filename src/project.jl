function _merge_projection_options(
    ::Type{ProjectionOptions2D},
    options::Union{Nothing,ProjectionOptions2D},
    detector_supersampling,
    projector_kernel = nothing,
)
    base = options === nothing ? ProjectionOptions2D() : options
    return ProjectionOptions2D(
        detector_supersampling = detector_supersampling === nothing ? base.detector_supersampling : detector_supersampling,
        projector_kernel = projector_kernel === nothing ? base.projector_kernel : projector_kernel,
    )
end

function _merge_projection_options(
    ::Type{ProjectionOptions3D},
    options::Union{Nothing,ProjectionOptions3D},
    detector_supersampling,
)
    base = options === nothing ? ProjectionOptions3D() : options
    return ProjectionOptions3D(
        detector_supersampling = detector_supersampling === nothing ? base.detector_supersampling : detector_supersampling,
    )
end

function _merge_backprojection_options(
    ::Type{BackprojectionOptions2D},
    options::Union{Nothing,BackprojectionOptions2D},
    detector_supersampling,
    pixel_supersampling,
    projector_kernel = nothing,
)
    base = options === nothing ? BackprojectionOptions2D() : options
    return BackprojectionOptions2D(
        detector_supersampling = detector_supersampling === nothing ? base.detector_supersampling : detector_supersampling,
        pixel_supersampling = pixel_supersampling === nothing ? base.pixel_supersampling : pixel_supersampling,
        projector_kernel = projector_kernel === nothing ? base.projector_kernel : projector_kernel,
    )
end

function _merge_backprojection_options(
    ::Type{BackprojectionOptions3D},
    options::Union{Nothing,BackprojectionOptions3D},
    voxel_supersampling,
)
    base = options === nothing ? BackprojectionOptions3D() : options
    return BackprojectionOptions3D(
        voxel_supersampling = voxel_supersampling === nothing ? base.voxel_supersampling : voxel_supersampling,
    )
end

function _project_cpu!(sinogram::Array{Float32,2}, volume::Array{Float32,2}, geometry::Parallel2DGeometry, options::ProjectionOptions2D = ProjectionOptions2D())
    kernel = options.projector_kernel
    if kernel == :line
        LowLevel.fp2d_parallel_line_f32!(
            sinogram,
            volume,
            geometry.angles;
            detector_width = geometry.detector_width,
        )
    elseif kernel == :linear
        LowLevel.fp2d_parallel_linear_f32!(
            sinogram,
            volume,
            geometry.angles;
            detector_width = geometry.detector_width,
        )
    elseif kernel == :strip
        LowLevel.fp2d_parallel_strip_f32!(
            sinogram,
            volume,
            geometry.angles;
            detector_width = geometry.detector_width,
        )
    elseif kernel == :distance_driven
        LowLevel.fp2d_parallel_distance_driven_f32!(
            sinogram,
            volume,
            geometry.angles;
            detector_width = geometry.detector_width,
        )
    else
        throw(ArgumentError("unsupported projector_kernel $kernel"))
    end
end

function _project_cpu!(sinogram::Array{Float32,2}, volume::Array{Float32,2}, geometry::ParallelVec2DGeometry, options::ProjectionOptions2D = ProjectionOptions2D())
    kernel = options.projector_kernel
    if kernel == :line
        LowLevel.fp2d_parallel_vec_line_f32!(sinogram, volume, geometry.vectors)
    elseif kernel == :linear
        LowLevel.fp2d_parallel_vec_linear_f32!(sinogram, volume, geometry.vectors)
    elseif kernel == :strip
        LowLevel.fp2d_parallel_vec_strip_f32!(sinogram, volume, geometry.vectors)
    elseif kernel == :distance_driven
        LowLevel.fp2d_parallel_vec_distance_driven_f32!(sinogram, volume, geometry.vectors)
    else
        throw(ArgumentError("unsupported projector_kernel $kernel"))
    end
end

function _project_cpu!(sinogram::Array{Float32,2}, volume::Array{Float32,2}, geometry::FanFlat2DGeometry, options::ProjectionOptions2D = ProjectionOptions2D())
    kernel = options.projector_kernel
    if kernel == :line
        LowLevel.fp2d_fanflat_line_f32!(
            sinogram,
            volume,
            geometry.angles;
            detector_width = geometry.detector_width,
            origin_source_distance = geometry.origin_source_distance,
            origin_detector_distance = geometry.origin_detector_distance,
        )
    elseif kernel == :strip
        LowLevel.fp2d_fanflat_strip_f32!(
            sinogram,
            volume,
            geometry.angles;
            detector_width = geometry.detector_width,
            origin_source_distance = geometry.origin_source_distance,
            origin_detector_distance = geometry.origin_detector_distance,
        )
    elseif kernel in (:linear, :distance_driven)
        throw(ArgumentError("projector_kernel :$kernel is only supported for parallel beam geometry; use :line or :strip for fan-flat"))
    else
        throw(ArgumentError("unsupported projector_kernel $kernel"))
    end
end

function _project_cpu!(sinogram::Array{Float32,2}, volume::Array{Float32,2}, geometry::FanFlatVec2DGeometry, options::ProjectionOptions2D = ProjectionOptions2D())
    kernel = options.projector_kernel
    if kernel == :line
        LowLevel.fp2d_fanflat_vec_line_f32!(sinogram, volume, geometry.vectors)
    elseif kernel in (:linear, :strip, :distance_driven)
        throw(ArgumentError("projector_kernel :$kernel is not supported for fan-flat vector geometry; use :line"))
    else
        throw(ArgumentError("unsupported projector_kernel $kernel"))
    end
end

function _project_cuda!(sinogram::Array{Float32,2}, volume::Array{Float32,2}, geometry::Parallel2DGeometry, backend::CUDABackend, options::ProjectionOptions2D)
    LowLevel.cuda_fp2d_parallel_line_f32!(
        sinogram,
        volume,
        geometry.angles;
        detector_width = geometry.detector_width,
        detector_supersampling = options.detector_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _project_cuda!(sinogram::Array{Float32,2}, volume::Array{Float32,2}, geometry::ParallelVec2DGeometry, backend::CUDABackend, options::ProjectionOptions2D)
    LowLevel.cuda_fp2d_parallel_vec_f32!(
        sinogram,
        volume,
        geometry.vectors;
        detector_supersampling = options.detector_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _project_cuda!(sinogram::Array{Float32,2}, volume::Array{Float32,2}, geometry::FanFlat2DGeometry, backend::CUDABackend, options::ProjectionOptions2D)
    LowLevel.cuda_fp2d_fanflat_f32!(
        sinogram,
        volume,
        geometry.angles;
        detector_width = geometry.detector_width,
        origin_source_distance = geometry.origin_source_distance,
        origin_detector_distance = geometry.origin_detector_distance,
        detector_supersampling = options.detector_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _project_cuda!(sinogram::Array{Float32,2}, volume::Array{Float32,2}, geometry::FanFlatVec2DGeometry, backend::CUDABackend, options::ProjectionOptions2D)
    LowLevel.cuda_fp2d_fanflat_vec_f32!(
        sinogram,
        volume,
        geometry.vectors;
        detector_supersampling = options.detector_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _project_cpu!(sinogram::Array{Float32,2}, volume::Array{Float32,2}, geometry::SparseMatrix2DGeometry, options::ProjectionOptions2D = ProjectionOptions2D())
    LowLevel.fp2d_sparse_matrix_f32!(
        sinogram,
        volume,
        geometry.row_starts,
        geometry.col_indices,
        geometry.values;
        angle_count = geometry.angle_count,
        detector_count = geometry.detector_count,
        nnz = length(geometry.values),
    )
end

function _project_cuda!(sinogram::Array{Float32,2}, volume::Array{Float32,2}, geometry::SparseMatrix2DGeometry, backend::CUDABackend, options::ProjectionOptions2D)
    throw(ArgumentError("Sparse matrix projector is only supported on CPUBackend"))
end

"""
    project!(sinogram::Array{Float32,2}, volume::Array{Float32,2}, geometry::AbstractGeometry2D; ...)

In-place 2D forward projection.

# Arguments
- `sinogram`: 2D array to store the projection result. Size must be `(detector_count, angle_count)`.
- `volume`: 2D array of the volume to project. Size is typically `(nx, ny)`.
- `geometry`: 2D geometry object.
- `backend`: Computational backend (`cpu()` or `cuda()`). Defaults to `cpu()`.
- `options`: Optional `ProjectionOptions2D`.
- `detector_supersampling`: Optional detector supersampling level (GPU only).
- `projector_kernel`: Optional projector kernel (`:line`, `:linear`, or `:strip`, CPU only).

# Returns
- `sinogram`: The updated projection array (mutated in-place).

# Examples
```julia
sinogram = zeros(Float32, 256, 180)
project!(sinogram, volume, geom; backend=cuda())
```
"""
function project!(
    sinogram::Array{Float32,2},
    volume::Array{Float32,2},
    geometry::AbstractGeometry2D;
    backend::AbstractBackend = cpu(),
    options::Union{Nothing,ProjectionOptions2D} = nothing,
    detector_supersampling = nothing,
    projector_kernel = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    _check_volume(volume)
    _check_sinogram(sinogram, geometry)
    options = _merge_projection_options(ProjectionOptions2D, options, detector_supersampling, projector_kernel)
    vol_geom = _resolve_volume_geometry(volume, volume_geometry, window)
    LowLevel.with_volume_window(vol_geom) do
        if backend isa CPUBackend
            options.detector_supersampling == 1 || throw(ArgumentError("2D CPU project does not support detector_supersampling"))
            _project_cpu!(sinogram, volume, geometry, options)
        elseif backend isa CUDABackend
            options.projector_kernel == :line || throw(ArgumentError("CUDA project only supports default projector_kernel :line"))
            _project_cuda!(sinogram, volume, geometry, backend, options)
        else
            throw(ArgumentError("unsupported backend type $(typeof(backend))"))
        end
    end
    return sinogram
end

function project!(
    sinogram::Array{Float32,2},
    volume::Array{Float32,2},
    geometry::AbstractGeometry2D,
    vol_geom::VolumeGeometry2D;
    kwargs...,
)
    return project!(sinogram, volume, geometry; volume_geometry = vol_geom, kwargs...)
end

"""
    project(volume, geometry; ...)
    project(volume, geometry, vol_geom::VolumeGeometry2D; ...)

2D forward projection. Returns a new `Array{Float32,2}`.

# Arguments
- `volume`: 2D array of the volume to project.
- `geometry`: 2D geometry object.
- `vol_geom`: Optional `VolumeGeometry2D` object.
- `backend`: Computational backend (`cpu()` or `cuda()`). Defaults to `cpu()`.
- `options`: Optional `ProjectionOptions2D`.
- `detector_supersampling`: Optional detector supersampling level (GPU only).
- `projector_kernel`: Optional projector kernel (`:line`, `:linear`, `:strip`, or `:distance_driven`, CPU only).
- `volume_geometry`: Optional `VolumeGeometry2D` keyword argument.
- `window`: Optional 4-element collection `(min_x, max_x, min_y, max_y)`.

# Returns
- `sinogram`: 2D array of size `(detector_count, angle_count)`.

# Examples
```julia
sino = project(volume, geom; backend=cuda())
sino_roi = project(volume, geom, VolumeGeometry2D(size(volume); window=(-10, 10, -10, 10)))
```
"""
function project(
    volume::Array{Float32,2},
    geometry::AbstractGeometry2D;
    backend::AbstractBackend = cpu(),
    options::Union{Nothing,ProjectionOptions2D} = nothing,
    detector_supersampling = nothing,
    projector_kernel = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    _check_volume(volume)
    sinogram = Array{Float32}(undef, _detector_count(geometry), _angle_count(geometry))
    return project!(
        sinogram,
        volume,
        geometry;
        backend = backend,
        options = options,
        detector_supersampling = detector_supersampling,
        projector_kernel = projector_kernel,
        volume_geometry = volume_geometry,
        window = window,
    )
end

function project(
    volume::Array{Float32,2},
    geometry::AbstractGeometry2D,
    vol_geom::VolumeGeometry2D;
    kwargs...,
)
    return project(volume, geometry; volume_geometry = vol_geom, kwargs...)
end

function project(
    volume::AbstractMatrix{<:Real},
    geometry::AbstractGeometry2D;
    kwargs...,
)
    return project(
        Matrix{Float32}(volume),
        geometry;
        kwargs...,
    )
end

function project(
    volume::AbstractMatrix{<:Real},
    geometry::AbstractGeometry2D,
    vol_geom::VolumeGeometry2D;
    kwargs...,
)
    return project(
        Matrix{Float32}(volume),
        geometry;
        volume_geometry = vol_geom,
        kwargs...,
    )
end

function _project_cuda!(sinogram::Array{Float32,3}, volume::Array{Float32,3}, geometry::Parallel3DGeometry, backend::CUDABackend, options::ProjectionOptions3D)
    options.detector_supersampling == 1 || throw(ArgumentError("Parallel3DGeometry project does not support detector_supersampling"))
    LowLevel.cuda_fp3d_parallel_f32!(
        sinogram,
        volume,
        geometry.angles;
        detector_spacing_u = geometry.detector_spacing_u,
        detector_spacing_v = geometry.detector_spacing_v,
        gpu_index = backend.gpu_index,
    )
end

function _project_cuda!(sinogram::Array{Float32,3}, volume::Array{Float32,3}, geometry::ParallelVec3DGeometry, backend::CUDABackend, options::ProjectionOptions3D)
    options.detector_supersampling == 1 || throw(ArgumentError("ParallelVec3DGeometry project does not support detector_supersampling"))
    LowLevel.cuda_fp3d_parallel_vec_f32!(sinogram, volume, geometry.vectors; gpu_index = backend.gpu_index)
end

function _project_cuda!(sinogram::Array{Float32,3}, volume::Array{Float32,3}, geometry::Cone3DGeometry, backend::CUDABackend, options::ProjectionOptions3D)
    LowLevel.cuda_fp3d_cone_f32!(
        sinogram,
        volume,
        geometry.angles;
        detector_spacing_u = geometry.detector_spacing_u,
        detector_spacing_v = geometry.detector_spacing_v,
        origin_source_distance = geometry.origin_source_distance,
        origin_detector_distance = geometry.origin_detector_distance,
        detector_supersampling = options.detector_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _project_cuda!(sinogram::Array{Float32,3}, volume::Array{Float32,3}, geometry::ConeVec3DGeometry, backend::CUDABackend, options::ProjectionOptions3D)
    LowLevel.cuda_fp3d_cone_vec_f32!(
        sinogram,
        volume,
        geometry.vectors;
        detector_supersampling = options.detector_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _project_cuda!(sinogram::Array{Float32,3}, volume::Array{Float32,3}, geometry::CylConeVec3DGeometry, backend::CUDABackend, options::ProjectionOptions3D)
    options.detector_supersampling == 1 || throw(ArgumentError("CylConeVec3DGeometry project does not support detector_supersampling"))
    LowLevel.cuda_fp3d_cyl_cone_vec_f32!(sinogram, volume, geometry.vectors; gpu_index = backend.gpu_index)
end

"""
    project!(sinogram::Array{Float32,3}, volume::Array{Float32,3}, geometry::AbstractGeometry3D; ...)

In-place 3D forward projection.

# Arguments
- `sinogram`: 3D array to store the projection result. Size must be `(detector_count_u, angle_count, detector_count_v)`.
- `volume`: 3D array of the volume to project. Size is typically `(nx, ny, nz)`.
- `geometry`: 3D geometry object.
- `backend`: Computational backend (only `cuda()` is supported for 3D in the current high-level API). Defaults to `cuda()`.
- `options`: Optional `ProjectionOptions3D`.
- `detector_supersampling`: Optional detector supersampling level.

# Returns
- `sinogram`: The updated 3D projection array (mutated in-place).

# Examples
```julia
sinogram = zeros(Float32, 256, 180, 256)
project!(sinogram, volume, geom; backend=cuda())
```
"""
function project!(
    sinogram::Array{Float32,3},
    volume::Array{Float32,3},
    geometry::AbstractGeometry3D;
    backend::AbstractBackend = cuda(),
    options::Union{Nothing,ProjectionOptions3D} = nothing,
    detector_supersampling = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    _check_volume(volume)
    _check_sinogram(sinogram, geometry)
    options = _merge_projection_options(ProjectionOptions3D, options, detector_supersampling)
    backend isa CUDABackend || throw(ArgumentError("3D project is only available on CUDA in the current high-level API"))
    vol_geom = _resolve_volume_geometry(volume, volume_geometry, window)
    LowLevel.with_volume_window(vol_geom) do
        _project_cuda!(sinogram, volume, geometry, backend, options)
    end
    return sinogram
end

function project!(
    sinogram::Array{Float32,3},
    volume::Array{Float32,3},
    geometry::AbstractGeometry3D,
    vol_geom::VolumeGeometry3D;
    kwargs...,
)
    return project!(sinogram, volume, geometry; volume_geometry = vol_geom, kwargs...)
end

"""
    project(volume, geometry; ...)
    project(volume, geometry, vol_geom::VolumeGeometry3D; ...)

3D forward projection. Returns a new `Array{Float32,3}`.

# Arguments
- `volume`: 3D array of the volume to project.
- `geometry`: 3D geometry object.
- `vol_geom`: Optional `VolumeGeometry3D` object.
- `backend`: Computational backend (only `cuda()` is supported for 3D in the current high-level API). Defaults to `cuda()`.
- `options`: Optional `ProjectionOptions3D`.
- `detector_supersampling`: Optional detector supersampling level.
- `volume_geometry`: Optional `VolumeGeometry3D` keyword argument.
- `window`: Optional 6-element collection `(min_x, max_x, min_y, max_y, min_z, max_z)`.

# Returns
- `sinogram`: 3D array of size `(detector_count_u, angle_count, detector_count_v)`.

# Examples
```julia
sino = project(volume, geom; backend=cuda())
sino_roi = project(volume, geom, VolumeGeometry3D(size(volume); window=(-10, 10, -10, 10, -5, 5)))
```
"""
function project(
    volume::Array{Float32,3},
    geometry::AbstractGeometry3D;
    backend::AbstractBackend = cuda(),
    options::Union{Nothing,ProjectionOptions3D} = nothing,
    detector_supersampling = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    _check_volume(volume)
    sinogram = Array{Float32}(undef, _detector_count_u(geometry), _angle_count(geometry), _detector_count_v(geometry))
    return project!(
        sinogram,
        volume,
        geometry;
        backend = backend,
        options = options,
        detector_supersampling = detector_supersampling,
        volume_geometry = volume_geometry,
        window = window,
    )
end

function project(
    volume::Array{Float32,3},
    geometry::AbstractGeometry3D,
    vol_geom::VolumeGeometry3D;
    kwargs...,
)
    return project(volume, geometry; volume_geometry = vol_geom, kwargs...)
end

function project(
    volume::AbstractArray{<:Real,3},
    geometry::AbstractGeometry3D;
    kwargs...,
)
    return project(
        Array{Float32,3}(volume),
        geometry;
        kwargs...,
    )
end

function project(
    volume::AbstractArray{<:Real,3},
    geometry::AbstractGeometry3D,
    vol_geom::VolumeGeometry3D;
    kwargs...,
)
    return project(
        Array{Float32,3}(volume),
        geometry;
        volume_geometry = vol_geom,
        kwargs...,
    )
end

