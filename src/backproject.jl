function _backproject_cpu!(volume::Array{Float32,2}, sinogram::Array{Float32,2}, geometry::Parallel2DGeometry, options::BackprojectionOptions2D = BackprojectionOptions2D())
    kernel = options.projector_kernel
    if kernel == :line
        LowLevel.bp2d_parallel_line_f32!(
            volume,
            sinogram,
            geometry.angles;
            detector_width = geometry.detector_width,
        )
    elseif kernel == :linear
        LowLevel.bp2d_parallel_linear_f32!(
            volume,
            sinogram,
            geometry.angles;
            detector_width = geometry.detector_width,
        )
    elseif kernel == :strip
        LowLevel.bp2d_parallel_strip_f32!(
            volume,
            sinogram,
            geometry.angles;
            detector_width = geometry.detector_width,
        )
    elseif kernel == :distance_driven
        LowLevel.bp2d_parallel_distance_driven_f32!(
            volume,
            sinogram,
            geometry.angles;
            detector_width = geometry.detector_width,
        )
    else
        throw(ArgumentError("unsupported projector_kernel $kernel"))
    end
end

function _backproject_cpu!(volume::Array{Float32,2}, sinogram::Array{Float32,2}, geometry::ParallelVec2DGeometry, options::BackprojectionOptions2D = BackprojectionOptions2D())
    kernel = options.projector_kernel
    if kernel == :line
        LowLevel.bp2d_parallel_vec_line_f32!(volume, sinogram, geometry.vectors)
    elseif kernel == :linear
        LowLevel.bp2d_parallel_vec_linear_f32!(volume, sinogram, geometry.vectors)
    elseif kernel == :strip
        LowLevel.bp2d_parallel_vec_strip_f32!(volume, sinogram, geometry.vectors)
    elseif kernel == :distance_driven
        LowLevel.bp2d_parallel_vec_distance_driven_f32!(volume, sinogram, geometry.vectors)
    else
        throw(ArgumentError("unsupported projector_kernel $kernel"))
    end
end

function _backproject_cpu!(volume::Array{Float32,2}, sinogram::Array{Float32,2}, geometry::FanFlat2DGeometry, options::BackprojectionOptions2D = BackprojectionOptions2D())
    kernel = options.projector_kernel
    if kernel == :line
        LowLevel.bp2d_fanflat_line_f32!(
            volume,
            sinogram,
            geometry.angles;
            detector_width = geometry.detector_width,
            origin_source_distance = geometry.origin_source_distance,
            origin_detector_distance = geometry.origin_detector_distance,
        )
    elseif kernel == :strip
        LowLevel.bp2d_fanflat_strip_f32!(
            volume,
            sinogram,
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

function _backproject_cpu!(volume::Array{Float32,2}, sinogram::Array{Float32,2}, geometry::FanFlatVec2DGeometry, options::BackprojectionOptions2D = BackprojectionOptions2D())
    kernel = options.projector_kernel
    if kernel == :line
        LowLevel.bp2d_fanflat_vec_line_f32!(volume, sinogram, geometry.vectors)
    elseif kernel in (:linear, :strip, :distance_driven)
        throw(ArgumentError("projector_kernel :$kernel is not supported for fan-flat vector geometry; use :line"))
    else
        throw(ArgumentError("unsupported projector_kernel $kernel"))
    end
end

function _backproject_cuda!(volume::Array{Float32,2}, sinogram::Array{Float32,2}, geometry::Parallel2DGeometry, backend::CUDABackend, options::BackprojectionOptions2D)
    LowLevel.cuda_bp2d_parallel_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        detector_supersampling = options.detector_supersampling,
        pixel_supersampling = options.pixel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _backproject_cuda!(volume::Array{Float32,2}, sinogram::Array{Float32,2}, geometry::ParallelVec2DGeometry, backend::CUDABackend, options::BackprojectionOptions2D)
    LowLevel.cuda_bp2d_parallel_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        detector_supersampling = options.detector_supersampling,
        pixel_supersampling = options.pixel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _backproject_cuda!(volume::Array{Float32,2}, sinogram::Array{Float32,2}, geometry::FanFlat2DGeometry, backend::CUDABackend, options::BackprojectionOptions2D)
    LowLevel.cuda_bp2d_fanflat_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        origin_source_distance = geometry.origin_source_distance,
        origin_detector_distance = geometry.origin_detector_distance,
        detector_supersampling = options.detector_supersampling,
        pixel_supersampling = options.pixel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _backproject_cuda!(volume::Array{Float32,2}, sinogram::Array{Float32,2}, geometry::FanFlatVec2DGeometry, backend::CUDABackend, options::BackprojectionOptions2D)
    LowLevel.cuda_bp2d_fanflat_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        detector_supersampling = options.detector_supersampling,
        pixel_supersampling = options.pixel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _backproject_cpu!(volume::Array{Float32,2}, sinogram::Array{Float32,2}, geometry::SparseMatrix2DGeometry, options::BackprojectionOptions2D = BackprojectionOptions2D())
    LowLevel.bp2d_sparse_matrix_f32!(
        volume,
        sinogram,
        geometry.row_starts,
        geometry.col_indices,
        geometry.values;
        angle_count = geometry.angle_count,
        detector_count = geometry.detector_count,
        nnz = length(geometry.values),
    )
end

function _backproject_cuda!(volume::Array{Float32,2}, sinogram::Array{Float32,2}, geometry::SparseMatrix2DGeometry, backend::CUDABackend, options::BackprojectionOptions2D)
    throw(ArgumentError("Sparse matrix projector is only supported on CPUBackend"))
end

"""
    backproject!(volume::Array{Float32,2}, sinogram::Array{Float32,2}, geometry::AbstractGeometry2D; ...)

In-place 2D backprojection.

# Arguments
- `volume`: 2D array to store the backprojection result. Size is typically `(nx, ny)`.
- `sinogram`: 2D array of the sinogram to backproject. Size must be `(detector_count, angle_count)`.
- `geometry`: 2D geometry object.
- `backend`: Computational backend (`cpu()` or `cuda()`). Defaults to `cpu()`.
- `options`: Optional `BackprojectionOptions2D`.
- `detector_supersampling`: Optional detector supersampling level (GPU only).
- `pixel_supersampling`: Optional pixel supersampling level (GPU only).
- `projector_kernel`: Optional projector kernel (`:line`, `:linear`, or `:strip`, CPU only).

# Returns
- `volume`: The updated volume array (mutated in-place).

# Examples
```julia
volume = zeros(Float32, 256, 256)
backproject!(volume, sinogram, geom; backend=cuda())
```
"""
function backproject!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::AbstractGeometry2D;
    backend::AbstractBackend = cpu(),
    options::Union{Nothing,BackprojectionOptions2D} = nothing,
    detector_supersampling = nothing,
    pixel_supersampling = nothing,
    projector_kernel = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    _check_volume(volume)
    _check_sinogram(sinogram, geometry)
    options = _merge_backprojection_options(BackprojectionOptions2D, options, detector_supersampling, pixel_supersampling, projector_kernel)
    vol_geom = _resolve_volume_geometry(volume, volume_geometry, window)
    LowLevel.with_volume_window(vol_geom) do
        if backend isa CPUBackend
            options.detector_supersampling == 1 || throw(ArgumentError("2D CPU backproject does not support detector_supersampling"))
            options.pixel_supersampling == 1 || throw(ArgumentError("2D CPU backproject does not support pixel_supersampling"))
            _backproject_cpu!(volume, sinogram, geometry, options)
        elseif backend isa CUDABackend
            options.projector_kernel == :line || throw(ArgumentError("CUDA backproject only supports default projector_kernel :line"))
            _backproject_cuda!(volume, sinogram, geometry, backend, options)
        else
            throw(ArgumentError("unsupported backend type $(typeof(backend))"))
        end
    end
    return volume
end

function backproject!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::AbstractGeometry2D,
    vol_geom::VolumeGeometry2D;
    kwargs...,
)
    return backproject!(volume, sinogram, geometry; volume_geometry = vol_geom, kwargs...)
end

"""
    backproject(sinogram, geometry; volume_size=nothing, volume_geometry=nothing, window=nothing, ...)
    backproject(sinogram, geometry, vol_geom::VolumeGeometry2D; ...)

2D backprojection. Returns a new `Array{Float32,2}`.

# Arguments
- `sinogram`: 2D array of the sinogram to backproject. Size must be `(detector_count, angle_count)`.
- `geometry`: 2D geometry object.
- `vol_geom`: Optional `VolumeGeometry2D` object.
- `volume_size`: Tuple `(nx, ny)` specifying the size of the output volume (optional if `volume_geometry` is provided).
- `volume_geometry`: Optional `VolumeGeometry2D` keyword argument.
- `window`: Optional 4-element collection `(min_x, max_x, min_y, max_y)`.
- `backend`: Computational backend (`cpu()` or `cuda()`). Defaults to `cpu()`.
- `options`: Optional `BackprojectionOptions2D`.
- `detector_supersampling`: Optional detector supersampling level (GPU only).
- `pixel_supersampling`: Optional pixel supersampling level (GPU only).
- `projector_kernel`: Optional projector kernel (`:line`, `:linear`, `:strip`, or `:distance_driven`, CPU only).

# Returns
- `volume`: 2D array of size `volume_size`.

# Examples
```julia
volume = backproject(sinogram, geom; volume_size=(256, 256), backend=cuda())
volume_roi = backproject(sinogram, geom, VolumeGeometry2D(256, 256; window=(-10, 10, -10, 10)))
```
"""
function backproject(
    sinogram::Array{Float32,2},
    geometry::AbstractGeometry2D;
    volume_size::Union{Nothing,Tuple{Int,Int}} = nothing,
    backend::AbstractBackend = cpu(),
    options::Union{Nothing,BackprojectionOptions2D} = nothing,
    detector_supersampling = nothing,
    pixel_supersampling = nothing,
    projector_kernel = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    vol_geom = if volume_geometry !== nothing
        window === nothing || throw(ArgumentError("cannot specify both `volume_geometry` and `window`"))
        volume_size === nothing || volume_size == (volume_geometry.width, volume_geometry.height) || throw(ArgumentError("volume_size does not match volume_geometry"))
        volume_geometry
    else
        volume_size === nothing && throw(ArgumentError("must specify either `volume_size` or `volume_geometry`"))
        VolumeGeometry2D(volume_size; window = window)
    end
    volume = Array{Float32}(undef, vol_geom.width, vol_geom.height)
    return backproject!(
        volume,
        sinogram,
        geometry;
        backend = backend,
        options = options,
        detector_supersampling = detector_supersampling,
        pixel_supersampling = pixel_supersampling,
        projector_kernel = projector_kernel,
        volume_geometry = vol_geom,
    )
end

function backproject(
    sinogram::Array{Float32,2},
    geometry::AbstractGeometry2D,
    vol_geom::VolumeGeometry2D;
    kwargs...,
)
    return backproject(sinogram, geometry; volume_geometry = vol_geom, kwargs...)
end

function backproject(
    sinogram::AbstractMatrix{<:Real},
    geometry::AbstractGeometry2D;
    kwargs...,
)
    return backproject(
        Matrix{Float32}(sinogram),
        geometry;
        kwargs...,
    )
end

function backproject(
    sinogram::AbstractMatrix{<:Real},
    geometry::AbstractGeometry2D,
    vol_geom::VolumeGeometry2D;
    kwargs...,
)
    return backproject(
        Matrix{Float32}(sinogram),
        geometry;
        volume_geometry = vol_geom,
        kwargs...,
    )
end

function backproject(
    sinogram::Array{Float32,2},
    geometry::SparseMatrix2DGeometry;
    volume_size::Union{Nothing,Tuple{Int,Int}} = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
    backend::AbstractBackend = cpu(),
    options::Union{Nothing,BackprojectionOptions2D} = nothing,
    detector_supersampling = nothing,
    pixel_supersampling = nothing,
    projector_kernel = nothing,
)
    vol_geom = if volume_geometry !== nothing
        volume_geometry
    elseif volume_size !== nothing
        VolumeGeometry2D(volume_size; window = window)
    else
        VolumeGeometry2D((geometry.volume_width, geometry.volume_height); window = window)
    end
    volume = zeros(Float32, vol_geom.width, vol_geom.height)
    return backproject!(
        volume,
        sinogram,
        geometry;
        backend = backend,
        options = options,
        detector_supersampling = detector_supersampling,
        pixel_supersampling = pixel_supersampling,
        projector_kernel = projector_kernel,
        volume_geometry = vol_geom,
    )
end

function backproject(
    sinogram::Array{Float32,2},
    geometry::SparseMatrix2DGeometry,
    vol_geom::VolumeGeometry2D;
    kwargs...,
)
    return backproject(sinogram, geometry; volume_geometry = vol_geom, kwargs...)
end

function backproject(
    sinogram::AbstractMatrix{<:Real},
    geometry::SparseMatrix2DGeometry;
    kwargs...,
)
    return backproject(
        Matrix{Float32}(sinogram),
        geometry;
        kwargs...,
    )
end

function backproject(
    sinogram::AbstractMatrix{<:Real},
    geometry::SparseMatrix2DGeometry,
    vol_geom::VolumeGeometry2D;
    kwargs...,
)
    return backproject(
        Matrix{Float32}(sinogram),
        geometry;
        volume_geometry = vol_geom,
        kwargs...,
    )
end

function _backproject_cuda!(volume::Array{Float32,3}, sinogram::Array{Float32,3}, geometry::Parallel3DGeometry, backend::CUDABackend, options::BackprojectionOptions3D)
    LowLevel.cuda_bp3d_parallel_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_spacing_u = geometry.detector_spacing_u,
        detector_spacing_v = geometry.detector_spacing_v,
        voxel_supersampling = options.voxel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _backproject_cuda!(volume::Array{Float32,3}, sinogram::Array{Float32,3}, geometry::ParallelVec3DGeometry, backend::CUDABackend, options::BackprojectionOptions3D)
    LowLevel.cuda_bp3d_parallel_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        voxel_supersampling = options.voxel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _backproject_cuda!(volume::Array{Float32,3}, sinogram::Array{Float32,3}, geometry::Cone3DGeometry, backend::CUDABackend, options::BackprojectionOptions3D)
    LowLevel.cuda_bp3d_cone_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_spacing_u = geometry.detector_spacing_u,
        detector_spacing_v = geometry.detector_spacing_v,
        origin_source_distance = geometry.origin_source_distance,
        origin_detector_distance = geometry.origin_detector_distance,
        voxel_supersampling = options.voxel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _backproject_cuda!(volume::Array{Float32,3}, sinogram::Array{Float32,3}, geometry::ConeVec3DGeometry, backend::CUDABackend, options::BackprojectionOptions3D)
    LowLevel.cuda_bp3d_cone_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        voxel_supersampling = options.voxel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _backproject_cuda!(volume::Array{Float32,3}, sinogram::Array{Float32,3}, geometry::CylConeVec3DGeometry, backend::CUDABackend, options::BackprojectionOptions3D)
    LowLevel.cuda_bp3d_cyl_cone_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        voxel_supersampling = options.voxel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

"""
    backproject!(volume::Array{Float32,3}, sinogram::Array{Float32,3}, geometry::AbstractGeometry3D; ...)

In-place 3D backprojection.

# Arguments
- `volume`: 3D array to store the backprojection result. Size is typically `(nx, ny, nz)`.
- `sinogram`: 3D array of the sinogram to backproject. Size must be `(detector_count_u, angle_count, detector_count_v)`.
- `geometry`: 3D geometry object.
- `backend`: Computational backend (only `cuda()` is supported for 3D in the current high-level API). Defaults to `cuda()`.
- `options`: Optional `BackprojectionOptions3D`.
- `voxel_supersampling`: Optional voxel supersampling level.

# Returns
- `volume`: The updated 3D volume array (mutated in-place).

# Examples
```julia
volume = zeros(Float32, 256, 256, 256)
backproject!(volume, sinogram, geom; backend=cuda())
```
"""
function backproject!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geometry::AbstractGeometry3D;
    backend::AbstractBackend = cuda(),
    options::Union{Nothing,BackprojectionOptions3D} = nothing,
    voxel_supersampling::Integer = 1,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    _check_volume(volume)
    _check_sinogram(sinogram, geometry)
    options = _merge_backprojection_options(BackprojectionOptions3D, options, voxel_supersampling)
    backend isa CUDABackend || throw(ArgumentError("3D backproject is only available on CUDA in the current high-level API"))
    vol_geom = _resolve_volume_geometry(volume, volume_geometry, window)
    LowLevel.with_volume_window(vol_geom) do
        _backproject_cuda!(volume, sinogram, geometry, backend, options)
    end
    return volume
end

function backproject!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geometry::AbstractGeometry3D,
    vol_geom::VolumeGeometry3D;
    kwargs...,
)
    return backproject!(volume, sinogram, geometry; volume_geometry = vol_geom, kwargs...)
end

"""
    backproject(sinogram, geometry; volume_size=nothing, volume_geometry=nothing, window=nothing, ...)
    backproject(sinogram, geometry, vol_geom::VolumeGeometry3D; ...)

3D backprojection. Returns a new `Array{Float32,3}`.

# Arguments
- `sinogram`: 3D array of the sinogram to backproject. Size must be `(detector_count_u, angle_count, detector_count_v)`.
- `geometry`: 3D geometry object.
- `vol_geom`: Optional `VolumeGeometry3D` object.
- `volume_size`: Tuple `(nx, ny, nz)` specifying the size of the output volume (optional if `volume_geometry` is provided).
- `volume_geometry`: Optional `VolumeGeometry3D` keyword argument.
- `window`: Optional 6-element collection `(min_x, max_x, min_y, max_y, min_z, max_z)`.
- `backend`: Computational backend (only `cuda()` is supported for 3D in the current high-level API). Defaults to `cuda()`.
- `options`: Optional `BackprojectionOptions3D`.
- `voxel_supersampling`: Optional voxel supersampling level.

# Returns
- `volume`: 3D array of size `volume_size`.

# Examples
```julia
volume = backproject(sinogram, geom; volume_size=(256, 256, 256), backend=cuda())
volume_roi = backproject(sinogram, geom, VolumeGeometry3D(256, 256, 256; window=(-10, 10, -10, 10, -5, 5)))
```
"""
function backproject(
    sinogram::Array{Float32,3},
    geometry::AbstractGeometry3D;
    volume_size::Union{Nothing,Tuple{Int,Int,Int}} = nothing,
    backend::AbstractBackend = cuda(),
    options::Union{Nothing,BackprojectionOptions3D} = nothing,
    voxel_supersampling::Integer = 1,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    vol_geom = if volume_geometry !== nothing
        window === nothing || throw(ArgumentError("cannot specify both `volume_geometry` and `window`"))
        volume_size === nothing || volume_size == (volume_geometry.width, volume_geometry.height, volume_geometry.depth) || throw(ArgumentError("volume_size does not match volume_geometry"))
        volume_geometry
    else
        volume_size === nothing && throw(ArgumentError("must specify either `volume_size` or `volume_geometry`"))
        VolumeGeometry3D(volume_size; window = window)
    end
    volume = Array{Float32}(undef, vol_geom.width, vol_geom.height, vol_geom.depth)
    return backproject!(
        volume,
        sinogram,
        geometry;
        backend = backend,
        options = options,
        voxel_supersampling = voxel_supersampling,
        volume_geometry = vol_geom,
    )
end

function backproject(
    sinogram::Array{Float32,3},
    geometry::AbstractGeometry3D,
    vol_geom::VolumeGeometry3D;
    kwargs...,
)
    return backproject(sinogram, geometry; volume_geometry = vol_geom, kwargs...)
end

function backproject(
    sinogram::AbstractArray{<:Real,3},
    geometry::AbstractGeometry3D;
    kwargs...,
)
    return backproject(
        Array{Float32,3}(sinogram),
        geometry;
        kwargs...,
    )
end

function backproject(
    sinogram::AbstractArray{<:Real,3},
    geometry::AbstractGeometry3D,
    vol_geom::VolumeGeometry3D;
    kwargs...,
)
    return backproject(
        Array{Float32,3}(sinogram),
        geometry;
        volume_geometry = vol_geom,
        kwargs...,
    )
end

