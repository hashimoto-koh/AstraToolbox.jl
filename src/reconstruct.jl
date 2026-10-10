function _fbp_cpu!(volume::Array{Float32,2}, sinogram::Array{Float32,2}, geometry::Parallel2DGeometry, method::FBPMethod)
    LowLevel.fbp2d_parallel_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        filter_type = method.filter_type,
        filter_d = method.filter_d,
        filter_parameter = method.filter_parameter,
        custom_filter = method.custom_filter,
    )
end

function _fbp_cuda!(volume::Array{Float32,2}, sinogram::Array{Float32,2}, geometry::Parallel2DGeometry, method::FBPMethod, backend::CUDABackend)
    LowLevel.cuda_fbp2d_parallel_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        filter_type = method.filter_type,
        filter_d = method.filter_d,
        filter_parameter = method.filter_parameter,
        custom_filter = method.custom_filter,
        gpu_index = backend.gpu_index,
    )
end

function _fbp_cuda!(volume::Array{Float32,2}, sinogram::Array{Float32,2}, geometry::ParallelVec2DGeometry, method::FBPMethod, backend::CUDABackend)
    LowLevel.cuda_fbp2d_parallel_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        filter_type = method.filter_type,
        filter_d = method.filter_d,
        filter_parameter = method.filter_parameter,
        custom_filter = method.custom_filter,
        gpu_index = backend.gpu_index,
    )
end

function _fbp_cuda!(volume::Array{Float32,2}, sinogram::Array{Float32,2}, geometry::FanFlat2DGeometry, method::FBPMethod, backend::CUDABackend)
    LowLevel.cuda_fbp2d_fanflat_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        origin_source_distance = geometry.origin_source_distance,
        origin_detector_distance = geometry.origin_detector_distance,
        filter_type = method.filter_type,
        filter_d = method.filter_d,
        filter_parameter = method.filter_parameter,
        custom_filter = method.custom_filter,
        short_scan = method.short_scan,
        gpu_index = backend.gpu_index,
    )
end

function _fbp_cuda!(volume::Array{Float32,2}, sinogram::Array{Float32,2}, geometry::FanFlatVec2DGeometry, method::FBPMethod, backend::CUDABackend)
    LowLevel.cuda_fbp2d_fanflat_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        filter_type = method.filter_type,
        filter_d = method.filter_d,
        filter_parameter = method.filter_parameter,
        custom_filter = method.custom_filter,
        short_scan = method.short_scan,
        gpu_index = backend.gpu_index,
    )
end

function _sirt_cuda!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::Parallel2DGeometry,
    method::SIRTMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_sirt2d_parallel_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        iterations = method.iterations,
        relaxation = method.relaxation,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        residual_norm = residual_norm,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
        detector_supersampling = method.detector_supersampling,
        pixel_supersampling = method.pixel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _sirt_cuda!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::ParallelVec2DGeometry,
    method::SIRTMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_sirt2d_parallel_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        relaxation = method.relaxation,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        residual_norm = residual_norm,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
        detector_supersampling = method.detector_supersampling,
        pixel_supersampling = method.pixel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _sirt_cuda!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::FanFlat2DGeometry,
    method::SIRTMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_sirt2d_fanflat_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        origin_source_distance = geometry.origin_source_distance,
        origin_detector_distance = geometry.origin_detector_distance,
        iterations = method.iterations,
        relaxation = method.relaxation,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        residual_norm = residual_norm,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
        detector_supersampling = method.detector_supersampling,
        pixel_supersampling = method.pixel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _sirt_cuda!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::FanFlatVec2DGeometry,
    method::SIRTMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_sirt2d_fanflat_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        relaxation = method.relaxation,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        residual_norm = residual_norm,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
        detector_supersampling = method.detector_supersampling,
        pixel_supersampling = method.pixel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _sart_cuda!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::Parallel2DGeometry,
    method::SARTMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_sart2d_parallel_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        iterations = method.iterations,
        relaxation = method.relaxation,
        reconstruction_mask = reconstruction_mask,
        residual_norm = residual_norm,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
        projection_order = method.projection_order,
        projection_order_list = method.projection_order_list,
        detector_supersampling = method.detector_supersampling,
        pixel_supersampling = method.pixel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _sart_cuda!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::ParallelVec2DGeometry,
    method::SARTMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_sart2d_parallel_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        relaxation = method.relaxation,
        reconstruction_mask = reconstruction_mask,
        residual_norm = residual_norm,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
        projection_order = method.projection_order,
        projection_order_list = method.projection_order_list,
        detector_supersampling = method.detector_supersampling,
        pixel_supersampling = method.pixel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _sart_cuda!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::FanFlat2DGeometry,
    method::SARTMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_sart2d_fanflat_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        origin_source_distance = geometry.origin_source_distance,
        origin_detector_distance = geometry.origin_detector_distance,
        iterations = method.iterations,
        relaxation = method.relaxation,
        reconstruction_mask = reconstruction_mask,
        residual_norm = residual_norm,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
        projection_order = method.projection_order,
        projection_order_list = method.projection_order_list,
        detector_supersampling = method.detector_supersampling,
        pixel_supersampling = method.pixel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _sart_cuda!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::FanFlatVec2DGeometry,
    method::SARTMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_sart2d_fanflat_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        relaxation = method.relaxation,
        reconstruction_mask = reconstruction_mask,
        residual_norm = residual_norm,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
        projection_order = method.projection_order,
        projection_order_list = method.projection_order_list,
        detector_supersampling = method.detector_supersampling,
        pixel_supersampling = method.pixel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _cgls_cuda!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::Parallel2DGeometry,
    method::CGLSMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_cgls2d_parallel_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        iterations = method.iterations,
        reconstruction_mask = reconstruction_mask,
        residual_norm = residual_norm,
        detector_supersampling = method.detector_supersampling,
        pixel_supersampling = method.pixel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _cgls_cuda!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::ParallelVec2DGeometry,
    method::CGLSMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_cgls2d_parallel_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        reconstruction_mask = reconstruction_mask,
        residual_norm = residual_norm,
        detector_supersampling = method.detector_supersampling,
        pixel_supersampling = method.pixel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _cgls_cuda!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::FanFlat2DGeometry,
    method::CGLSMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_cgls2d_fanflat_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        origin_source_distance = geometry.origin_source_distance,
        origin_detector_distance = geometry.origin_detector_distance,
        iterations = method.iterations,
        reconstruction_mask = reconstruction_mask,
        residual_norm = residual_norm,
        detector_supersampling = method.detector_supersampling,
        pixel_supersampling = method.pixel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _cgls_cuda!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::FanFlatVec2DGeometry,
    method::CGLSMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_cgls2d_fanflat_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        reconstruction_mask = reconstruction_mask,
        residual_norm = residual_norm,
        detector_supersampling = method.detector_supersampling,
        pixel_supersampling = method.pixel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _em_cuda!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::Parallel2DGeometry,
    method::EMMethod,
    backend::CUDABackend;
    residual_norm = nothing,
)
    LowLevel.cuda_em2d_parallel_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        iterations = method.iterations,
        residual_norm = residual_norm,
        detector_supersampling = method.detector_supersampling,
        pixel_supersampling = method.pixel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

"""
    reconstruct!(volume::Array{Float32,2}, sinogram::Array{Float32,2}, geometry::AbstractGeometry2D, method::AbstractMethod; ...)

In-place 2D reconstruction.

# Arguments
- `volume`: 2D array to store the reconstruction result. Size is typically `(nx, ny)`.
- `sinogram`: 2D array of the sinogram to reconstruct. Size must be `(detector_count, angle_count)`.
- `geometry`: 2D geometry object.
- `method`: Reconstruction method (e.g., `FBPMethod`, `SIRTMethod`, `SARTMethod`, `CGLSMethod`, `EMMethod`).
- `backend`: Computational backend. Defaults depend on the method (FBP: `cpu()`, iterative methods: `cuda()`).
- `options`: Optional `ReconstructionOptions2D`.
- `reconstruction_mask`: Optional 2D mask for the reconstruction volume.
- `sinogram_mask`: Optional 2D mask for the sinogram.
- `residual_norm`: Optional `Ref{Float32}` to store the final residual norm (for iterative methods).

# Returns
- `volume`: The updated volume array (mutated in-place).

# Examples
```julia
volume = zeros(Float32, 256, 256)
method = FBPMethod()
reconstruct!(volume, sinogram, geom, method)
```
"""
function reconstruct!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::Parallel2DGeometry,
    method::FBPMethod;
    backend::AbstractBackend = cpu(),
    options::Union{Nothing,ReconstructionOptions2D} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    _check_volume(volume)
    _check_sinogram(sinogram, geometry)
    options = _merge_reconstruction_options(ReconstructionOptions2D, options, reconstruction_mask, sinogram_mask, residual_norm)
    options.reconstruction_mask === nothing || throw(ArgumentError("FBP does not support reconstruction_mask"))
    options.sinogram_mask === nothing || throw(ArgumentError("FBP does not support sinogram_mask"))
    options.residual_norm === nothing || throw(ArgumentError("FBP does not support residual_norm"))
    vol_geom = _resolve_volume_geometry(volume, volume_geometry, window)
    LowLevel.with_volume_window(vol_geom) do
        if backend isa CPUBackend
            _fbp_cpu!(volume, sinogram, geometry, method)
        elseif backend isa CUDABackend
            _fbp_cuda!(volume, sinogram, geometry, method, backend)
        else
            throw(ArgumentError("unsupported backend type $(typeof(backend))"))
        end
    end
    return volume
end

function reconstruct!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::Union{ParallelVec2DGeometry,FanFlat2DGeometry,FanFlatVec2DGeometry},
    method::FBPMethod;
    backend::AbstractBackend = cpu(),
    options::Union{Nothing,ReconstructionOptions2D} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    _check_volume(volume)
    _check_sinogram(sinogram, geometry)
    options = _merge_reconstruction_options(ReconstructionOptions2D, options, reconstruction_mask, sinogram_mask, residual_norm)
    options.reconstruction_mask === nothing || throw(ArgumentError("FBP does not support reconstruction_mask"))
    options.sinogram_mask === nothing || throw(ArgumentError("FBP does not support sinogram_mask"))
    options.residual_norm === nothing || throw(ArgumentError("FBP does not support residual_norm"))
    backend isa CUDABackend || throw(ArgumentError("FBP is only available on CUDA for this geometry"))
    vol_geom = _resolve_volume_geometry(volume, volume_geometry, window)
    LowLevel.with_volume_window(vol_geom) do
        _fbp_cuda!(volume, sinogram, geometry, method, backend)
    end
    return volume
end

# ---------------- CPU Iterative Reconstruction Helpers ----------------
function _sirt_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::Parallel2DGeometry,
    method::SIRTMethod;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
)
    LowLevel.sirt2d_parallel_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        iterations = method.iterations,
        relaxation = method.relaxation,
        projector_kernel = method.projector_kernel,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
    )
end

function _sirt_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::ParallelVec2DGeometry,
    method::SIRTMethod;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
)
    LowLevel.sirt2d_parallel_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        relaxation = method.relaxation,
        projector_kernel = method.projector_kernel,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
    )
end

function _sirt_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::FanFlat2DGeometry,
    method::SIRTMethod;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
)
    LowLevel.sirt2d_fanflat_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        origin_source_distance = geometry.origin_source_distance,
        origin_detector_distance = geometry.origin_detector_distance,
        iterations = method.iterations,
        relaxation = method.relaxation,
        projector_kernel = method.projector_kernel,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
    )
end

function _sirt_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::FanFlatVec2DGeometry,
    method::SIRTMethod;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
)
    LowLevel.sirt2d_fanflat_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        relaxation = method.relaxation,
        projector_kernel = method.projector_kernel,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
    )
end

function _sirt_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::SparseMatrix2DGeometry,
    method::SIRTMethod;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
)
    LowLevel.sirt2d_sparse_matrix_f32!(
        volume,
        sinogram,
        geometry.row_starts,
        geometry.col_indices,
        geometry.values;
        angle_count = geometry.angle_count,
        detector_count = geometry.detector_count,
        nnz = length(geometry.values),
        iterations = method.iterations,
        relaxation = method.relaxation,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
    )
end

function _sart_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::Parallel2DGeometry,
    method::SARTMethod;
    reconstruction_mask = nothing,
)
    LowLevel.sart2d_parallel_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        iterations = method.iterations,
        relaxation = method.relaxation,
        projection_order = method.projection_order,
        projection_order_list = method.projection_order_list,
        projector_kernel = method.projector_kernel,
        reconstruction_mask = reconstruction_mask,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
    )
end

function _sart_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::ParallelVec2DGeometry,
    method::SARTMethod;
    reconstruction_mask = nothing,
)
    LowLevel.sart2d_parallel_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        relaxation = method.relaxation,
        projection_order = method.projection_order,
        projection_order_list = method.projection_order_list,
        projector_kernel = method.projector_kernel,
        reconstruction_mask = reconstruction_mask,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
    )
end

function _sart_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::FanFlat2DGeometry,
    method::SARTMethod;
    reconstruction_mask = nothing,
)
    LowLevel.sart2d_fanflat_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        origin_source_distance = geometry.origin_source_distance,
        origin_detector_distance = geometry.origin_detector_distance,
        iterations = method.iterations,
        relaxation = method.relaxation,
        projection_order = method.projection_order,
        projection_order_list = method.projection_order_list,
        projector_kernel = method.projector_kernel,
        reconstruction_mask = reconstruction_mask,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
    )
end

function _sart_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::FanFlatVec2DGeometry,
    method::SARTMethod;
    reconstruction_mask = nothing,
)
    LowLevel.sart2d_fanflat_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        relaxation = method.relaxation,
        projection_order = method.projection_order,
        projection_order_list = method.projection_order_list,
        projector_kernel = method.projector_kernel,
        reconstruction_mask = reconstruction_mask,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
    )
end

function _sart_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::SparseMatrix2DGeometry,
    method::SARTMethod;
    reconstruction_mask = nothing,
)
    LowLevel.sart2d_sparse_matrix_f32!(
        volume,
        sinogram,
        geometry.row_starts,
        geometry.col_indices,
        geometry.values;
        angle_count = geometry.angle_count,
        detector_count = geometry.detector_count,
        nnz = length(geometry.values),
        iterations = method.iterations,
        relaxation = method.relaxation,
        projection_order = method.projection_order,
        projection_order_list = method.projection_order_list,
        reconstruction_mask = reconstruction_mask,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
    )
end

function _cgls_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::Parallel2DGeometry,
    method::CGLSMethod;
    reconstruction_mask = nothing,
)
    LowLevel.cgls2d_parallel_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        iterations = method.iterations,
        projector_kernel = method.projector_kernel,
        reconstruction_mask = reconstruction_mask,
    )
end

function _cgls_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::ParallelVec2DGeometry,
    method::CGLSMethod;
    reconstruction_mask = nothing,
)
    LowLevel.cgls2d_parallel_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        projector_kernel = method.projector_kernel,
        reconstruction_mask = reconstruction_mask,
    )
end

function _cgls_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::FanFlat2DGeometry,
    method::CGLSMethod;
    reconstruction_mask = nothing,
)
    LowLevel.cgls2d_fanflat_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        origin_source_distance = geometry.origin_source_distance,
        origin_detector_distance = geometry.origin_detector_distance,
        iterations = method.iterations,
        projector_kernel = method.projector_kernel,
        reconstruction_mask = reconstruction_mask,
    )
end

function _cgls_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::FanFlatVec2DGeometry,
    method::CGLSMethod;
    reconstruction_mask = nothing,
)
    LowLevel.cgls2d_fanflat_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        projector_kernel = method.projector_kernel,
        reconstruction_mask = reconstruction_mask,
    )
end

function _cgls_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::SparseMatrix2DGeometry,
    method::CGLSMethod;
    reconstruction_mask = nothing,
)
    LowLevel.cgls2d_sparse_matrix_f32!(
        volume,
        sinogram,
        geometry.row_starts,
        geometry.col_indices,
        geometry.values;
        angle_count = geometry.angle_count,
        detector_count = geometry.detector_count,
        nnz = length(geometry.values),
        iterations = method.iterations,
        reconstruction_mask = reconstruction_mask,
    )
end

function _art_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::Parallel2DGeometry,
    method::ARTMethod;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
)
    LowLevel.art2d_parallel_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        iterations = method.iterations,
        relaxation = method.relaxation,
        projector_kernel = method.projector_kernel,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
    )
end

function _art_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::ParallelVec2DGeometry,
    method::ARTMethod;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
)
    LowLevel.art2d_parallel_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        relaxation = method.relaxation,
        projector_kernel = method.projector_kernel,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
    )
end

function _art_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::FanFlat2DGeometry,
    method::ARTMethod;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
)
    LowLevel.art2d_fanflat_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_width = geometry.detector_width,
        origin_source_distance = geometry.origin_source_distance,
        origin_detector_distance = geometry.origin_detector_distance,
        iterations = method.iterations,
        relaxation = method.relaxation,
        projector_kernel = method.projector_kernel,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
    )
end

function _art_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::FanFlatVec2DGeometry,
    method::ARTMethod;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
)
    LowLevel.art2d_fanflat_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        relaxation = method.relaxation,
        projector_kernel = method.projector_kernel,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
    )
end

function _art_cpu!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::SparseMatrix2DGeometry,
    method::ARTMethod;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
)
    LowLevel.art2d_sparse_matrix_f32!(
        volume,
        sinogram,
        geometry.row_starts,
        geometry.col_indices,
        geometry.values;
        angle_count = geometry.angle_count,
        detector_count = geometry.detector_count,
        nnz = length(geometry.values),
        iterations = method.iterations,
        relaxation = method.relaxation,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
    )
end

function reconstruct!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::AbstractGeometry2D,
    method::SIRTMethod;
    backend::AbstractBackend = cuda(),
    options::Union{Nothing,ReconstructionOptions2D} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    _check_volume(volume)
    _check_sinogram(sinogram, geometry)
    options = _merge_reconstruction_options(ReconstructionOptions2D, options, reconstruction_mask, sinogram_mask, residual_norm)
    reconstruction_mask = _check_reconstruction_mask(options.reconstruction_mask, size(volume))
    sinogram_mask = _check_sinogram_mask(options.sinogram_mask, size(sinogram))
    residual_norm = _check_residual_norm(options.residual_norm)
    vol_geom = _resolve_volume_geometry(volume, volume_geometry, window)
    LowLevel.with_volume_window(vol_geom) do
        if backend isa CPUBackend
            method.detector_supersampling == 1 || throw(ArgumentError("2D CPU SIRT does not support detector_supersampling"))
            method.pixel_supersampling == 1 || throw(ArgumentError("2D CPU SIRT does not support pixel_supersampling"))
            options.residual_norm === nothing || throw(ArgumentError("2D CPU SIRT does not support residual_norm"))
            _sirt_cpu!(volume, sinogram, geometry, method; reconstruction_mask = reconstruction_mask, sinogram_mask = sinogram_mask)
        elseif backend isa CUDABackend
            method.projector_kernel == :line || throw(ArgumentError("CUDA SIRT only supports default projector_kernel :line"))
            _sirt_cuda!(
                volume,
                sinogram,
                geometry,
                method,
                backend;
                reconstruction_mask = reconstruction_mask,
                sinogram_mask = sinogram_mask,
                residual_norm = residual_norm,
            )
        else
            throw(ArgumentError("unsupported backend type $(typeof(backend))"))
        end
    end
    return volume
end

function reconstruct!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::AbstractGeometry2D,
    method::SARTMethod;
    backend::AbstractBackend = cuda(),
    options::Union{Nothing,ReconstructionOptions2D} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    _check_volume(volume)
    _check_sinogram(sinogram, geometry)
    options = _merge_reconstruction_options(ReconstructionOptions2D, options, reconstruction_mask, sinogram_mask, residual_norm)
    options.sinogram_mask === nothing || throw(ArgumentError("SART does not support sinogram_mask"))
    reconstruction_mask = _check_reconstruction_mask(options.reconstruction_mask, size(volume))
    residual_norm = _check_residual_norm(options.residual_norm)
    vol_geom = _resolve_volume_geometry(volume, volume_geometry, window)
    LowLevel.with_volume_window(vol_geom) do
        if backend isa CPUBackend
            method.detector_supersampling == 1 || throw(ArgumentError("2D CPU SART does not support detector_supersampling"))
            method.pixel_supersampling == 1 || throw(ArgumentError("2D CPU SART does not support pixel_supersampling"))
            options.residual_norm === nothing || throw(ArgumentError("2D CPU SART does not support residual_norm"))
            _sart_cpu!(volume, sinogram, geometry, method; reconstruction_mask = reconstruction_mask)
        elseif backend isa CUDABackend
            method.projector_kernel == :line || throw(ArgumentError("CUDA SART only supports default projector_kernel :line"))
            _sart_cuda!(volume, sinogram, geometry, method, backend; reconstruction_mask = reconstruction_mask, residual_norm = residual_norm)
        else
            throw(ArgumentError("unsupported backend type $(typeof(backend))"))
        end
    end
    return volume
end

function reconstruct!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::AbstractGeometry2D,
    method::CGLSMethod;
    backend::AbstractBackend = cuda(),
    options::Union{Nothing,ReconstructionOptions2D} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    _check_volume(volume)
    _check_sinogram(sinogram, geometry)
    options = _merge_reconstruction_options(ReconstructionOptions2D, options, reconstruction_mask, sinogram_mask, residual_norm)
    options.sinogram_mask === nothing || throw(ArgumentError("CGLS does not support sinogram_mask"))
    reconstruction_mask = _check_reconstruction_mask(options.reconstruction_mask, size(volume))
    residual_norm = _check_residual_norm(options.residual_norm)
    vol_geom = _resolve_volume_geometry(volume, volume_geometry, window)
    LowLevel.with_volume_window(vol_geom) do
        if backend isa CPUBackend
            method.detector_supersampling == 1 || throw(ArgumentError("2D CPU CGLS does not support detector_supersampling"))
            method.pixel_supersampling == 1 || throw(ArgumentError("2D CPU CGLS does not support pixel_supersampling"))
            options.residual_norm === nothing || throw(ArgumentError("2D CPU CGLS does not support residual_norm"))
            _cgls_cpu!(volume, sinogram, geometry, method; reconstruction_mask = reconstruction_mask)
        elseif backend isa CUDABackend
            method.projector_kernel == :line || throw(ArgumentError("CUDA CGLS only supports default projector_kernel :line"))
            _cgls_cuda!(volume, sinogram, geometry, method, backend; reconstruction_mask = reconstruction_mask, residual_norm = residual_norm)
        else
            throw(ArgumentError("unsupported backend type $(typeof(backend))"))
        end
    end
    return volume
end

function reconstruct!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::AbstractGeometry2D,
    method::ARTMethod;
    backend::AbstractBackend = cpu(),
    options::Union{Nothing,ReconstructionOptions2D} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    _check_volume(volume)
    _check_sinogram(sinogram, geometry)
    options = _merge_reconstruction_options(ReconstructionOptions2D, options, reconstruction_mask, sinogram_mask, residual_norm)
    reconstruction_mask = _check_reconstruction_mask(options.reconstruction_mask, size(volume))
    sinogram_mask = _check_sinogram_mask(options.sinogram_mask, size(sinogram))
    residual_norm = _check_residual_norm(options.residual_norm)
    vol_geom = _resolve_volume_geometry(volume, volume_geometry, window)
    LowLevel.with_volume_window(vol_geom) do
        if backend isa CPUBackend
            options.residual_norm === nothing || throw(ArgumentError("2D CPU ART does not support residual_norm"))
            _art_cpu!(volume, sinogram, geometry, method; reconstruction_mask = reconstruction_mask, sinogram_mask = sinogram_mask)
        else
            throw(ArgumentError("ART is only available on CPU backend"))
        end
    end
    return volume
end

function reconstruct!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::Parallel2DGeometry,
    method::EMMethod;
    backend::AbstractBackend = cuda(),
    options::Union{Nothing,ReconstructionOptions2D} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    _check_volume(volume)
    _check_sinogram(sinogram, geometry)
    options = _merge_reconstruction_options(ReconstructionOptions2D, options, reconstruction_mask, sinogram_mask, residual_norm)
    options.reconstruction_mask === nothing || throw(ArgumentError("EM does not support reconstruction_mask"))
    options.sinogram_mask === nothing || throw(ArgumentError("EM does not support sinogram_mask"))
    residual_norm = _check_residual_norm(options.residual_norm)
    backend isa CUDABackend || throw(ArgumentError("EM is only available on CUDA in the current high-level API"))
    vol_geom = _resolve_volume_geometry(volume, volume_geometry, window)
    LowLevel.with_volume_window(vol_geom) do
        _em_cuda!(volume, sinogram, geometry, method, backend; residual_norm = residual_norm)
    end
    return volume
end

function reconstruct!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::AbstractGeometry2D,
    method::EMMethod;
    backend::AbstractBackend = cuda(),
    options::Union{Nothing,ReconstructionOptions2D} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    throw(ArgumentError("EM is only available for Parallel2DGeometry"))
end

"""
    reconstruct(sinogram, geometry, method; volume_size, ...)

2D reconstruction. Returns a new `Array{Float32,2}`.

# Arguments
- `sinogram`: 2D array of the sinogram to reconstruct. Size must be `(detector_count, angle_count)`.
- `geometry`: 2D geometry object.
- `method`: Reconstruction method.
- `volume_size`: Tuple `(nx, ny)` specifying the size of the output volume.
- `backend`: Computational backend.
- `options`: Optional `ReconstructionOptions2D`.
- `reconstruction_mask`: Optional 2D mask.
- `sinogram_mask`: Optional 2D mask.
- `residual_norm`: Optional `Ref{Float32}`.

# Returns
- `volume`: 2D array of size `volume_size`.

# Examples
```julia
volume = reconstruct(sinogram, geom, SIRTMethod(iterations=100); volume_size=(256, 256))
```
"""
function reconstruct(
    sinogram::Array{Float32,2},
    geometry::AbstractGeometry2D,
    method::AbstractMethod;
    volume_size::Union{Nothing,Tuple{Int,Int}} = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
    backend::AbstractBackend = (method isa FBPMethod || method isa ARTMethod) ? cpu() : cuda(),
    options::Union{Nothing,ReconstructionOptions2D} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
)
    dims = if volume_geometry !== nothing
        size(volume_geometry)
    elseif volume_size !== nothing
        volume_size
    else
        throw(ArgumentError("either `volume_size` or `volume_geometry` must be specified"))
    end
    volume = zeros(Float32, dims...)
    return reconstruct!(
        volume,
        sinogram,
        geometry,
        method;
        backend = backend,
        options = options,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        residual_norm = residual_norm,
        volume_geometry = volume_geometry,
        window = window,
    )
end

function reconstruct(
    sinogram::AbstractMatrix{<:Real},
    geometry::AbstractGeometry2D,
    method::AbstractMethod;
    volume_size::Union{Nothing,Tuple{Int,Int}} = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
    backend::AbstractBackend = (method isa FBPMethod || method isa ARTMethod) ? cpu() : cuda(),
    options::Union{Nothing,ReconstructionOptions2D} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
)
    return reconstruct(
        Matrix{Float32}(sinogram),
        geometry,
        method;
        volume_size = volume_size,
        volume_geometry = volume_geometry,
        window = window,
        backend = backend,
        options = options,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        residual_norm = residual_norm,
    )
end

function reconstruct!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::AbstractGeometry2D,
    vol_geom::VolumeGeometry2D,
    method::AbstractMethod;
    kwargs...,
)
    return reconstruct!(volume, sinogram, geometry, method; volume_geometry = vol_geom, kwargs...)
end

function reconstruct!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    geometry::AbstractGeometry2D,
    method::AbstractMethod,
    vol_geom::VolumeGeometry2D;
    kwargs...,
)
    return reconstruct!(volume, sinogram, geometry, method; volume_geometry = vol_geom, kwargs...)
end

function reconstruct(
    sinogram::Array{Float32,2},
    geometry::AbstractGeometry2D,
    vol_geom::VolumeGeometry2D,
    method::AbstractMethod;
    kwargs...,
)
    return reconstruct(sinogram, geometry, method; volume_geometry = vol_geom, kwargs...)
end

function reconstruct(
    sinogram::Array{Float32,2},
    geometry::AbstractGeometry2D,
    method::AbstractMethod,
    vol_geom::VolumeGeometry2D;
    kwargs...,
)
    return reconstruct(sinogram, geometry, method; volume_geometry = vol_geom, kwargs...)
end

function reconstruct(
    sinogram::Array{Float32,2},
    geometry::SparseMatrix2DGeometry,
    method::AbstractMethod;
    volume_size::Union{Nothing,Tuple{Int,Int}} = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
    backend::AbstractBackend = cpu(),
    options::Union{Nothing,ReconstructionOptions2D} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
)
    dims = if volume_geometry !== nothing
        size(volume_geometry)
    elseif volume_size !== nothing
        volume_size
    else
        (geometry.volume_width, geometry.volume_height)
    end
    volume = zeros(Float32, dims...)
    return reconstruct!(
        volume,
        sinogram,
        geometry,
        method;
        backend = backend,
        options = options,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        residual_norm = residual_norm,
        volume_geometry = volume_geometry,
        window = window,
    )
end

function reconstruct(
    sinogram::AbstractMatrix{<:Real},
    geometry::SparseMatrix2DGeometry,
    method::AbstractMethod;
    volume_size::Union{Nothing,Tuple{Int,Int}} = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
    backend::AbstractBackend = cpu(),
    options::Union{Nothing,ReconstructionOptions2D} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
)
    return reconstruct(
        Matrix{Float32}(sinogram),
        geometry,
        method;
        volume_size = volume_size,
        volume_geometry = volume_geometry,
        window = window,
        backend = backend,
        options = options,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        residual_norm = residual_norm,
    )
end

function _sirt3d_cuda!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geometry::Parallel3DGeometry,
    method::SIRT3DMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_sirt3d_parallel_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_spacing_u = geometry.detector_spacing_u,
        detector_spacing_v = geometry.detector_spacing_v,
        iterations = method.iterations,
        relaxation = method.relaxation,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        residual_norm = residual_norm,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
        voxel_supersampling = method.voxel_supersampling,
        detector_supersampling = method.detector_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _sirt3d_cuda!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geometry::ParallelVec3DGeometry,
    method::SIRT3DMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_sirt3d_parallel_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        relaxation = method.relaxation,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        residual_norm = residual_norm,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
        voxel_supersampling = method.voxel_supersampling,
        detector_supersampling = method.detector_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _sirt3d_cuda!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geometry::Cone3DGeometry,
    method::SIRT3DMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_sirt3d_cone_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_spacing_u = geometry.detector_spacing_u,
        detector_spacing_v = geometry.detector_spacing_v,
        origin_source_distance = geometry.origin_source_distance,
        origin_detector_distance = geometry.origin_detector_distance,
        iterations = method.iterations,
        relaxation = method.relaxation,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        residual_norm = residual_norm,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
        voxel_supersampling = method.voxel_supersampling,
        detector_supersampling = method.detector_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _sirt3d_cuda!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geometry::ConeVec3DGeometry,
    method::SIRT3DMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_sirt3d_cone_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        relaxation = method.relaxation,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        residual_norm = residual_norm,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
        voxel_supersampling = method.voxel_supersampling,
        detector_supersampling = method.detector_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _sirt3d_cuda!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geometry::CylConeVec3DGeometry,
    method::SIRT3DMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_sirt3d_cyl_cone_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        relaxation = method.relaxation,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        residual_norm = residual_norm,
        min_constraint = method.min_constraint,
        max_constraint = method.max_constraint,
        voxel_supersampling = method.voxel_supersampling,
        detector_supersampling = method.detector_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _cgls3d_cuda!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geometry::Parallel3DGeometry,
    method::CGLS3DMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_cgls3d_parallel_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_spacing_u = geometry.detector_spacing_u,
        detector_spacing_v = geometry.detector_spacing_v,
        iterations = method.iterations,
        reconstruction_mask = reconstruction_mask,
        residual_norm = residual_norm,
        voxel_supersampling = method.voxel_supersampling,
        detector_supersampling = method.detector_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _cgls3d_cuda!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geometry::ParallelVec3DGeometry,
    method::CGLS3DMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_cgls3d_parallel_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        reconstruction_mask = reconstruction_mask,
        residual_norm = residual_norm,
        voxel_supersampling = method.voxel_supersampling,
        detector_supersampling = method.detector_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _cgls3d_cuda!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geometry::Cone3DGeometry,
    method::CGLS3DMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_cgls3d_cone_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_spacing_u = geometry.detector_spacing_u,
        detector_spacing_v = geometry.detector_spacing_v,
        origin_source_distance = geometry.origin_source_distance,
        origin_detector_distance = geometry.origin_detector_distance,
        iterations = method.iterations,
        reconstruction_mask = reconstruction_mask,
        residual_norm = residual_norm,
        voxel_supersampling = method.voxel_supersampling,
        detector_supersampling = method.detector_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _cgls3d_cuda!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geometry::ConeVec3DGeometry,
    method::CGLS3DMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_cgls3d_cone_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        reconstruction_mask = reconstruction_mask,
        residual_norm = residual_norm,
        voxel_supersampling = method.voxel_supersampling,
        detector_supersampling = method.detector_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _cgls3d_cuda!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geometry::CylConeVec3DGeometry,
    method::CGLS3DMethod,
    backend::CUDABackend;
    reconstruction_mask = nothing,
    residual_norm = nothing,
)
    LowLevel.cuda_cgls3d_cyl_cone_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        iterations = method.iterations,
        reconstruction_mask = reconstruction_mask,
        residual_norm = residual_norm,
        voxel_supersampling = method.voxel_supersampling,
        detector_supersampling = method.detector_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _fdk3d_cuda!(volume::Array{Float32,3}, sinogram::Array{Float32,3}, geometry::Cone3DGeometry, method::FDK3DMethod, backend::CUDABackend)
    LowLevel.cuda_fdk3d_cone_f32!(
        volume,
        sinogram,
        geometry.angles;
        detector_spacing_u = geometry.detector_spacing_u,
        detector_spacing_v = geometry.detector_spacing_v,
        origin_source_distance = geometry.origin_source_distance,
        origin_detector_distance = geometry.origin_detector_distance,
        filter_type = method.filter_type,
        custom_filter = method.custom_filter,
        filter_d = method.filter_d,
        filter_parameter = method.filter_parameter,
        short_scan = method.short_scan,
        voxel_supersampling = method.voxel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

function _fdk3d_cuda!(volume::Array{Float32,3}, sinogram::Array{Float32,3}, geometry::ConeVec3DGeometry, method::FDK3DMethod, backend::CUDABackend)
    LowLevel.cuda_fdk3d_cone_vec_f32!(
        volume,
        sinogram,
        geometry.vectors;
        filter_type = method.filter_type,
        custom_filter = method.custom_filter,
        filter_d = method.filter_d,
        filter_parameter = method.filter_parameter,
        short_scan = method.short_scan,
        voxel_supersampling = method.voxel_supersampling,
        gpu_index = backend.gpu_index,
    )
end

"""
    reconstruct!(volume::Array{Float32,3}, sinogram::Array{Float32,3}, geometry::AbstractGeometry3D, method::AbstractMethod; ...)

In-place 3D reconstruction.

# Arguments
- `volume`: 3D array to store the reconstruction result. Size is typically `(nx, ny, nz)`.
- `sinogram`: 3D array of the sinogram to reconstruct. Size must be `(detector_count_u, angle_count, detector_count_v)`.
- `geometry`: 3D geometry object.
- `method`: Reconstruction method (e.g., `SIRT3DMethod`, `CGLS3DMethod`, `FDK3DMethod`).
- `backend`: Computational backend (only `cuda()` is supported for 3D in the current high-level API). Defaults to `cuda()`.
- `options`: Optional `ReconstructionOptions3D`.
- `reconstruction_mask`: Optional 3D mask for the reconstruction volume.
- `sinogram_mask`: Optional 3D mask for the sinogram.
- `residual_norm`: Optional `Ref{Float32}` to store the final residual norm (for iterative methods).

# Returns
- `volume`: The updated 3D volume array (mutated in-place).

# Examples
```julia
volume = zeros(Float32, 256, 256, 256)
method = FDK3DMethod()
reconstruct!(volume, sinogram, geom, method)
```
"""
function reconstruct!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geometry::AbstractGeometry3D,
    method::SIRT3DMethod;
    backend::AbstractBackend = cuda(),
    options::Union{Nothing,ReconstructionOptions3D} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    _check_volume(volume)
    _check_sinogram(sinogram, geometry)
    options = _merge_reconstruction_options(ReconstructionOptions3D, options, reconstruction_mask, sinogram_mask, residual_norm)
    reconstruction_mask = _check_reconstruction_mask(options.reconstruction_mask, size(volume))
    sinogram_mask = _check_sinogram_mask(options.sinogram_mask, size(sinogram))
    residual_norm = _check_residual_norm(options.residual_norm)
    backend isa CUDABackend || throw(ArgumentError("SIRT3D is only available on CUDA in the current high-level API"))
    vol_geom = _resolve_volume_geometry(volume, volume_geometry, window)
    LowLevel.with_volume_window(vol_geom) do
        _sirt3d_cuda!(
            volume,
            sinogram,
            geometry,
            method,
            backend;
            reconstruction_mask = reconstruction_mask,
            sinogram_mask = sinogram_mask,
            residual_norm = residual_norm,
        )
    end
    return volume
end

function reconstruct!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geometry::AbstractGeometry3D,
    method::CGLS3DMethod;
    backend::AbstractBackend = cuda(),
    options::Union{Nothing,ReconstructionOptions3D} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    _check_volume(volume)
    _check_sinogram(sinogram, geometry)
    options = _merge_reconstruction_options(ReconstructionOptions3D, options, reconstruction_mask, sinogram_mask, residual_norm)
    options.sinogram_mask === nothing || throw(ArgumentError("CGLS3D does not support sinogram_mask"))
    reconstruction_mask = _check_reconstruction_mask(options.reconstruction_mask, size(volume))
    residual_norm = _check_residual_norm(options.residual_norm)
    backend isa CUDABackend || throw(ArgumentError("CGLS3D is only available on CUDA in the current high-level API"))
    vol_geom = _resolve_volume_geometry(volume, volume_geometry, window)
    LowLevel.with_volume_window(vol_geom) do
        _cgls3d_cuda!(volume, sinogram, geometry, method, backend; reconstruction_mask = reconstruction_mask, residual_norm = residual_norm)
    end
    return volume
end

function reconstruct!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geometry::Union{Cone3DGeometry,ConeVec3DGeometry},
    method::FDK3DMethod;
    backend::AbstractBackend = cuda(),
    options::Union{Nothing,ReconstructionOptions3D} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    _check_volume(volume)
    _check_sinogram(sinogram, geometry)
    options = _merge_reconstruction_options(ReconstructionOptions3D, options, reconstruction_mask, sinogram_mask, residual_norm)
    options.reconstruction_mask === nothing || throw(ArgumentError("FDK3D does not support reconstruction_mask"))
    options.sinogram_mask === nothing || throw(ArgumentError("FDK3D does not support sinogram_mask"))
    options.residual_norm === nothing || throw(ArgumentError("FDK3D does not support residual_norm"))
    backend isa CUDABackend || throw(ArgumentError("FDK3D is only available on CUDA in the current high-level API"))
    vol_geom = _resolve_volume_geometry(volume, volume_geometry, window)
    LowLevel.with_volume_window(vol_geom) do
        _fdk3d_cuda!(volume, sinogram, geometry, method, backend)
    end
    return volume
end

function reconstruct!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geometry::AbstractGeometry3D,
    method::FDK3DMethod;
    backend::AbstractBackend = cuda(),
    options::Union{Nothing,ReconstructionOptions3D} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
)
    throw(ArgumentError("FDK3D is only available for Cone3DGeometry and ConeVec3DGeometry"))
end

function reconstruct!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geometry::AbstractGeometry3D,
    vol_geom::VolumeGeometry3D,
    method::AbstractMethod;
    kwargs...,
)
    return reconstruct!(volume, sinogram, geometry, method; volume_geometry = vol_geom, kwargs...)
end

function reconstruct!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geometry::AbstractGeometry3D,
    method::AbstractMethod,
    vol_geom::VolumeGeometry3D;
    kwargs...,
)
    return reconstruct!(volume, sinogram, geometry, method; volume_geometry = vol_geom, kwargs...)
end

"""
    reconstruct(sinogram, geometry, method; volume_size, ...)

3D reconstruction. Returns a new `Array{Float32,3}`.

# Arguments
- `sinogram`: 3D array of the sinogram to reconstruct. Size must be `(detector_count_u, angle_count, detector_count_v)`.
- `geometry`: 3D geometry object.
- `method`: Reconstruction method.
- `volume_size`: Optional tuple `(nx, ny, nz)` specifying the size of the output volume.
- `volume_geometry`: Optional `VolumeGeometry3D` object.
- `window`: Optional 6-element collection `(min_x, max_x, min_y, max_y, min_z, max_z)`.
- `backend`: Computational backend.
- `options`: Optional `ReconstructionOptions3D`.
- `reconstruction_mask`: Optional 3D mask.
- `sinogram_mask`: Optional 3D mask.
- `residual_norm`: Optional `Ref{Float32}`.

# Returns
- `volume`: 3D array of size `volume_size`.

# Examples
```julia
volume = reconstruct(sinogram, geom, SIRT3DMethod(iterations=100); volume_size=(256, 256, 256))
```
"""
function reconstruct(
    sinogram::Array{Float32,3},
    geometry::AbstractGeometry3D,
    method::AbstractMethod;
    volume_size::Union{Nothing,Tuple{Int,Int,Int}} = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
    backend::AbstractBackend = cuda(),
    options::Union{Nothing,ReconstructionOptions3D} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
)
    dims = if volume_geometry !== nothing
        size(volume_geometry)
    elseif volume_size !== nothing
        volume_size
    else
        throw(ArgumentError("either `volume_size` or `volume_geometry` must be specified"))
    end
    volume = zeros(Float32, dims...)
    return reconstruct!(
        volume,
        sinogram,
        geometry,
        method;
        backend = backend,
        options = options,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        residual_norm = residual_norm,
        volume_geometry = volume_geometry,
        window = window,
    )
end

function reconstruct(
    sinogram::AbstractArray{<:Real,3},
    geometry::AbstractGeometry3D,
    method::AbstractMethod;
    volume_size::Union{Nothing,Tuple{Int,Int,Int}} = nothing,
    volume_geometry::Union{Nothing,AbstractVolumeGeometry} = nothing,
    window = nothing,
    backend::AbstractBackend = cuda(),
    options::Union{Nothing,ReconstructionOptions3D} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
)
    return reconstruct(
        Array{Float32,3}(sinogram),
        geometry,
        method;
        volume_size = volume_size,
        volume_geometry = volume_geometry,
        window = window,
        backend = backend,
        options = options,
        reconstruction_mask = reconstruction_mask,
        sinogram_mask = sinogram_mask,
        residual_norm = residual_norm,
    )
end

function reconstruct(
    sinogram::Array{Float32,3},
    geometry::AbstractGeometry3D,
    vol_geom::VolumeGeometry3D,
    method::AbstractMethod;
    kwargs...,
)
    return reconstruct(sinogram, geometry, method; volume_geometry = vol_geom, kwargs...)
end

function reconstruct(
    sinogram::Array{Float32,3},
    geometry::AbstractGeometry3D,
    method::AbstractMethod,
    vol_geom::VolumeGeometry3D;
    kwargs...,
)
    return reconstruct(sinogram, geometry, method; volume_geometry = vol_geom, kwargs...)
end

