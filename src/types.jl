"""
    AbstractBackend

Abstract supertype for computational backends in `AstraToolbox.jl`.

Subtypes include [`CPUBackend`](@ref) for CPU execution and [`CUDABackend`](@ref) for GPU-accelerated execution.
"""
abstract type AbstractBackend end

"""
CPU computational backend.
"""
struct CPUBackend <: AbstractBackend end

"""
CUDA (GPU) computational backend.

Fields:
- `gpu_index::Int`: Index of the GPU to use. Default is -1 (system default).
"""
struct CUDABackend <: AbstractBackend
    gpu_index::Int
end

"""
    cpu() -> CPUBackend

Create a CPU computational backend instance for CPU-based projection and reconstruction.

# Returns
- `CPUBackend`: Computational backend targeting the CPU.

# Examples
```julia
backend = cpu()
```
"""
cpu() = CPUBackend()

"""
    cuda(; gpu_index::Integer = -1) -> CUDABackend

Create a CUDA GPU computational backend instance for GPU-accelerated projection and reconstruction.

# Arguments
- `gpu_index`: 0-based device index of the GPU to use. Default is `-1` (currently active GPU).

# Returns
- `CUDABackend`: Computational backend targeting CUDA GPU.

# Examples
```julia
backend = cuda()
backend_gpu1 = cuda(gpu_index=1)
```
"""
cuda(; gpu_index::Integer = -1) = CUDABackend(Int(gpu_index))

"""
Abstract supertype for all geometries.
"""
abstract type AbstractGeometry end
"""
Abstract supertype for all 2D geometries.
"""
abstract type AbstractGeometry2D <: AbstractGeometry end
"""
Abstract supertype for all 3D geometries.
"""
abstract type AbstractGeometry3D <: AbstractGeometry end
"""
Abstract supertype for all volume geometries.
"""
abstract type AbstractVolumeGeometry <: AbstractGeometry end

"""
2D Volume geometry defining grid dimensions and spatial bounding box (window).

Fields:
- `width::Int`: Number of columns in volume grid (X dimension).
- `height::Int`: Number of rows in volume grid (Y dimension).
- `window_min_x::Float32`: Minimum X coordinate.
- `window_min_y::Float32`: Minimum Y coordinate.
- `window_max_x::Float32`: Maximum X coordinate.
- `window_max_y::Float32`: Maximum Y coordinate.
"""
struct VolumeGeometry2D <: AbstractVolumeGeometry
    width::Int
    height::Int
    window_min_x::Float32
    window_min_y::Float32
    window_max_x::Float32
    window_max_y::Float32
end

"""
3D Volume geometry defining grid dimensions and spatial bounding box (window).

Fields:
- `width::Int`: Number of columns in volume grid (X dimension).
- `height::Int`: Number of rows in volume grid (Y dimension).
- `depth::Int`: Number of slices in volume grid (Z dimension).
- `window_min_x::Float32`: Minimum X coordinate.
- `window_min_y::Float32`: Minimum Y coordinate.
- `window_min_z::Float32`: Minimum Z coordinate.
- `window_max_x::Float32`: Maximum X coordinate.
- `window_max_y::Float32`: Maximum Y coordinate.
- `window_max_z::Float32`: Maximum Z coordinate.
"""
struct VolumeGeometry3D <: AbstractVolumeGeometry
    width::Int
    height::Int
    depth::Int
    window_min_x::Float32
    window_min_y::Float32
    window_min_z::Float32
    window_max_x::Float32
    window_max_y::Float32
    window_max_z::Float32
end

"""
Abstract supertype for all projection options.
"""
abstract type AbstractProjectionOptions end

"""
Projection options for 2D.
"""
struct ProjectionOptions2D <: AbstractProjectionOptions
    detector_supersampling::Int
    projector_kernel::Symbol
end

"""
Projection options for 3D.
"""
struct ProjectionOptions3D <: AbstractProjectionOptions
    detector_supersampling::Int
end

"""
Abstract supertype for all backprojection options.
"""
abstract type AbstractBackprojectionOptions end

"""
Backprojection options for 2D.
"""
struct BackprojectionOptions2D <: AbstractBackprojectionOptions
    detector_supersampling::Int
    pixel_supersampling::Int
    projector_kernel::Symbol
end

"""
Backprojection options for 3D.
"""
struct BackprojectionOptions3D <: AbstractBackprojectionOptions
    voxel_supersampling::Int
end

"""
2D Parallel-beam geometry.

Fields:
- `angles::Vector{Float32}`: Rotation angles in radians.
- `detector_count::Int`: Number of detector pixels.
- `detector_width::Float32`: Width of a single detector pixel.
"""
struct Parallel2DGeometry <: AbstractGeometry2D
    angles::Vector{Float32}
    detector_count::Int
    detector_width::Float32
end

"""
2D Parallel-beam vector geometry.

Fields:
- `vectors::Matrix{Float32}`: 6xN matrix of vectors defining source and detector positions.
- `detector_count::Int`: Number of detector pixels.
"""
struct ParallelVec2DGeometry <: AbstractGeometry2D
    vectors::Matrix{Float32}
    detector_count::Int
end

"""
2D Fan-beam flat-detector geometry.

Fields:
- `angles::Vector{Float32}`: Rotation angles in radians.
- `detector_count::Int`: Number of detector pixels.
- `detector_width::Float32`: Width of a single detector pixel.
- `origin_source_distance::Float32`: Distance from origin to source.
- `origin_detector_distance::Float32`: Distance from origin to detector center.
"""
struct FanFlat2DGeometry <: AbstractGeometry2D
    angles::Vector{Float32}
    detector_count::Int
    detector_width::Float32
    origin_source_distance::Float32
    origin_detector_distance::Float32
end

"""
2D Fan-beam flat-detector vector geometry.

Fields:
- `vectors::Matrix{Float32}`: 6xN matrix of vectors defining source and detector positions.
- `detector_count::Int`: Number of detector pixels.
"""
struct FanFlatVec2DGeometry <: AbstractGeometry2D
    vectors::Matrix{Float32}
    detector_count::Int
end

"""
2D Sparse matrix projection geometry.

Fields:
- `angle_count::Int`: Number of projection angles.
- `detector_count::Int`: Number of detector pixels per projection.
- `volume_width::Int`: Width of reconstruction volume (grid columns).
- `volume_height::Int`: Height of reconstruction volume (grid rows).
- `matrix::SparseMatrixCSC{Float32, Int}`: Sparse projection matrix (M x N) where M = angle_count * detector_count, N = volume_width * volume_height.
- `row_starts::Vector{UInt64}`: CSR row start offsets (0-based).
- `col_indices::Vector{UInt32}`: CSR column indices (0-based).
- `values::Vector{Float32}`: CSR values.
"""
struct SparseMatrix2DGeometry <: AbstractGeometry2D
    angle_count::Int
    detector_count::Int
    volume_width::Int
    volume_height::Int
    matrix::SparseMatrixCSC{Float32, Int}
    row_starts::Vector{UInt64}
    col_indices::Vector{UInt32}
    values::Vector{Float32}
end

const Sparse2DGeometry = SparseMatrix2DGeometry

"""
3D Parallel-beam geometry.

Fields:
- `angles::Vector{Float32}`: Rotation angles in radians.
- `detector_count_u::Int`: Number of detector pixels (horizontal).
- `detector_count_v::Int`: Number of detector pixels (vertical).
- `detector_spacing_u::Float32`: Horizontal pixel spacing.
- `detector_spacing_v::Float32`: Vertical pixel spacing.
"""
struct Parallel3DGeometry <: AbstractGeometry3D
    angles::Vector{Float32}
    detector_count_u::Int
    detector_count_v::Int
    detector_spacing_u::Float32
    detector_spacing_v::Float32
end

"""
3D Parallel-beam vector geometry.

Fields:
- `vectors::Matrix{Float32}`: 12xN matrix defining geometry.
- `detector_count_u::Int`: Horizontal detector pixel count.
- `detector_count_v::Int`: Vertical detector pixel count.
"""
struct ParallelVec3DGeometry <: AbstractGeometry3D
    vectors::Matrix{Float32}
    detector_count_u::Int
    detector_count_v::Int
end

"""
3D Cone-beam geometry.

Fields:
- `angles::Vector{Float32}`: Rotation angles in radians.
- `detector_count_u::Int`: Horizontal detector pixel count.
- `detector_count_v::Int`: Vertical detector pixel count.
- `detector_spacing_u::Float32`: Horizontal pixel spacing.
- `detector_spacing_v::Float32`: Vertical pixel spacing.
- `origin_source_distance::Float32`: Distance from origin to source.
- `origin_detector_distance::Float32`: Distance from origin to detector.
"""
struct Cone3DGeometry <: AbstractGeometry3D
    angles::Vector{Float32}
    detector_count_u::Int
    detector_count_v::Int
    detector_spacing_u::Float32
    detector_spacing_v::Float32
    origin_source_distance::Float32
    origin_detector_distance::Float32
end

"""
3D Cone-beam vector geometry.

Fields:
- `vectors::Matrix{Float32}`: 12xN matrix defining geometry.
- `detector_count_u::Int`: Horizontal detector pixel count.
- `detector_count_v::Int`: Vertical detector pixel count.
"""
struct ConeVec3DGeometry <: AbstractGeometry3D
    vectors::Matrix{Float32}
    detector_count_u::Int
    detector_count_v::Int
end

"""
3D Cone-beam vector geometry for cylindrical detectors.
"""
struct CylConeVec3DGeometry <: AbstractGeometry3D
    vectors::Matrix{Float32}
    detector_count_u::Int
    detector_count_v::Int
end

"""
Abstract supertype for all reconstruction methods.
"""
abstract type AbstractMethod end

"""
Filtered Back-Projection (FBP) method.

Parameters:
- `filter_type::Int`: Filter type (e.g., `FILTER_RAMLAK`).
- `filter_d::Float32`: Filter frequency scaling.
- `filter_parameter::Float32`: Optional filter parameter.
- `custom_filter`: Optional custom frequency domain filter.
- `short_scan::Bool`: Whether to use Parker weighting for short scans.
"""
struct FBPMethod <: AbstractMethod
    filter_type::Int
    filter_d::Float32
    filter_parameter::Float32
    custom_filter::Union{Nothing,Vector{Float32},Matrix{Float32}}
    short_scan::Bool
end

"""
Simultaneous Iterative Reconstruction Technique (SIRT).

Parameters:
- `iterations::Int`: Number of iterations.
- `relaxation::Float32`: Relaxation parameter.
- `detector_supersampling::Int`: GPU-only detector supersampling.
- `pixel_supersampling::Int`: CPU-only pixel supersampling.
- `min_constraint`: Optional non-negativity or other lower bound.
- `max_constraint`: Optional upper bound.
- `projector_kernel::Symbol`: Projection kernel (`:line`, `:linear`, or `:strip`).
"""
struct SIRTMethod <: AbstractMethod
    iterations::Int
    relaxation::Float32
    detector_supersampling::Int
    pixel_supersampling::Int
    min_constraint::Union{Nothing,Float32}
    max_constraint::Union{Nothing,Float32}
    projector_kernel::Symbol
end

"""
Simultaneous Algebraic Reconstruction Technique (SART).

Parameters:
- `iterations::Int`: Number of iterations.
- `relaxation::Float32`: Relaxation parameter.
- `projection_order::Symbol`: `:random`, `:sequential`, or `:custom`.
- `projection_order_list`: List of projection indices for `:custom` order.
- `detector_supersampling::Int`: GPU-only detector supersampling.
- `pixel_supersampling::Int`: CPU-only pixel supersampling.
- `min_constraint`, `max_constraint`: Optional bounds.
- `projector_kernel::Symbol`: Projection kernel (`:line`, `:linear`, or `:strip`).
"""
struct SARTMethod <: AbstractMethod
    iterations::Int
    relaxation::Float32
    projection_order::Symbol
    projection_order_list::Union{Nothing,Vector{Int32}}
    detector_supersampling::Int
    pixel_supersampling::Int
    min_constraint::Union{Nothing,Float32}
    max_constraint::Union{Nothing,Float32}
    projector_kernel::Symbol
end

"""
Conjugate Gradient Least Squares (CGLS).

Parameters:
- `iterations::Int`: Number of iterations.
- `detector_supersampling::Int`: Detector supersampling level.
- `pixel_supersampling::Int`: Pixel supersampling level.
- `projector_kernel::Symbol`: Projection kernel (`:line`, `:linear`, or `:strip`).
"""
struct CGLSMethod <: AbstractMethod
    iterations::Int
    detector_supersampling::Int
    pixel_supersampling::Int
    projector_kernel::Symbol
end

"""
Algebraic Reconstruction Technique (ART).

Parameters:
- `iterations::Int`: Number of ray projection iterations.
- `relaxation::Float32`: Relaxation parameter.
- `min_constraint`: Optional non-negativity or other lower bound.
- `max_constraint`: Optional upper bound.
- `projector_kernel::Symbol`: Projection kernel (`:line`, `:linear`, or `:strip`).
"""
struct ARTMethod <: AbstractMethod
    iterations::Int
    relaxation::Float32
    min_constraint::Union{Nothing,Float32}
    max_constraint::Union{Nothing,Float32}
    projector_kernel::Symbol
end

"""
Expectation Maximization (EM).

Parameters:
- `iterations::Int`: Number of iterations.
- `detector_supersampling::Int`: Detector supersampling level.
- `pixel_supersampling::Int`: Pixel supersampling level.
"""
struct EMMethod <: AbstractMethod
    iterations::Int
    detector_supersampling::Int
    pixel_supersampling::Int
end

"""
3D SIRT method.
"""
struct SIRT3DMethod <: AbstractMethod
    iterations::Int
    relaxation::Float32
    voxel_supersampling::Int
    detector_supersampling::Int
    min_constraint::Union{Nothing,Float32}
    max_constraint::Union{Nothing,Float32}
end

"""
3D CGLS method.
"""
struct CGLS3DMethod <: AbstractMethod
    iterations::Int
    voxel_supersampling::Int
    detector_supersampling::Int
end

"""
Feldkamp-Davis-Kress (FDK) method for cone-beam reconstruction.
"""
struct FDK3DMethod <: AbstractMethod
    filter_type::Int
    filter_d::Float32
    filter_parameter::Float32
    custom_filter::Union{Nothing,Vector{Float32},Matrix{Float32}}
    short_scan::Bool
    voxel_supersampling::Int
end

"""
Abstract supertype for all reconstruction options.
"""
abstract type AbstractReconstructionOptions end

"""
Reconstruction options for 2D.

Fields:
- `reconstruction_mask::Union{Nothing,Array{Float32,2}}`: Optional mask for the reconstruction volume.
- `sinogram_mask::Union{Nothing,Array{Float32,2}}`: Optional mask for the sinogram.
- `residual_norm::Union{Nothing,Base.RefValue{Float32}}`: Optional reference to store the final residual norm.
"""
struct ReconstructionOptions2D <: AbstractReconstructionOptions
    reconstruction_mask::Union{Nothing,Array{Float32,2}}
    sinogram_mask::Union{Nothing,Array{Float32,2}}
    residual_norm::Union{Nothing,Base.RefValue{Float32}}
end

"""
Reconstruction options for 3D.

Fields:
- `reconstruction_mask::Union{Nothing,Array{Float32,3}}`: Optional mask for the reconstruction volume.
- `sinogram_mask::Union{Nothing,Array{Float32,3}}`: Optional mask for the sinogram.
- `residual_norm::Union{Nothing,Base.RefValue{Float32}}`: Optional reference to store the final residual norm.
"""
struct ReconstructionOptions3D <: AbstractReconstructionOptions
    reconstruction_mask::Union{Nothing,Array{Float32,3}}
    sinogram_mask::Union{Nothing,Array{Float32,3}}
    residual_norm::Union{Nothing,Base.RefValue{Float32}}
end

