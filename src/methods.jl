"""
    FBPMethod(; filter_type=FILTER_RAMLAK, filter_d=1.0, filter_parameter=-1.0, custom_filter=nothing, short_scan=false)

Create an FBP reconstruction method.

# Arguments
- `filter_type`: Filter type (e.g., `FILTER_RAMLAK`). Defaults to `FILTER_RAMLAK`.
- `filter_d`: Filter frequency scaling. Defaults to 1.0.
- `filter_parameter`: Optional filter parameter. Defaults to -1.0.
- `custom_filter`: Optional custom frequency domain filter.
- `short_scan`: Whether to use Parker weighting for short scans. Defaults to `false`.

# Returns
- `FBPMethod`: Configured FBP reconstruction method.

# Examples
```julia
method = FBPMethod(filter_type=FILTER_SHEPPLOGAN)
```
"""
function FBPMethod(;
    filter_type::Integer = FILTER_RAMLAK,
    filter_d::Real = 1.0f0,
    filter_parameter::Real = -1.0f0,
    custom_filter = nothing,
    short_scan::Bool = false,
)
    return FBPMethod(
        Int(filter_type),
        Float32(filter_d),
        Float32(filter_parameter),
        _normalize_custom_filter(custom_filter),
        short_scan,
    )
end

"""
    SIRTMethod(; iterations, relaxation=1.0, detector_supersampling=1, pixel_supersampling=1, min_constraint=nothing, max_constraint=nothing, projector_kernel=:line)

Create a SIRT reconstruction method.

# Arguments
- `iterations`: Number of iterations. (Required)
- `relaxation`: Relaxation parameter. Defaults to 1.0.
- `detector_supersampling`: GPU-only detector supersampling. Defaults to 1.
- `pixel_supersampling`: CPU-only pixel supersampling. Defaults to 1.
- `min_constraint`: Optional non-negativity or other lower bound.
- `max_constraint`: Optional upper bound.
- `projector_kernel`: Projector kernel for CPU reconstruction (`:line`, `:linear`, `:strip`). Defaults to `:line`.

# Returns
- `SIRTMethod`: Configured SIRT reconstruction method.

# Examples
```julia
method = SIRTMethod(iterations=100, min_constraint=0.0)
```
"""
function SIRTMethod(;
    iterations::Integer,
    relaxation::Real = 1.0f0,
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    min_constraint = nothing,
    max_constraint = nothing,
    projector_kernel::Symbol = :line,
)
    iterations > 0 || throw(ArgumentError("iterations must be > 0"))
    projector_kernel in (:line, :linear, :strip, :distance_driven) ||
        throw(ArgumentError("projector_kernel must be :line, :linear, :strip, or :distance_driven, got :$projector_kernel"))
    return SIRTMethod(
        Int(iterations),
        Float32(relaxation),
        Int(detector_supersampling),
        Int(pixel_supersampling),
        min_constraint === nothing ? nothing : Float32(min_constraint),
        max_constraint === nothing ? nothing : Float32(max_constraint),
        projector_kernel,
    )
end

"""
    SARTMethod(; iterations, relaxation=1.0, projection_order=:random, projection_order_list=nothing, detector_supersampling=1, pixel_supersampling=1, min_constraint=nothing, max_constraint=nothing, projector_kernel=:line)

Create a SART reconstruction method.

# Arguments
- `iterations`: Number of iterations. (Required)
- `relaxation`: Relaxation parameter. Defaults to 1.0.
- `projection_order`: `:random`, `:sequential`, or `:custom`. Defaults to `:random`.
- `projection_order_list`: List of projection indices for `:custom` order.
- `detector_supersampling`: GPU-only detector supersampling. Defaults to 1.
- `pixel_supersampling`: CPU-only pixel supersampling. Defaults to 1.
- `min_constraint`, `max_constraint`: Optional bounds.
- `projector_kernel`: Projector kernel for CPU reconstruction (`:line`, `:linear`, `:strip`). Defaults to `:line`.

# Returns
- `SARTMethod`: Configured SART reconstruction method.

# Examples
```julia
method = SARTMethod(iterations=50, projection_order=:sequential)
```
"""
function SARTMethod(;
    iterations::Integer,
    relaxation::Real = 1.0f0,
    projection_order::Symbol = :random,
    projection_order_list::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    min_constraint = nothing,
    max_constraint = nothing,
    projector_kernel::Symbol = :line,
)
    iterations > 0 || throw(ArgumentError("iterations must be > 0"))
    projection_order in (:random, :sequential, :custom) ||
        throw(ArgumentError("projection_order must be :random, :sequential, or :custom"))
    projector_kernel in (:line, :linear, :strip, :distance_driven) ||
        throw(ArgumentError("projector_kernel must be :line, :linear, :strip, or :distance_driven, got :$projector_kernel"))
    if projection_order == :custom
        projection_order_list === nothing && throw(ArgumentError("projection_order_list is required for :custom"))
    else
        projection_order_list === nothing || throw(ArgumentError("projection_order_list is only valid with projection_order = :custom"))
    end
    return SARTMethod(
        Int(iterations),
        Float32(relaxation),
        projection_order,
        _normalize_projection_order_list(projection_order_list),
        Int(detector_supersampling),
        Int(pixel_supersampling),
        min_constraint === nothing ? nothing : Float32(min_constraint),
        max_constraint === nothing ? nothing : Float32(max_constraint),
        projector_kernel,
    )
end

"""
    CGLSMethod(; iterations, detector_supersampling=1, pixel_supersampling=1, projector_kernel=:line)

Create a CGLS reconstruction method.

# Arguments
- `iterations`: Number of iterations. (Required)
- `detector_supersampling`: Detector supersampling level. Defaults to 1.
- `pixel_supersampling`: Pixel supersampling level. Defaults to 1.
- `projector_kernel`: Projector kernel for CPU reconstruction (`:line`, `:linear`, `:strip`, `:distance_driven`). Defaults to `:line`.

# Returns
- `CGLSMethod`: Configured CGLS reconstruction method.

# Examples
```julia
method = CGLSMethod(iterations=20)
```
"""
function CGLSMethod(;
    iterations::Integer,
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    projector_kernel::Symbol = :line,
)
    iterations > 0 || throw(ArgumentError("iterations must be > 0"))
    projector_kernel in (:line, :linear, :strip, :distance_driven) ||
        throw(ArgumentError("projector_kernel must be :line, :linear, :strip, or :distance_driven, got :$projector_kernel"))
    return CGLSMethod(Int(iterations), Int(detector_supersampling), Int(pixel_supersampling), projector_kernel)
end

"""
    ARTMethod(; iterations, relaxation=1.0, min_constraint=nothing, max_constraint=nothing, projector_kernel=:line)

Create an ART reconstruction method.

# Arguments
- `iterations`: Number of ray projection iterations. (Required)
- `relaxation`: Relaxation parameter. Defaults to 1.0.
- `min_constraint`: Optional non-negativity or other lower bound.
- `max_constraint`: Optional upper bound.
- `projector_kernel`: Projector kernel (`:line`, `:linear`, `:strip`, `:distance_driven`). Defaults to `:line`.

# Returns
- `ARTMethod`: Configured ART reconstruction method.

# Examples
```julia
method = ARTMethod(iterations=1000, relaxation=0.5, min_constraint=0.0)
```
"""
function ARTMethod(;
    iterations::Integer,
    relaxation::Real = 1.0f0,
    min_constraint = nothing,
    max_constraint = nothing,
    projector_kernel::Symbol = :line,
)
    iterations > 0 || throw(ArgumentError("iterations must be > 0"))
    projector_kernel in (:line, :linear, :strip, :distance_driven) ||
        throw(ArgumentError("projector_kernel must be :line, :linear, :strip, or :distance_driven, got :$projector_kernel"))
    return ARTMethod(
        Int(iterations),
        Float32(relaxation),
        min_constraint === nothing ? nothing : Float32(min_constraint),
        max_constraint === nothing ? nothing : Float32(max_constraint),
        projector_kernel,
    )
end

"""
    EMMethod(; iterations, detector_supersampling=1, pixel_supersampling=1)

Create an EM reconstruction method.

# Arguments
- `iterations`: Number of iterations. (Required)
- `detector_supersampling`: Detector supersampling level. Defaults to 1.
- `pixel_supersampling`: Pixel supersampling level. Defaults to 1.

# Returns
- `EMMethod`: Configured EM reconstruction method.

# Examples
```julia
method = EMMethod(iterations=50)
```
"""
function EMMethod(; iterations::Integer, detector_supersampling::Integer = 1, pixel_supersampling::Integer = 1)
    iterations > 0 || throw(ArgumentError("iterations must be > 0"))
    return EMMethod(Int(iterations), Int(detector_supersampling), Int(pixel_supersampling))
end

"""
    SIRT3DMethod(; iterations, relaxation=1.0, voxel_supersampling=1, detector_supersampling=1, min_constraint=nothing, max_constraint=nothing)

Create a 3D SIRT reconstruction method.

# Arguments
- `iterations`: Number of iterations. (Required)
- `relaxation`: Relaxation parameter. Defaults to 1.0.
- `voxel_supersampling`: Voxel supersampling level. Defaults to 1.
- `detector_supersampling`: Detector supersampling level. Defaults to 1.
- `min_constraint`, `max_constraint`: Optional bounds.

# Returns
- `SIRT3DMethod`: Configured 3D SIRT reconstruction method.

# Examples
```julia
method = SIRT3DMethod(iterations=100)
```
"""
function SIRT3DMethod(;
    iterations::Integer,
    relaxation::Real = 1.0f0,
    voxel_supersampling::Integer = 1,
    detector_supersampling::Integer = 1,
    min_constraint = nothing,
    max_constraint = nothing,
)
    iterations > 0 || throw(ArgumentError("iterations must be > 0"))
    return SIRT3DMethod(
        Int(iterations),
        Float32(relaxation),
        Int(voxel_supersampling),
        Int(detector_supersampling),
        min_constraint === nothing ? nothing : Float32(min_constraint),
        max_constraint === nothing ? nothing : Float32(max_constraint),
    )
end

"""
    CGLS3DMethod(; iterations, voxel_supersampling=1, detector_supersampling=1)

Create a 3D CGLS reconstruction method.

# Arguments
- `iterations`: Number of iterations. (Required)
- `voxel_supersampling`: Voxel supersampling level. Defaults to 1.
- `detector_supersampling`: Detector supersampling level. Defaults to 1.

# Returns
- `CGLS3DMethod`: Configured 3D CGLS reconstruction method.

# Examples
```julia
method = CGLS3DMethod(iterations=20)
```
"""
function CGLS3DMethod(; iterations::Integer, voxel_supersampling::Integer = 1, detector_supersampling::Integer = 1)
    iterations > 0 || throw(ArgumentError("iterations must be > 0"))
    return CGLS3DMethod(Int(iterations), Int(voxel_supersampling), Int(detector_supersampling))
end

"""
    FDK3DMethod(; filter_type=FILTER_RAMLAK, filter_d=1.0, filter_parameter=-1.0, custom_filter=nothing, short_scan=false, voxel_supersampling=1)

Create an FDK (cone-beam) reconstruction method.

# Arguments
- `filter_type`: Filter type (e.g., `FILTER_RAMLAK`). Defaults to `FILTER_RAMLAK`.
- `filter_d`: Filter frequency scaling. Defaults to 1.0.
- `filter_parameter`: Optional filter parameter. Defaults to -1.0.
- `custom_filter`: Optional custom frequency domain filter.
- `short_scan`: Whether to use Parker weighting for short scans. Defaults to `false`.
- `voxel_supersampling`: Voxel supersampling level. Defaults to 1.

# Returns
- `FDK3DMethod`: Configured 3D FDK reconstruction method.

# Examples
```julia
method = FDK3DMethod(filter_type=FILTER_RAMLAK)
```
"""
function FDK3DMethod(;
    filter_type::Integer = FILTER_RAMLAK,
    filter_d::Real = 1.0f0,
    filter_parameter::Real = -1.0f0,
    custom_filter = nothing,
    short_scan::Bool = false,
    voxel_supersampling::Integer = 1,
)
    return FDK3DMethod(
        Int(filter_type),
        Float32(filter_d),
        Float32(filter_parameter),
        _normalize_custom_filter(custom_filter),
        short_scan,
        Int(voxel_supersampling),
    )
end

"""
    ProjectionOptions2D(; detector_supersampling=1, projector_kernel=:line)

Create 2D projection options.
"""
function ProjectionOptions2D(; detector_supersampling::Integer = 1, projector_kernel::Symbol = :line)
    detector_supersampling > 0 || throw(ArgumentError("detector_supersampling must be > 0"))
    projector_kernel in (:line, :linear, :strip, :distance_driven) ||
        throw(ArgumentError("projector_kernel must be :line, :linear, :strip, or :distance_driven, got :$projector_kernel"))
    return ProjectionOptions2D(Int(detector_supersampling), projector_kernel)
end
ProjectionOptions2D(detector_supersampling::Int) = ProjectionOptions2D(detector_supersampling, :line)

"""
    ProjectionOptions3D(; detector_supersampling=1)

Create 3D projection options.
"""
function ProjectionOptions3D(; detector_supersampling::Integer = 1)
    detector_supersampling > 0 || throw(ArgumentError("detector_supersampling must be > 0"))
    return ProjectionOptions3D(Int(detector_supersampling))
end

"""
    BackprojectionOptions2D(; detector_supersampling=1, pixel_supersampling=1, projector_kernel=:line)

Create 2D backprojection options.
"""
function BackprojectionOptions2D(;
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    projector_kernel::Symbol = :line,
)
    detector_supersampling > 0 || throw(ArgumentError("detector_supersampling must be > 0"))
    pixel_supersampling > 0 || throw(ArgumentError("pixel_supersampling must be > 0"))
    projector_kernel in (:line, :linear, :strip, :distance_driven) ||
        throw(ArgumentError("projector_kernel must be :line, :linear, :strip, or :distance_driven, got :$projector_kernel"))
    return BackprojectionOptions2D(Int(detector_supersampling), Int(pixel_supersampling), projector_kernel)
end
BackprojectionOptions2D(detector_supersampling::Int, pixel_supersampling::Int) = BackprojectionOptions2D(detector_supersampling, pixel_supersampling, :line)

"""
    BackprojectionOptions3D(; voxel_supersampling=1)

Create 3D backprojection options.
"""
function BackprojectionOptions3D(; voxel_supersampling::Integer = 1)
    voxel_supersampling > 0 || throw(ArgumentError("voxel_supersampling must be > 0"))
    return BackprojectionOptions3D(Int(voxel_supersampling))
end

_normalize_custom_filter(custom_filter::Nothing) = nothing
_normalize_custom_filter(custom_filter::AbstractVector{<:Real}) = Vector{Float32}(custom_filter)
_normalize_custom_filter(custom_filter::AbstractMatrix{<:Real}) = Matrix{Float32}(custom_filter)

function _normalize_custom_filter(custom_filter)
    throw(ArgumentError("custom_filter must be nothing, AbstractVector{<:Real}, or AbstractMatrix{<:Real}"))
end

_normalize_projection_order_list(projection_order_list::Nothing) = nothing
_normalize_projection_order_list(projection_order_list::AbstractVector{<:Integer}) = Int32.(projection_order_list)

_normalize_mask_option(mask::Nothing, ::Type{Array{Float32,2}}, _name::AbstractString) = nothing
_normalize_mask_option(mask::Nothing, ::Type{Array{Float32,3}}, _name::AbstractString) = nothing

function _normalize_mask_option(mask::AbstractMatrix{<:Real}, ::Type{Array{Float32,2}}, _name::AbstractString)
    return Array{Float32,2}(mask)
end

function _normalize_mask_option(mask::AbstractArray{<:Real,3}, ::Type{Array{Float32,3}}, _name::AbstractString)
    return Array{Float32,3}(mask)
end

function _normalize_mask_option(mask, ::Type{Array{Float32,2}}, name::AbstractString)
    throw(ArgumentError("$name must be nothing or AbstractArray{<:Real,2}"))
end

function _normalize_mask_option(mask, ::Type{Array{Float32,3}}, name::AbstractString)
    throw(ArgumentError("$name must be nothing or AbstractArray{<:Real,3}"))
end

"""
    ReconstructionOptions2D(; reconstruction_mask=nothing, sinogram_mask=nothing, residual_norm=nothing)

Create 2D reconstruction options.
"""
function ReconstructionOptions2D(;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
)
    residual_norm === nothing || residual_norm isa Base.RefValue{Float32} ||
        throw(ArgumentError("residual_norm must be nothing or Ref{Float32}"))
    return ReconstructionOptions2D(
        _normalize_mask_option(reconstruction_mask, Array{Float32,2}, "reconstruction_mask"),
        _normalize_mask_option(sinogram_mask, Array{Float32,2}, "sinogram_mask"),
        residual_norm,
    )
end

"""
    ReconstructionOptions3D(; reconstruction_mask=nothing, sinogram_mask=nothing, residual_norm=nothing)

Create 3D reconstruction options.
"""
function ReconstructionOptions3D(;
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
)
    residual_norm === nothing || residual_norm isa Base.RefValue{Float32} ||
        throw(ArgumentError("residual_norm must be nothing or Ref{Float32}"))
    return ReconstructionOptions3D(
        _normalize_mask_option(reconstruction_mask, Array{Float32,3}, "reconstruction_mask"),
        _normalize_mask_option(sinogram_mask, Array{Float32,3}, "sinogram_mask"),
        residual_norm,
    )
end

"""
    set_library!(path::AbstractString)

Set the path to the ASTRA-toolbox shared library.

# Arguments
- `path`: Absolute path to the shared library (e.g., `libastra.so`, `libastra.dll`, or `libastra.dylib`).
"""
set_library!(path::AbstractString) = LowLevel.set_library!(path)

"""
    library_path()

Return the current path to the ASTRA-toolbox shared library.
"""
library_path() = LowLevel.library_path()

"""
    last_error()

Return the last error message from the ASTRA-toolbox C API.
"""
last_error() = LowLevel.last_error()

_angle_count(geometry::Parallel2DGeometry) = length(geometry.angles)
_angle_count(geometry::FanFlat2DGeometry) = length(geometry.angles)
_angle_count(geometry::ParallelVec2DGeometry) = size(geometry.vectors, 2)
_angle_count(geometry::FanFlatVec2DGeometry) = size(geometry.vectors, 2)
_angle_count(geometry::SparseMatrix2DGeometry) = geometry.angle_count
_angle_count(geometry::Parallel3DGeometry) = length(geometry.angles)
_angle_count(geometry::Cone3DGeometry) = length(geometry.angles)
_angle_count(geometry::ParallelVec3DGeometry) = size(geometry.vectors, 2)
_angle_count(geometry::ConeVec3DGeometry) = size(geometry.vectors, 2)
_angle_count(geometry::CylConeVec3DGeometry) = size(geometry.vectors, 2)

_detector_count(geometry::AbstractGeometry2D) = geometry.detector_count
_detector_count_u(geometry::AbstractGeometry3D) = geometry.detector_count_u
_detector_count_v(geometry::AbstractGeometry3D) = geometry.detector_count_v

function _check_volume(volume::Array{Float32,2})
    stride(volume, 1) == 1 || throw(ArgumentError("volume must satisfy stride(volume, 1) == 1"))
    return volume
end

function _check_volume(volume::Array{Float32,3})
    stride(volume, 1) == 1 || throw(ArgumentError("volume must satisfy stride(volume, 1) == 1"))
    return volume
end

function _check_sinogram(sinogram::Array{Float32,2}, geometry::AbstractGeometry2D)
    stride(sinogram, 1) == 1 || throw(ArgumentError("sinogram must satisfy stride(sinogram, 1) == 1"))
    size(sinogram, 1) == _detector_count(geometry) ||
        throw(ArgumentError("size(sinogram, 1) must equal geometry.detector_count"))
    size(sinogram, 2) == _angle_count(geometry) ||
        throw(ArgumentError("size(sinogram, 2) must equal geometry angle count"))
    return sinogram
end

function _check_sinogram(sinogram::Array{Float32,3}, geometry::AbstractGeometry3D)
    stride(sinogram, 1) == 1 || throw(ArgumentError("sinogram must satisfy stride(sinogram, 1) == 1"))
    size(sinogram, 1) == _detector_count_u(geometry) ||
        throw(ArgumentError("size(sinogram, 1) must equal geometry.detector_count_u"))
    size(sinogram, 2) == _angle_count(geometry) ||
        throw(ArgumentError("size(sinogram, 2) must equal geometry angle count"))
    size(sinogram, 3) == _detector_count_v(geometry) ||
        throw(ArgumentError("size(sinogram, 3) must equal geometry.detector_count_v"))
    return sinogram
end

_check_reconstruction_mask(mask::Nothing, _dims) = nothing

function _check_reconstruction_mask(mask::Array{Float32,N}, dims::NTuple{N,Int}) where {N}
    stride(mask, 1) == 1 || throw(ArgumentError("reconstruction_mask must satisfy stride(mask, 1) == 1"))
    size(mask) == dims || throw(ArgumentError("size(reconstruction_mask) must equal $dims"))
    return mask
end

function _check_reconstruction_mask(mask, _dims)
    throw(ArgumentError("reconstruction_mask must be `nothing` or `Array{Float32}` with matching size"))
end

_check_sinogram_mask(mask::Nothing, _dims) = nothing

function _check_sinogram_mask(mask::Array{Float32,N}, dims::NTuple{N,Int}) where {N}
    stride(mask, 1) == 1 || throw(ArgumentError("sinogram_mask must satisfy stride(mask, 1) == 1"))
    size(mask) == dims || throw(ArgumentError("size(sinogram_mask) must equal $dims"))
    return mask
end

function _check_sinogram_mask(mask, _dims)
    throw(ArgumentError("sinogram_mask must be `nothing` or `Array{Float32}` with matching size"))
end

_check_residual_norm(residual_norm::Nothing) = nothing
_check_residual_norm(residual_norm::Base.RefValue{Float32}) = residual_norm

function _check_residual_norm(residual_norm)
    throw(ArgumentError("residual_norm must be `nothing` or `Ref{Float32}`"))
end

function _merge_reconstruction_options(
    ::Type{ReconstructionOptions2D},
    options::Union{Nothing,ReconstructionOptions2D},
    reconstruction_mask,
    sinogram_mask,
    residual_norm,
)
    base = options === nothing ? ReconstructionOptions2D() : options
    return ReconstructionOptions2D(
        reconstruction_mask = reconstruction_mask === nothing ? base.reconstruction_mask : reconstruction_mask,
        sinogram_mask = sinogram_mask === nothing ? base.sinogram_mask : sinogram_mask,
        residual_norm = residual_norm === nothing ? base.residual_norm : residual_norm,
    )
end

function _merge_reconstruction_options(
    ::Type{ReconstructionOptions3D},
    options::Union{Nothing,ReconstructionOptions3D},
    reconstruction_mask,
    sinogram_mask,
    residual_norm,
)
    base = options === nothing ? ReconstructionOptions3D() : options
    return ReconstructionOptions3D(
        reconstruction_mask = reconstruction_mask === nothing ? base.reconstruction_mask : reconstruction_mask,
        sinogram_mask = sinogram_mask === nothing ? base.sinogram_mask : sinogram_mask,
        residual_norm = residual_norm === nothing ? base.residual_norm : residual_norm,
    )
end

