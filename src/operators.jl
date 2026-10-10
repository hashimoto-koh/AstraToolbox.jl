# Matrix-free linear projector operator (OpTomo equivalent for Julia)

using LinearAlgebra
import LinearAlgebra: mul!, adjoint, size

"""
    AstraProjectorMap(geometry, volume_size; backend=..., options=...)

A matrix-free linear operator representing the CT forward projection matrix ``A``.
Its adjoint ``A'`` represents the backprojection operator ``A^T``.

# Arguments
- `geometry`: 2D or 3D geometry object (e.g., `Parallel2DGeometry`, `Cone3DGeometry`).
- `volume_size`: Dimensions of the reconstruction volume:
  - For 2D: `(nx, ny)` or integer `n` (for `(n, n)`).
  - For 3D: `(nx, ny, nz)` or integer `n` (for `(n, n, n)`).
- `backend`: Computational backend (`cpu()` or `cuda()`). Defaults to `cpu()` for 2D geometries and `cuda()` for 3D geometries.
- `options`: Optional projection options (`ProjectionOptions2D` or `ProjectionOptions3D`).

# Properties
- `size(A)`: Returns `(prod(sinogram_size), prod(volume_size))`.
- `A.volume_size`: Dimensions of the reconstruction volume.
- `A.sinogram_size`: Dimensions of the projection sinogram array.
- `A.geometry`: Underlying projection geometry.
- `A.backend`: Computational backend.

# Operations
- `A * x`: Forward projection of vector `x` or volume array `x`.
- `A' * y`: Backprojection of sinogram vector `y` or sinogram array `y`.
- `mul!(y, A, x)`: In-place forward projection ``y \\leftarrow A x``.
- `mul!(x, A', y)`: In-place backprojection ``x \\leftarrow A^T y``.
- 5-argument `mul!(y, A, x, α, β)`: In-place ``y \\leftarrow \\alpha A x + \\beta y``.
- Interoperates directly with `LinearAlgebra`, `IterativeSolvers.jl`, and `Krylov.jl`.

# Returns
- `AstraProjectorMap`: A matrix-free linear operator implementing `AbstractMatrix{Float32}`.

# Examples
```julia
geom = Parallel2DGeometry(range(0, pi, length=180); detector_count=256)
A = AstraProjectorMap(geom, (256, 256))

# Forward projection
vol = shepp_logan(256)
sino = A * vol              # Returns 256x180 Array{Float32, 2}
sino_vec = A * vec(vol)     # Returns 1D Vector{Float32}

# Backprojection
bp = A' * sino              # Returns 256x256 Array{Float32, 2}
bp_vec = A' * sino_vec      # Returns 1D Vector{Float32}
```
"""
struct AstraProjectorMap{N, M, G<:AbstractGeometry, B<:AbstractBackend, Opt, VG} <: AbstractMatrix{Float32}
    geometry::G
    volume_size::NTuple{N, Int}
    sinogram_size::NTuple{M, Int}
    backend::B
    projection_options::Opt
    volume_geometry::VG
end

function AstraProjectorMap(
    geometry::G,
    volume_size::NTuple{N, Int},
    sinogram_size::NTuple{M, Int},
    backend::B,
    projection_options::Opt,
) where {N, M, G<:AbstractGeometry, B<:AbstractBackend, Opt}
    return AstraProjectorMap(geometry, volume_size, sinogram_size, backend, projection_options, nothing)
end

function AstraProjectorMap(
    geometry::AbstractGeometry2D,
    volume_size::Union{Integer, Tuple{Integer, Integer}};
    backend::AbstractBackend = cpu(),
    options::Union{Nothing, ProjectionOptions2D} = nothing,
    volume_geometry::Union{Nothing, VolumeGeometry2D} = nothing,
    window = nothing,
)
    vsize = volume_size isa Integer ? (Int(volume_size), Int(volume_size)) : (Int(volume_size[1]), Int(volume_size[2]))
    (vsize[1] > 0 && vsize[2] > 0) || throw(ArgumentError("volume_size dimensions must be > 0"))
    ssize = (_detector_count(geometry), _angle_count(geometry))
    vg = _resolve_volume_geometry(vsize, volume_geometry, window)
    return AstraProjectorMap(geometry, vsize, ssize, backend, options, vg)
end

function AstraProjectorMap(
    geometry::AbstractGeometry2D,
    vol_geom::VolumeGeometry2D;
    backend::AbstractBackend = cpu(),
    options::Union{Nothing, ProjectionOptions2D} = nothing,
)
    return AstraProjectorMap(geometry, size(vol_geom); backend = backend, options = options, volume_geometry = vol_geom)
end

function AstraProjectorMap(
    geometry::AbstractGeometry3D,
    volume_size::Union{Integer, Tuple{Integer, Integer, Integer}};
    backend::AbstractBackend = cuda(),
    options::Union{Nothing, ProjectionOptions3D} = nothing,
    volume_geometry::Union{Nothing, VolumeGeometry3D} = nothing,
    window = nothing,
)
    vsize = volume_size isa Integer ? (Int(volume_size), Int(volume_size), Int(volume_size)) : (Int(volume_size[1]), Int(volume_size[2]), Int(volume_size[3]))
    (vsize[1] > 0 && vsize[2] > 0 && vsize[3] > 0) || throw(ArgumentError("volume_size dimensions must be > 0"))
    ssize = (_detector_count_u(geometry), _angle_count(geometry), _detector_count_v(geometry))
    vg = _resolve_volume_geometry(vsize, volume_geometry, window)
    return AstraProjectorMap(geometry, vsize, ssize, backend, options, vg)
end

function AstraProjectorMap(
    geometry::AbstractGeometry3D,
    vol_geom::VolumeGeometry3D;
    backend::AbstractBackend = cuda(),
    options::Union{Nothing, ProjectionOptions3D} = nothing,
)
    return AstraProjectorMap(geometry, size(vol_geom); backend = backend, options = options, volume_geometry = vol_geom)
end

# AbstractMatrix interface
Base.size(A::AstraProjectorMap) = (prod(A.sinogram_size), prod(A.volume_size))
Base.IndexStyle(::Type{<:AstraProjectorMap}) = IndexCartesian()

function Base.getindex(A::AstraProjectorMap, i::Integer, j::Integer)
    (1 <= i <= size(A, 1) && 1 <= j <= size(A, 2)) || throw(BoundsError(A, (i, j)))
    ej = zeros(Float32, size(A, 2))
    ej[j] = 1.0f0
    y = A * ej
    return y[i]
end

# In-place mul! (Forward projection)
function LinearAlgebra.mul!(y::AbstractVector, A::AstraProjectorMap, x::AbstractVector)
    length(x) == size(A, 2) || throw(DimensionMismatch("length(x) ($(length(x))) must equal size(A, 2) ($(size(A, 2)))"))
    length(y) == size(A, 1) || throw(DimensionMismatch("length(y) ($(length(y))) must equal size(A, 1) ($(size(A, 1)))"))

    x_arr = x isa Array{Float32} ? reshape(x, A.volume_size) : reshape(Float32.(x), A.volume_size)
    if y isa Array{Float32}
        y_arr = reshape(y, A.sinogram_size)
        project!(y_arr, x_arr, A.geometry; backend = A.backend, options = A.projection_options, volume_geometry = A.volume_geometry)
    else
        y_tmp = Array{Float32}(undef, A.sinogram_size)
        project!(y_tmp, x_arr, A.geometry; backend = A.backend, options = A.projection_options, volume_geometry = A.volume_geometry)
        copyto!(y, y_tmp)
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, A::AstraProjectorMap, x::AbstractVector, α::Number, β::Number)
    if iszero(β)
        mul!(y, A, x)
        if !isone(α)
            rmul!(y, α)
        end
        return y
    end
    y_tmp = A * x
    if isone(β)
        y .+= α .* y_tmp
    else
        y .= α .* y_tmp .+ β .* y
    end
    return y
end

# In-place mul! (Adjoint / Backprojection)
function LinearAlgebra.mul!(x::AbstractVector, A_adj::Adjoint{<:Any, <:AstraProjectorMap}, y::AbstractVector)
    A = A_adj.parent
    length(y) == size(A, 1) || throw(DimensionMismatch("length(y) ($(length(y))) must equal size(A, 1) ($(size(A, 1)))"))
    length(x) == size(A, 2) || throw(DimensionMismatch("length(x) ($(length(x))) must equal size(A, 2) ($(size(A, 2)))"))

    y_arr = y isa Array{Float32} ? reshape(y, A.sinogram_size) : reshape(Float32.(y), A.sinogram_size)
    if x isa Array{Float32}
        x_arr = reshape(x, A.volume_size)
        backproject!(x_arr, y_arr, A.geometry; backend = A.backend, volume_geometry = A.volume_geometry)
    else
        x_tmp = Array{Float32}(undef, A.volume_size)
        backproject!(x_tmp, y_arr, A.geometry; backend = A.backend, volume_geometry = A.volume_geometry)
        copyto!(x, x_tmp)
    end
    return x
end

function LinearAlgebra.mul!(x::AbstractVector, A_adj::Adjoint{<:Any, <:AstraProjectorMap}, y::AbstractVector, α::Number, β::Number)
    A = A_adj.parent
    if iszero(β)
        mul!(x, A_adj, y)
        if !isone(α)
            rmul!(x, α)
        end
        return x
    end
    x_tmp = A_adj * y
    if isone(β)
        x .+= α .* x_tmp
    else
        x .= α .* x_tmp .+ β .* x
    end
    return x
end

# High-level * multiplication
function Base.:*(A::AstraProjectorMap, x::AbstractVector)
    length(x) == size(A, 2) || throw(DimensionMismatch("length(x) ($(length(x))) must equal size(A, 2) ($(size(A, 2)))"))
    y = Vector{Float32}(undef, size(A, 1))
    mul!(y, A, x)
    return y
end

function Base.:*(A::AstraProjectorMap, volume::AbstractMatrix{<:Real})
    if size(volume) == A.volume_size
        return project(Array{Float32}(volume), A.geometry; backend = A.backend, options = A.projection_options, volume_geometry = A.volume_geometry)
    elseif size(volume, 1) == size(A, 2)
        Y = Matrix{Float32}(undef, size(A, 1), size(volume, 2))
        for j in 1:size(volume, 2)
            mul!(view(Y, :, j), A, view(volume, :, j))
        end
        return Y
    else
        throw(DimensionMismatch("matrix size $(size(volume)) matches neither volume_size $(A.volume_size) nor operator input dimension $(size(A, 2))"))
    end
end

function Base.:*(A::AstraProjectorMap, volume::AbstractArray{<:Real, 3})
    if size(volume) == A.volume_size
        return project(Array{Float32}(volume), A.geometry; backend = A.backend, options = A.projection_options, volume_geometry = A.volume_geometry)
    else
        throw(DimensionMismatch("volume size $(size(volume)) does not match operator volume_size $(A.volume_size)"))
    end
end

function Base.:*(A_adj::Adjoint{<:Any, <:AstraProjectorMap}, y::AbstractVector)
    A = A_adj.parent
    length(y) == size(A, 1) || throw(DimensionMismatch("length(y) ($(length(y))) must equal size(A, 1) ($(size(A, 1)))"))
    x = Vector{Float32}(undef, size(A, 2))
    mul!(x, A_adj, y)
    return x
end

function Base.:*(A_adj::Adjoint{<:Any, <:AstraProjectorMap}, sinogram::AbstractMatrix{<:Real})
    A = A_adj.parent
    if size(sinogram) == A.sinogram_size
        return backproject(Array{Float32}(sinogram), A.geometry; volume_size = A.volume_size, volume_geometry = A.volume_geometry, backend = A.backend)
    elseif size(sinogram, 1) == size(A, 1)
        X = Matrix{Float32}(undef, size(A, 2), size(sinogram, 2))
        for j in 1:size(sinogram, 2)
            mul!(view(X, :, j), A_adj, view(sinogram, :, j))
        end
        return X
    else
        throw(DimensionMismatch("matrix size $(size(sinogram)) matches neither sinogram_size $(A.sinogram_size) nor operator input dimension $(size(A, 1))"))
    end
end

function Base.:*(A_adj::Adjoint{<:Any, <:AstraProjectorMap}, sinogram::AbstractArray{<:Real, 3})
    A = A_adj.parent
    if size(sinogram) == A.sinogram_size
        return backproject(Array{Float32}(sinogram), A.geometry; volume_size = A.volume_size, volume_geometry = A.volume_geometry, backend = A.backend)
    else
        throw(DimensionMismatch("sinogram size $(size(sinogram)) does not match operator sinogram_size $(A.sinogram_size)"))
    end
end
