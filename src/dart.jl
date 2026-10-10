# Discrete Algebraic Reconstruction Technique (DART)
# Reference: Batenburg & Sijbers, "DART: A Practical Reconstruction Algorithm for Discrete Tomography", IEEE TMI 2011.

using Random

"""
    dart_segment!(segmentation::AbstractArray{Float32}, volume::AbstractArray{<:Real}, gray_levels::AbstractVector{<:Real}; thresholds=nothing) -> segmentation

Segment `volume` in-place into discrete `gray_levels`.
If `thresholds` is not provided, decision thresholds are computed as midpoints between adjacent sorted gray levels.

# Arguments
- `segmentation`: Destination Float32 array of the same size as `volume`.
- `volume`: Input continuous volume array.
- `gray_levels`: Collection of discrete attenuation/density values.
- `thresholds`: Optional collection of length `length(gray_levels) - 1` specifying partition boundaries.

# Returns
- `segmentation`: The modified segmentation array.

# Examples
```julia
S = zeros(Float32, size(V))
dart_segment!(S, V, [0.0, 1.0, 2.0])
```
"""
function dart_segment!(
    segmentation::AbstractArray{Float32},
    volume::AbstractArray{<:Real},
    gray_levels::AbstractVector{<:Real};
    thresholds = nothing,
)
    size(segmentation) == size(volume) || throw(DimensionMismatch("size(segmentation) ($(size(segmentation))) must equal size(volume) ($(size(volume)))"))
    length(gray_levels) >= 1 || throw(ArgumentError("gray_levels must have at least one level"))

    sorted_levels = Float32.(sort(collect(gray_levels)))
    K = length(sorted_levels)

    if K == 1
        fill!(segmentation, sorted_levels[1])
        return segmentation
    end

    thresh = if thresholds !== nothing
        length(thresholds) == K - 1 || throw(ArgumentError("thresholds must have length $(K - 1)"))
        Float32.(thresholds)
    else
        [(sorted_levels[i] + sorted_levels[i + 1]) / 2.0f0 for i in 1:(K - 1)]
    end

    @inbounds for i in eachindex(volume)
        v = Float32(volume[i])
        assigned = sorted_levels[K]
        for k in 1:(K - 1)
            if v < thresh[k]
                assigned = sorted_levels[k]
                break
            end
        end
        segmentation[i] = assigned
    end

    return segmentation
end

"""
    dart_segment(volume::AbstractArray{<:Real}, gray_levels::AbstractVector{<:Real}; thresholds=nothing) -> Array{Float32}

Segment `volume` into discrete `gray_levels`. Out-of-place version of `dart_segment!`.

# Arguments
- `volume`: Input continuous volume array.
- `gray_levels`: Collection of discrete attenuation/density values.
- `thresholds`: Optional collection of length `length(gray_levels) - 1` specifying partition boundaries.

# Returns
- `Array{Float32}`: A new segmented discrete volume array.

# Examples
```julia
S = dart_segment(V, [0.0, 1.0, 2.0])
```
"""
function dart_segment(
    volume::AbstractArray{<:Real},
    gray_levels::AbstractVector{<:Real};
    thresholds = nothing,
)
    segmentation = Array{Float32}(undef, size(volume))
    return dart_segment!(segmentation, volume, gray_levels; thresholds = thresholds)
end

# -------------------------------------------------------------------------
# DART Boundary Masking
# -------------------------------------------------------------------------

"""
    dart_mask!(mask::AbstractArray{Float32}, segmentation::AbstractArray{Float32};
               connectivity=nothing, radius::Integer=1, threshold::Integer=1,
               p_free::Real=0.0, backend::AbstractBackend=cpu(), rng=Random.default_rng()) -> mask

Compute DART boundary mask from a segmented volume in-place.
Voxels located on discrete material boundaries (where neighborhood values vary) are marked as 1.0f0 (free voxels),
while homogeneous regions are marked as 0.0f0 (fixed voxels).
Optionally adds random free pixels with probability `p_free` (between 0.0 and 1.0).

# Connectivity
- 2D: `4` or `8` (default: 8).
- 3D: `6` or `26` (default: 26).

# Arguments
- `mask`: Destination array of the same dimensions as `segmentation`.
- `segmentation`: Input discrete segmented array.
- `radius`: Search radius around each voxel (default: 1).
- `threshold`: Minimum number of varying neighbors required to flag a boundary (default: 1).
- `p_free`: Probability of randomly selecting non-boundary voxels as free (default: 0.0).
- `backend`: `cpu()` or `cuda()`.
- `rng`: Random number generator used for random free pixel selection.

# Returns
- `mask`: The modified boundary mask array (mutated in-place).

# Examples
```julia
mask = zeros(Float32, size(S))
dart_mask!(mask, S; radius=1, p_free=0.05)
```
"""
function dart_mask!(
    mask::AbstractArray{Float32},
    segmentation::AbstractArray{Float32};
    connectivity = nothing,
    radius::Integer = 1,
    threshold::Integer = 1,
    p_free::Real = 0.0f0,
    backend::AbstractBackend = cpu(),
    rng::Random.AbstractRNG = Random.default_rng(),
)
    size(mask) == size(segmentation) || throw(DimensionMismatch("size(mask) ($(size(mask))) must equal size(segmentation) ($(size(segmentation)))"))
    radius >= 1 || throw(ArgumentError("radius must be >= 1"))
    threshold >= 1 || throw(ArgumentError("threshold must be >= 1"))
    0.0 <= p_free <= 1.0 || throw(ArgumentError("p_free must be in [0.0, 1.0]"))

    nd = ndims(segmentation)
    if nd == 2
        conn = connectivity !== nothing ? Int(connectivity) : 8
        conn == 4 || conn == 8 || throw(ArgumentError("2D connectivity must be 4 or 8"))
        if backend isa CUDABackend
            _dart_mask_2d_cuda!(mask, segmentation, backend; connectivity = conn, radius = radius, threshold = threshold, p_free = p_free, rng = rng)
        else
            _dart_mask_2d_cpu!(mask, segmentation; connectivity = conn, radius = radius, threshold = threshold, p_free = p_free, rng = rng)
        end
    elseif nd == 3
        conn = connectivity !== nothing ? Int(connectivity) : 26
        conn == 6 || conn == 26 || throw(ArgumentError("3D connectivity must be 6 or 26"))
        if backend isa CUDABackend
            _dart_mask_3d_cuda!(mask, segmentation, backend; connectivity = conn, radius = radius, threshold = threshold, p_free = p_free, rng = rng)
        else
            _dart_mask_3d_cpu!(mask, segmentation; connectivity = conn, radius = radius, threshold = threshold, p_free = p_free, rng = rng)
        end
    else
        throw(ArgumentError("dart_mask! supports only 2D and 3D arrays"))
    end

    return mask
end

"""
    dart_mask(segmentation::AbstractArray{Float32}; kwargs...) -> Array{Float32}

Compute DART boundary mask from a segmented volume. Out-of-place version of `dart_mask!`.

# Arguments
- `segmentation`: Input discrete segmented array.
- `kwargs...`: Optional keyword arguments forwarded to `dart_mask!`.

# Returns
- `Array{Float32}`: A new boundary mask array (1.0 = free, 0.0 = fixed).

# Examples
```julia
mask = dart_mask(S; radius=1, p_free=0.05)
```
"""
function dart_mask(
    segmentation::AbstractArray{Float32};
    kwargs...,
)
    mask = zeros(Float32, size(segmentation))
    return dart_mask!(mask, segmentation; kwargs...)
end

function _dart_mask_2d_cpu!(
    mask::AbstractMatrix{Float32},
    segmentation::AbstractMatrix{Float32};
    connectivity::Integer = 8,
    radius::Integer = 1,
    threshold::Integer = 1,
    p_free::Real = 0.0f0,
    rng::Random.AbstractRNG = Random.default_rng(),
)
    w, h = size(segmentation)
    fill!(mask, 0.0f0)
    r = Int(radius)
    thresh = Int(threshold)
    conn = Int(connectivity)

    for y in (1 + r):(h - r)
        for x in (1 + r):(w - r)
            center_val = segmentation[x, y]
            diff_count = 0
            if conn == 8
                for dy in -r:r
                    for dx in -r:r
                        if dx == 0 && dy == 0
                            continue
                        end
                        if segmentation[x + dx, y + dy] != center_val
                            diff_count += 1
                            if diff_count >= thresh
                                mask[x, y] = 1.0f0
                                @goto next_pixel_2d
                            end
                        end
                    end
                end
            else # conn == 4
                for dx in -r:r
                    if dx != 0 && segmentation[x + dx, y] != center_val
                        diff_count += 1
                        if diff_count >= thresh
                            mask[x, y] = 1.0f0
                            @goto next_pixel_2d
                        end
                    end
                end
                for dy in -r:r
                    if dy != 0 && segmentation[x, y + dy] != center_val
                        diff_count += 1
                        if diff_count >= thresh
                            mask[x, y] = 1.0f0
                            @goto next_pixel_2d
                        end
                    end
                end
            end
            @label next_pixel_2d
        end
    end

    if p_free > 0.0f0
        p = Float32(p_free)
        for i in eachindex(mask)
            if mask[i] == 0.0f0 && rand(rng, Float32) < p
                mask[i] = 1.0f0
            end
        end
    end

    return mask
end

function _dart_mask_2d_cuda!(
    mask::AbstractMatrix{Float32},
    segmentation::AbstractMatrix{Float32},
    backend::CUDABackend;
    connectivity::Integer = 8,
    radius::Integer = 1,
    threshold::Integer = 1,
    p_free::Real = 0.0f0,
    rng::Random.AbstractRNG = Random.default_rng(),
)
    mask_arr = mask isa Array{Float32,2} ? mask : Array{Float32,2}(mask)
    seg_arr = segmentation isa Array{Float32,2} ? segmentation : Array{Float32,2}(segmentation)

    LowLevel.cuda_dart_mask_2d_f32!(
        mask_arr,
        seg_arr;
        connectivity = connectivity,
        radius = radius,
        threshold = threshold,
        gpu_index = backend.gpu_index,
    )

    if mask !== mask_arr
        copyto!(mask, mask_arr)
    end

    if p_free > 0.0f0
        p = Float32(p_free)
        for i in eachindex(mask)
            if mask[i] == 0.0f0 && rand(rng, Float32) < p
                mask[i] = 1.0f0
            end
        end
    end

    return mask
end

function _dart_mask_3d_cpu!(
    mask::AbstractArray{Float32,3},
    segmentation::AbstractArray{Float32,3};
    connectivity::Integer = 26,
    radius::Integer = 1,
    threshold::Integer = 1,
    p_free::Real = 0.0f0,
    rng::Random.AbstractRNG = Random.default_rng(),
)
    w, h, d = size(segmentation)
    fill!(mask, 0.0f0)
    r = Int(radius)
    thresh = Int(threshold)
    conn = Int(connectivity)

    for z in 1:d
        for y in (1 + r):(h - r)
            for x in (1 + r):(w - r)
                center_val = segmentation[x, y, z]
                diff_count = 0
                if conn == 26
                    for dz in -r:r
                        nz = z + dz
                        if nz < 1 || nz > d
                            continue
                        end
                        for dy in -r:r
                            for dx in -r:r
                                if dx == 0 && dy == 0 && dz == 0
                                    continue
                                end
                                if segmentation[x + dx, y + dy, nz] != center_val
                                    diff_count += 1
                                    if diff_count >= thresh
                                        mask[x, y, z] = 1.0f0
                                        @goto next_voxel_3d
                                    end
                                end
                            end
                        end
                    end
                else # conn == 6
                    neighbors = ((x - 1, y, z), (x + 1, y, z), (x, y - 1, z), (x, y + 1, z), (x, y, z - 1), (x, y, z + 1))
                    for (nx, ny, nz) in neighbors
                        if 1 <= nx <= w && 1 <= ny <= h && 1 <= nz <= d
                            if segmentation[nx, ny, nz] != center_val
                                diff_count += 1
                                if diff_count >= thresh
                                    mask[x, y, z] = 1.0f0
                                    @goto next_voxel_3d
                                end
                            end
                        end
                    end
                end
                @label next_voxel_3d
            end
        end
    end

    if p_free > 0.0f0
        p = Float32(p_free)
        for i in eachindex(mask)
            if mask[i] == 0.0f0 && rand(rng, Float32) < p
                mask[i] = 1.0f0
            end
        end
    end

    return mask
end

function _dart_mask_3d_cuda!(
    mask::AbstractArray{Float32,3},
    segmentation::AbstractArray{Float32,3},
    backend::CUDABackend;
    connectivity::Integer = 26,
    radius::Integer = 1,
    threshold::Integer = 1,
    p_free::Real = 0.0f0,
    rng::Random.AbstractRNG = Random.default_rng(),
)
    fill!(mask, 0.0f0)
    mask_arr = mask isa Array{Float32,3} ? mask : Array{Float32,3}(mask)
    seg_arr = segmentation isa Array{Float32,3} ? segmentation : Array{Float32,3}(segmentation)

    LowLevel.cuda_dart_mask_3d_f32!(
        mask_arr,
        seg_arr;
        connectivity = connectivity,
        radius = radius,
        threshold = threshold,
        gpu_index = backend.gpu_index,
    )

    if mask !== mask_arr
        copyto!(mask, mask_arr)
    end

    if p_free > 0.0f0
        p = Float32(p_free)
        for i in eachindex(mask)
            if mask[i] == 0.0f0 && rand(rng, Float32) < p
                mask[i] = 1.0f0
            end
        end
    end

    return mask
end

# -------------------------------------------------------------------------
# DART Smoothing
# -------------------------------------------------------------------------

"""
    dart_smoothing!(out::AbstractArray{Float32}, in::AbstractArray{Float32};
                    b::Real=0.1, radius::Integer=1, backend::AbstractBackend=cpu()) -> out

Apply DART spatial smoothing filter.
Each pixel is updated as: `out = (1 - b) * in + b * avg_neighbors`.

# Arguments
- `out`: Destination array of the same dimensions as `in`.
- `in`: Input continuous volume array.
- `b`: Smoothing intensity in `[0.0, 1.0]` (default: 0.1).
- `radius`: Smoothing radius (default: 1).
- `backend`: `cpu()` or `cuda()`.

# Returns
- `out`: The smoothed volume array (mutated in-place).

# Examples
```julia
out = zeros(Float32, size(V))
dart_smoothing!(out, V; b=0.1, radius=1)
```
"""
function dart_smoothing!(
    out::AbstractArray{Float32},
    in::AbstractArray{Float32};
    b::Real = 0.1f0,
    radius::Integer = 1,
    backend::AbstractBackend = cpu(),
)
    size(out) == size(in) || throw(DimensionMismatch("size(out) ($(size(out))) must equal size(in) ($(size(in)))"))
    0.0 <= b <= 1.0 || throw(ArgumentError("smoothing intensity b must be in [0.0, 1.0]"))
    radius >= 1 || throw(ArgumentError("radius must be >= 1"))

    nd = ndims(in)
    if nd == 2
        if backend isa CUDABackend
            _dart_smoothing_2d_cuda!(out, in, backend; b = b, radius = radius)
        else
            _dart_smoothing_2d_cpu!(out, in; b = b, radius = radius)
        end
    elseif nd == 3
        if backend isa CUDABackend
            _dart_smoothing_3d_cuda!(out, in, backend; b = b, radius = radius)
        else
            _dart_smoothing_3d_cpu!(out, in; b = b, radius = radius)
        end
    else
        throw(ArgumentError("dart_smoothing! supports only 2D and 3D arrays"))
    end

    return out
end

"""
    dart_smoothing(in::AbstractArray{Float32}; kwargs...) -> Array{Float32}

Apply DART spatial smoothing filter. Out-of-place version of `dart_smoothing!`.

# Arguments
- `in`: Input continuous volume array.
- `kwargs...`: Optional keyword arguments forwarded to `dart_smoothing!`.

# Returns
- `Array{Float32}`: A new smoothed volume array.

# Examples
```julia
V_smooth = dart_smoothing(V; b=0.1, radius=1)
```
"""
function dart_smoothing(
    in::AbstractArray{Float32};
    kwargs...,
)
    out = Array{Float32}(undef, size(in))
    return dart_smoothing!(out, in; kwargs...)
end

function _dart_smoothing_2d_cpu!(
    out::AbstractMatrix{Float32},
    in::AbstractMatrix{Float32};
    b::Real = 0.1f0,
    radius::Integer = 1,
)
    w, h = size(in)
    copyto!(out, in)
    r = Int(radius)
    bf = Float32(b)
    count = Float32(4 * r * (r + 1))

    for y in (1 + r):(h - r)
        for x in (1 + r):(w - r)
            center_val = in[x, y]
            res = -center_val
            for dy in -r:r
                for dx in -r:r
                    res += in[x + dx, y + dy]
                end
            end
            res *= bf / count
            res += (1.0f0 - bf) * center_val
            out[x, y] = res
        end
    end
    return out
end

function _dart_smoothing_2d_cuda!(
    out::AbstractMatrix{Float32},
    in::AbstractMatrix{Float32},
    backend::CUDABackend;
    b::Real = 0.1f0,
    radius::Integer = 1,
)
    out_arr = out isa Array{Float32,2} ? out : Array{Float32,2}(out)
    in_arr = in isa Array{Float32,2} ? in : Array{Float32,2}(in)

    LowLevel.cuda_dart_smoothing_2d_f32!(
        out_arr,
        in_arr;
        b = b,
        radius = radius,
        gpu_index = backend.gpu_index,
    )

    if out !== out_arr
        copyto!(out, out_arr)
    end
    return out
end

function _dart_smoothing_3d_cpu!(
    out::AbstractArray{Float32,3},
    in::AbstractArray{Float32,3};
    b::Real = 0.1f0,
    radius::Integer = 1,
)
    w, h, d = size(in)
    copyto!(out, in)
    bf = Float32(b)

    for z in 1:d
        for y in 2:(h - 1)
            for x in 2:(w - 1)
                res = 0.0f0
                if z > 1
                    for dy in -1:1
                        for dx in -1:1
                            res += in[x + dx, y + dy, z - 1]
                        end
                    end
                end
                if z < d
                    for dy in -1:1
                        for dx in -1:1
                            res += in[x + dx, y + dy, z + 1]
                        end
                    end
                end
                for dy in -1:1
                    for dx in -1:1
                        if dx != 0 || dy != 0
                            res += in[x + dx, y + dy, z]
                        end
                    end
                end
                out[x, y, z] = (1.0f0 - bf) * in[x, y, z] + bf * (res / 26.0f0)
            end
        end
    end
    return out
end

function _dart_smoothing_3d_cuda!(
    out::AbstractArray{Float32,3},
    in::AbstractArray{Float32,3},
    backend::CUDABackend;
    b::Real = 0.1f0,
    radius::Integer = 1,
)
    copyto!(out, in)
    out_arr = out isa Array{Float32,3} ? out : Array{Float32,3}(out)
    in_arr = in isa Array{Float32,3} ? in : Array{Float32,3}(in)

    LowLevel.cuda_dart_smoothing_3d_f32!(
        out_arr,
        in_arr;
        b = b,
        radius = radius,
        gpu_index = backend.gpu_index,
    )

    if out !== out_arr
        copyto!(out, out_arr)
    end
    return out
end

# -------------------------------------------------------------------------
# DART Full Iterative Reconstruction Algorithm
# -------------------------------------------------------------------------

"""
    reconstruct_dart(sinogram, geometry, gray_levels;
                     iterations=10,
                     initial_reconstruction=nothing,
                     initial_iterations=50,
                     arm_iterations=5,
                     initial_method=nothing,
                     arm_method_type=SIRTMethod,
                     connectivity=nothing,
                     radius=1,
                     threshold=1,
                     p_free=0.05,
                     smoothing_b=0.1,
                     smoothing_radius=1,
                     thresholds=nothing,
                     backend=...,
                     volume_size=nothing,
                     volume_geometry=nothing,
                     window=nothing,
                     rng=Random.default_rng(),
                     verbose=false) -> NamedTuple{(:reconstruction, :segmentation, :mask)}

Execute Discrete Algebraic Reconstruction Technique (DART) for 2D or 3D CT reconstruction.

# Arguments
- `sinogram`: 2D or 3D sinogram array.
- `geometry`: 2D or 3D projection geometry.
- `gray_levels`: Vector of known discrete attenuation levels.
- `iterations`: Number of outer DART iterations (default: 10).
- `initial_reconstruction`: Optional initial reconstruction estimate. If omitted, computed using `initial_method`.
- `initial_iterations`: Number of iterations for initial reconstruction (default: 50).
- `arm_iterations`: Number of Algebraic Reconstruction Method iterations per DART cycle on free pixels (default: 5).
- `initial_method`: Optional `AbstractMethod` for initial continuous reconstruction.
- `arm_method_type`: Method type for ARM updates (default: `SIRTMethod` for 2D, `SIRT3DMethod` for 3D).
- `connectivity`: Neighborhood connectivity (2D: `4` or `8` [default 8]; 3D: `6` or `26` [default 26]).
- `radius`: Masking neighborhood radius (default: 1).
- `threshold`: Boundary threshold count (default: 1).
- `p_free`: Probability of random free pixel selection (default: 0.05).
- `smoothing_b`: Spatial smoothing intensity (default: 0.1). Set to 0.0 to disable smoothing.
- `smoothing_radius`: Spatial smoothing radius (default: 1).
- `thresholds`: Optional decision boundary thresholds. If omitted, midpoints between `gray_levels` are used.
- `backend`: Computational backend (`cpu()` or `cuda()`).
- `volume_size`: Dimensions of volume `(nx, ny)` or `(nx, ny, nz)`.
- `volume_geometry`: Optional `VolumeGeometry2D` / `VolumeGeometry3D`.
- `window`: Optional ROI window coordinates.
- `rng`: Random number generator.
- `verbose`: Whether to log iteration progress (default: false).

# Returns
A NamedTuple with:
- `reconstruction::Array{Float32}`: Final continuous reconstructed volume.
- `segmentation::Array{Float32}`: Final segmented discrete volume.
- `mask::Array{Float32}`: Final boundary mask (1.0 = free, 0.0 = fixed).

# Examples
```julia
geom = Parallel2DGeometry(range(0, pi, length=60); detector_count=128)
res = reconstruct_dart(sino, geom, [0.0, 1.0]; iterations=10)
vol = res.reconstruction
seg = res.segmentation
```
"""
function reconstruct_dart(
    sinogram::AbstractArray{<:Real},
    geometry::AbstractGeometry,
    gray_levels::AbstractVector{<:Real};
    iterations::Integer = 10,
    initial_reconstruction = nothing,
    initial_iterations::Integer = 50,
    arm_iterations::Integer = 5,
    initial_method::Union{Nothing,AbstractMethod} = nothing,
    arm_method_type = SIRTMethod,
    connectivity = nothing,
    radius::Integer = 1,
    threshold::Integer = 1,
    p_free::Real = 0.05f0,
    smoothing_b::Real = 0.1f0,
    smoothing_radius::Integer = 1,
    thresholds = nothing,
    backend::AbstractBackend = geometry isa AbstractGeometry3D ? cuda() : cpu(),
    volume_size = nothing,
    volume_geometry = nothing,
    window = nothing,
    rng::Random.AbstractRNG = Random.default_rng(),
    verbose::Bool = false,
)
    iterations >= 1 || throw(ArgumentError("iterations must be >= 1"))
    length(gray_levels) >= 1 || throw(ArgumentError("gray_levels must have at least one level"))

    # Resolve volume dimensions and geometry
    default_vsize = if geometry isa AbstractGeometry3D
        (_detector_count_u(geometry), _detector_count_u(geometry), _detector_count_v(geometry))
    else
        (_detector_count(geometry), _detector_count(geometry))
    end
    target_vsize = volume_geometry !== nothing ? size(volume_geometry) : (volume_size !== nothing ? volume_size : default_vsize)
    vol_geom = _resolve_volume_geometry(target_vsize, volume_geometry, window)
    dims = size(vol_geom)

    # Initial continuous reconstruction
    V = if initial_reconstruction !== nothing
        size(initial_reconstruction) == dims || throw(DimensionMismatch("size(initial_reconstruction) does not match volume size $dims"))
        Array{Float32}(initial_reconstruction)
    else
        init_m = if initial_method !== nothing
            initial_method
        elseif geometry isa AbstractGeometry3D
            SIRT3DMethod(iterations = initial_iterations)
        else
            SIRTMethod(iterations = initial_iterations)
        end
        reconstruct(sinogram, geometry, init_m; backend = backend, volume_geometry = vol_geom)
    end

    S = Array{Float32}(undef, dims)
    M = Array{Float32}(undef, dims)
    V_fixed = Array{Float32}(undef, dims)
    V_update = Array{Float32}(undef, dims)
    sino_f32 = sinogram isa Array{Float32} ? sinogram : Array{Float32}(sinogram)
    p_fixed = similar(sino_f32)
    R = similar(sino_f32)

    conn = connectivity !== nothing ? Int(connectivity) : (length(dims) == 3 ? 26 : 8)

    for iter in 1:iterations
        # 1. Segmentation
        dart_segment!(S, V, gray_levels; thresholds = thresholds)

        # 2. Boundary Mask
        dart_mask!(M, S; connectivity = conn, radius = radius, threshold = threshold, p_free = p_free, backend = backend, rng = rng)

        # 3. Fixed pixels: S on non-boundary (M == 0), 0 on boundary (M == 1)
        @inbounds for i in eachindex(V)
            V_fixed[i] = M[i] == 0.0f0 ? S[i] : 0.0f0
        end

        # 4. Residual projection
        project!(p_fixed, V_fixed, geometry; backend = backend, volume_geometry = vol_geom)
        @inbounds for i in eachindex(R)
            R[i] = sino_f32[i] - p_fixed[i]
        end

        # 5. ARM update on free pixels
        arm_method = if geometry isa AbstractGeometry3D
            if arm_method_type == SIRT3DMethod || arm_method_type == SIRTMethod
                SIRT3DMethod(iterations = arm_iterations)
            elseif arm_method_type == CGLS3DMethod || arm_method_type == CGLSMethod
                CGLS3DMethod(iterations = arm_iterations)
            else
                SIRT3DMethod(iterations = arm_iterations)
            end
        else
            if arm_method_type == SIRTMethod
                SIRTMethod(iterations = arm_iterations)
            elseif arm_method_type == SARTMethod
                SARTMethod(iterations = arm_iterations)
            elseif arm_method_type == CGLSMethod
                CGLSMethod(iterations = arm_iterations)
            elseif arm_method_type == ARTMethod
                ARTMethod(iterations = arm_iterations)
            else
                SIRTMethod(iterations = arm_iterations)
            end
        end

        # Free pixels initialized from V where M == 1
        @inbounds for i in eachindex(V)
            V_update[i] = M[i] == 1.0f0 ? V[i] : 0.0f0
        end

        reconstruct!(V_update, R, geometry, arm_method; backend = backend, reconstruction_mask = M, volume_geometry = vol_geom)

        # Update V: fixed pixels from S, free pixels from V_update
        @inbounds for i in eachindex(V)
            V[i] = M[i] == 1.0f0 ? V_update[i] : S[i]
        end

        # 6. Smoothing
        if smoothing_b > 0.0f0
            V_smoothed = dart_smoothing(V; b = smoothing_b, radius = smoothing_radius, backend = backend)
            copyto!(V, V_smoothed)
        end

        if verbose
            println("[DART] Iteration $iter / $iterations completed")
        end
    end

    # Final segmentation
    dart_segment!(S, V, gray_levels; thresholds = thresholds)

    return (reconstruction = V, segmentation = S, mask = M)
end
