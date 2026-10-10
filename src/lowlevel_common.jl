function _default_library_candidates()
    package_root = normpath(joinpath(@__DIR__, ".."))
    built_lib = @isdefined(libastra) ? libastra : joinpath(package_root, "deps", "usr", "lib", "libastra.so")
    String[
        get(ENV, "ASTRA_JULIA_LIB", ""),
        built_lib,
        "libastra.so",
    ]
end

function _resolve_library_path()
    for candidate in _default_library_candidates()
        isempty(candidate) && continue
        if occursin('/', candidate)
            isfile(candidate) && return candidate
        else
            try
                Libdl.dlopen(candidate) do _
                    return candidate
                end
            catch
            end
        end
    end
    error("libastra was not found; run Pkg.build(\"AstraToolbox\"), call set_library!(path), or set ASTRA_JULIA_LIB")
end

function set_library!(path::AbstractString)
    _library_path[] = String(path)
    return _library_path[]
end

function library_path()
    if isempty(_library_path[])
        _library_path[] = _resolve_library_path()
    end
    return _library_path[]
end

function _libhandle()
    return Libdl.dlopen(library_path())
end

function last_error()
    lib = _libhandle()
    err_sym = Libdl.dlsym(lib, :astra_julia_last_error)
    return unsafe_string(ccall(err_sym, Cstring, ()))
end

function _check_array(name::AbstractString, A::Array{Float32,N}) where {N}
    stride(A, 1) == 1 || throw(ArgumentError("$name must be column-major contiguous in the first dimension"))
    return A
end

function _check_vectors(name::AbstractString, vectors::Matrix{Float32}, angle_count::Integer)
    _check_array(name, vectors)
    size(vectors, 1) == 12 || throw(ArgumentError("$name must have size(vectors, 1) == 12"))
    size(vectors, 2) == angle_count || throw(ArgumentError("size($name, 2) must equal angle_count"))
    return vectors
end

function _check_cyl_vectors(name::AbstractString, vectors::Matrix{Float32}, angle_count::Integer)
    _check_array(name, vectors)
    size(vectors, 1) == 13 || throw(ArgumentError("$name must have size(vectors, 1) == 13"))
    size(vectors, 2) == angle_count || throw(ArgumentError("size($name, 2) must equal angle_count"))
    return vectors
end

function _check_vectors2d(name::AbstractString, vectors::Matrix{Float32}, angle_count::Integer)
    _check_array(name, vectors)
    size(vectors, 1) == 6 || throw(ArgumentError("$name must have size(vectors, 1) == 6"))
    size(vectors, 2) == angle_count || throw(ArgumentError("size($name, 2) must equal angle_count"))
    return vectors
end

function _prepare_custom_filter(custom_filter::Nothing)
    return (Ptr{Cfloat}(C_NULL), 0, 0)
end

function _prepare_custom_filter(custom_filter::Vector{Float32})
    _check_array("custom_filter", custom_filter)
    return (custom_filter, length(custom_filter), 1)
end

function _prepare_custom_filter(custom_filter::Matrix{Float32})
    _check_array("custom_filter", custom_filter)
    return (custom_filter, size(custom_filter, 1), size(custom_filter, 2))
end

function _prepare_custom_filter(custom_filter)
    throw(ArgumentError("custom_filter must be `nothing`, `Vector{Float32}`, or `Matrix{Float32}`"))
end

function _maybe_mask_ptr(mask::Nothing, _name::AbstractString, _dims)
    return Ptr{Cfloat}(C_NULL)
end

function _maybe_mask_ptr(mask::Array{Float32,2}, name::AbstractString, dims::Tuple{Int,Int})
    _check_array(name, mask)
    size(mask) == dims || throw(ArgumentError("$name must have size $dims"))
    return mask
end

function _maybe_mask_ptr(mask::Array{Float32,3}, name::AbstractString, dims::Tuple{Int,Int,Int})
    _check_array(name, mask)
    size(mask) == dims || throw(ArgumentError("$name must have size $dims"))
    return mask
end

function _maybe_residual_ptr(residual_norm::Nothing)
    return Ptr{Cfloat}(C_NULL)
end

function _maybe_residual_ptr(residual_norm::Base.RefValue{Float32})
    return residual_norm
end

function _maybe_residual_ptr(residual_norm)
    throw(ArgumentError("residual_norm must be `nothing` or `Ref{Float32}`"))
end

function _constraint_args(min_constraint, max_constraint)
    use_min = min_constraint === nothing ? 0 : 1
    use_max = max_constraint === nothing ? 0 : 1
    min_value = min_constraint === nothing ? 0.0f0 : Float32(min_constraint)
    max_value = max_constraint === nothing ? 0.0f0 : Float32(max_constraint)
    return (use_min, min_value, use_max, max_value)
end

function _kernel_code(kernel::Union{Symbol,Integer})
    kernel isa Integer && return Cint(kernel)
    kernel === :line && return Cint(0)
    kernel === :linear && return Cint(1)
    kernel === :strip && return Cint(2)
    kernel === :distance_driven && return Cint(3)
    throw(ArgumentError("projector_kernel must be :line, :linear, :strip, or :distance_driven (got $kernel)"))
end

function has_cuda()
    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_has_cuda)
    return ccall(sym, Cint, ()) != 0
end

function has_feature(feature::AbstractString)
    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_has_feature)
    return ccall(sym, Cint, (Cstring,), feature) != 0
end

function get_gpu_count()
    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_get_gpu_count)
    return Int(ccall(sym, Cint, ()))
end

function get_gpu_info(gpu_index::Integer = -1)
    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_get_gpu_info)
    ptr = ccall(sym, Cstring, (Cint,), Cint(gpu_index))
    ptr == C_NULL && return ""
    return unsafe_string(ptr)
end

function set_gpu_index(gpu_index::Integer)
    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_set_gpu_index)
    ok = ccall(sym, Cint, (Cint,), Cint(gpu_index)) != 0
    ok || error(last_error())
    return true
end

function get_gpu_index()
    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_get_gpu_index)
    return Int(ccall(sym, Cint, ()))
end

function set_volume_window_2d!(min_x::Real, min_y::Real, max_x::Real, max_y::Real)
    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_set_volume_window_2d)
    ok = ccall(sym, Cint, (Cfloat, Cfloat, Cfloat, Cfloat), Float32(min_x), Float32(min_y), Float32(max_x), Float32(max_y))
    ok != 0 || error("astra_julia_set_volume_window_2d failed: $(last_error())")
    return nothing
end

function clear_volume_window_2d!()
    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_clear_volume_window_2d)
    ccall(sym, Cint, ())
    return nothing
end

function get_volume_window_2d()
    out = Vector{Float32}(undef, 4)
    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_get_volume_window_2d)
    ok = ccall(sym, Cint, (Ptr{Cfloat},), out)
    return ok != 0 ? (out[1], out[2], out[3], out[4]) : nothing
end

function set_volume_window_3d!(min_x::Real, min_y::Real, min_z::Real, max_x::Real, max_y::Real, max_z::Real)
    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_set_volume_window_3d)
    ok = ccall(sym, Cint, (Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), Float32(min_x), Float32(min_y), Float32(min_z), Float32(max_x), Float32(max_y), Float32(max_z))
    ok != 0 || error("astra_julia_set_volume_window_3d failed: $(last_error())")
    return nothing
end

function clear_volume_window_3d!()
    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_clear_volume_window_3d)
    ccall(sym, Cint, ())
    return nothing
end

function get_volume_window_3d()
    out = Vector{Float32}(undef, 6)
    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_get_volume_window_3d)
    ok = ccall(sym, Cint, (Ptr{Cfloat},), out)
    return ok != 0 ? (out[1], out[2], out[3], out[4], out[5], out[6]) : nothing
end

function with_volume_window(f, vg)
    if vg === nothing
        return f()
    elseif hasproperty(vg, :window_min_z)
        if vg.window_min_x == -Float32(vg.width) / 2.0f0 &&
           vg.window_max_x == Float32(vg.width) / 2.0f0 &&
           vg.window_min_y == -Float32(vg.height) / 2.0f0 &&
           vg.window_max_y == Float32(vg.height) / 2.0f0 &&
           vg.window_min_z == -Float32(vg.depth) / 2.0f0 &&
           vg.window_max_z == Float32(vg.depth) / 2.0f0
            return f()
        end
        set_volume_window_3d!(vg.window_min_x, vg.window_min_y, vg.window_min_z, vg.window_max_x, vg.window_max_y, vg.window_max_z)
        try
            return f()
        finally
            clear_volume_window_3d!()
        end
    elseif hasproperty(vg, :window_min_x)
        # Default centered window: min = -w/2, max = w/2
        if vg.window_min_x == -Float32(vg.width) / 2.0f0 &&
           vg.window_max_x == Float32(vg.width) / 2.0f0 &&
           vg.window_min_y == -Float32(vg.height) / 2.0f0 &&
           vg.window_max_y == Float32(vg.height) / 2.0f0
            return f()
        end
        set_volume_window_2d!(vg.window_min_x, vg.window_min_y, vg.window_max_x, vg.window_max_y)
        try
            return f()
        finally
            clear_volume_window_2d!()
        end
    else
        return f()
    end
end

function cuda_dart_mask_2d_f32!(
    mask::Array{Float32,2},
    segmentation::Array{Float32,2};
    connectivity::Integer = 8,
    radius::Integer = 1,
    threshold::Integer = 1,
    gpu_index::Integer = -1,
)
    lib = _libhandle()
    w, h = size(segmentation)
    size(mask) == (w, h) || throw(DimensionMismatch("size(mask) must match size(segmentation)"))
    sym = Libdl.dlsym(lib, :astra_julia_cuda_dart_mask_2d_f32)
    status = ccall(
        sym,
        Cint,
        (Ptr{Cfloat}, Ptr{Cfloat}, Cuint, Cuint, Cuint, Cuint, Cuint, Cint),
        mask, segmentation, connectivity, radius, threshold, w, h, gpu_index,
    )
    status == 1 || throw(ErrorException("cuda_dart_mask_2d_f32! failed: $(last_error())"))
    return mask
end

function cuda_dart_smoothing_2d_f32!(
    out::Array{Float32,2},
    in::Array{Float32,2};
    b::Real = 0.1f0,
    radius::Integer = 1,
    gpu_index::Integer = -1,
)
    lib = _libhandle()
    w, h = size(in)
    size(out) == (w, h) || throw(DimensionMismatch("size(out) must match size(in)"))
    sym = Libdl.dlsym(lib, :astra_julia_cuda_dart_smoothing_2d_f32)
    status = ccall(
        sym,
        Cint,
        (Ptr{Cfloat}, Ptr{Cfloat}, Cfloat, Cuint, Cuint, Cuint, Cint),
        out, in, Float32(b), radius, w, h, gpu_index,
    )
    status == 1 || throw(ErrorException("cuda_dart_smoothing_2d_f32! failed: $(last_error())"))
    return out
end

function cuda_roi_select_2d_f32!(
    out::Array{Float32,2};
    radius::Real,
    gpu_index::Integer = -1,
)
    lib = _libhandle()
    w, h = size(out)
    sym = Libdl.dlsym(lib, :astra_julia_cuda_roi_select_2d_f32)
    status = ccall(
        sym,
        Cint,
        (Ptr{Cfloat}, Cfloat, Cuint, Cuint, Cint),
        out, Float32(radius), w, h, gpu_index,
    )
    status == 1 || throw(ErrorException("cuda_roi_select_2d_f32! failed: $(last_error())"))
    return out
end

function cuda_dart_mask_3d_f32!(
    mask::Array{Float32,3},
    segmentation::Array{Float32,3};
    connectivity::Integer = 26,
    radius::Integer = 1,
    threshold::Integer = 1,
    gpu_index::Integer = -1,
)
    lib = _libhandle()
    w, h, d = size(segmentation)
    size(mask) == (w, h, d) || throw(DimensionMismatch("size(mask) must match size(segmentation)"))
    sym = Libdl.dlsym(lib, :astra_julia_cuda_dart_mask_3d_f32)
    status = ccall(
        sym,
        Cint,
        (Ptr{Cfloat}, Ptr{Cfloat}, Cuint, Cuint, Cuint, Cuint, Cuint, Cuint, Cint),
        mask, segmentation, connectivity, radius, threshold, w, h, d, gpu_index,
    )
    status == 1 || throw(ErrorException("cuda_dart_mask_3d_f32! failed: $(last_error())"))
    return mask
end

function cuda_dart_smoothing_3d_f32!(
    out::Array{Float32,3},
    in::Array{Float32,3};
    b::Real = 0.1f0,
    radius::Integer = 1,
    gpu_index::Integer = -1,
)
    lib = _libhandle()
    w, h, d = size(in)
    size(out) == (w, h, d) || throw(DimensionMismatch("size(out) must match size(in)"))
    sym = Libdl.dlsym(lib, :astra_julia_cuda_dart_smoothing_3d_f32)
    status = ccall(
        sym,
        Cint,
        (Ptr{Cfloat}, Ptr{Cfloat}, Cfloat, Cuint, Cuint, Cuint, Cuint, Cint),
        out, in, Float32(b), radius, w, h, d, gpu_index,
    )
    status == 1 || throw(ErrorException("cuda_dart_smoothing_3d_f32! failed: $(last_error())"))
    return out
end
