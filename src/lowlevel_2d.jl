function fp2d_parallel_line_f32!(
    sinogram::Array{Float32,2},
    volume::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_fp2d_parallel_line_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        volume,
        sinogram,
    )
    ok != 0 || error("astra_julia_fp2d_parallel_line_f32 failed: $(last_error())")
    return sinogram
end

function cuda_fp2d_parallel_line_f32!(
    sinogram::Array{Float32,2},
    volume::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    detector_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_cuda_fp2d_parallel_line_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        volume,
        sinogram,
        detector_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_fp2d_parallel_line_f32 failed: $(last_error())")
    return sinogram
end

function bp2d_parallel_line_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_bp2d_parallel_line_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        sinogram,
        volume,
    )
    ok != 0 || error("astra_julia_bp2d_parallel_line_f32 failed: $(last_error())")
    return volume
end

function cuda_bp2d_parallel_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_cuda_bp2d_parallel_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        sinogram,
        volume,
        detector_supersampling,
        pixel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_bp2d_parallel_f32 failed: $(last_error())")
    return volume
end

function fp2d_parallel_vec_line_f32!(
    sinogram::Array{Float32,2},
    volume::Array{Float32,2},
    vectors::Matrix{Float32},
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_fp2d_parallel_vec_line_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        volume,
        sinogram,
    )
    ok != 0 || error("astra_julia_fp2d_parallel_vec_line_f32 failed: $(last_error())")
    return sinogram
end

function cuda_fp2d_parallel_vec_f32!(
    sinogram::Array{Float32,2},
    volume::Array{Float32,2},
    vectors::Matrix{Float32};
    detector_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_cuda_fp2d_parallel_vec_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        volume,
        sinogram,
        detector_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_fp2d_parallel_vec_f32 failed: $(last_error())")
    return sinogram
end

function bp2d_parallel_vec_line_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32},
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_bp2d_parallel_vec_line_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
    )
    ok != 0 || error("astra_julia_bp2d_parallel_vec_line_f32 failed: $(last_error())")
    return volume
end

function cuda_bp2d_parallel_vec_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32};
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_cuda_bp2d_parallel_vec_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
        detector_supersampling,
        pixel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_bp2d_parallel_vec_f32 failed: $(last_error())")
    return volume
end

function fp2d_fanflat_line_f32!(
    sinogram::Array{Float32,2},
    volume::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_fp2d_fanflat_line_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        origin_source_distance,
        origin_detector_distance,
        angles,
        volume,
        sinogram,
    )
    ok != 0 || error("astra_julia_fp2d_fanflat_line_f32 failed: $(last_error())")
    return sinogram
end

function bp2d_fanflat_line_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_bp2d_fanflat_line_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        origin_source_distance,
        origin_detector_distance,
        angles,
        sinogram,
        volume,
    )
    ok != 0 || error("astra_julia_bp2d_fanflat_line_f32 failed: $(last_error())")
    return volume
end

function cuda_fp2d_fanflat_f32!(
    sinogram::Array{Float32,2},
    volume::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
    detector_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_cuda_fp2d_fanflat_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        origin_source_distance,
        origin_detector_distance,
        angles,
        volume,
        sinogram,
        detector_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_fp2d_fanflat_f32 failed: $(last_error())")
    return sinogram
end

function cuda_bp2d_fanflat_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_cuda_bp2d_fanflat_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        origin_source_distance,
        origin_detector_distance,
        angles,
        sinogram,
        volume,
        detector_supersampling,
        pixel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_bp2d_fanflat_f32 failed: $(last_error())")
    return volume
end

function fp2d_fanflat_vec_line_f32!(
    sinogram::Array{Float32,2},
    volume::Array{Float32,2},
    vectors::Matrix{Float32},
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_fp2d_fanflat_vec_line_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        volume,
        sinogram,
    )
    ok != 0 || error("astra_julia_fp2d_fanflat_vec_line_f32 failed: $(last_error())")
    return sinogram
end

function cuda_fp2d_fanflat_vec_f32!(
    sinogram::Array{Float32,2},
    volume::Array{Float32,2},
    vectors::Matrix{Float32};
    detector_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_cuda_fp2d_fanflat_vec_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        volume,
        sinogram,
        detector_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_fp2d_fanflat_vec_f32 failed: $(last_error())")
    return sinogram
end

function bp2d_fanflat_vec_line_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32},
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_bp2d_fanflat_vec_line_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
    )
    ok != 0 || error("astra_julia_bp2d_fanflat_vec_line_f32 failed: $(last_error())")
    return volume
end

function cuda_bp2d_fanflat_vec_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32};
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_cuda_bp2d_fanflat_vec_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
        detector_supersampling,
        pixel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_bp2d_fanflat_vec_f32 failed: $(last_error())")
    return volume
end

function fbp2d_parallel_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    filter_type::Integer = FILTER_RAMLAK,
    filter_d::Float32 = 1.0f0,
    filter_parameter::Float32 = -1.0f0,
    custom_filter = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    custom_filter_ptr, custom_filter_width, custom_filter_height = _prepare_custom_filter(custom_filter)

    lib = _libhandle()
    fbp_sym = Libdl.dlsym(lib, :astra_julia_fbp2d_parallel_f32)
    ok = ccall(
        fbp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Cint, Cfloat, Cfloat),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        sinogram,
        volume,
        filter_type,
        custom_filter_ptr,
        custom_filter_width,
        custom_filter_height,
        filter_d,
        filter_parameter,
    )
    ok != 0 || error("astra_julia_fbp2d_parallel_f32 failed: $(last_error())")
    return volume
end

function cuda_fbp2d_parallel_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    filter_type::Integer = FILTER_RAMLAK,
    filter_d::Float32 = 1.0f0,
    filter_parameter::Float32 = -1.0f0,
    custom_filter = nothing,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    custom_filter_ptr, custom_filter_width, custom_filter_height = _prepare_custom_filter(custom_filter)

    lib = _libhandle()
    fbp_sym = Libdl.dlsym(lib, :astra_julia_cuda_fbp2d_parallel_f32)
    ok = ccall(
        fbp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Cint, Cfloat, Cfloat, Cint),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        sinogram,
        volume,
        filter_type,
        custom_filter_ptr,
        custom_filter_width,
        custom_filter_height,
        filter_d,
        filter_parameter,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_fbp2d_parallel_f32 failed: $(last_error())")
    return volume
end

function cuda_fbp2d_parallel_vec_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32};
    filter_type::Integer = FILTER_RAMLAK,
    filter_d::Float32 = 1.0f0,
    filter_parameter::Float32 = -1.0f0,
    custom_filter = nothing,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))
    custom_filter_ptr, custom_filter_width, custom_filter_height = _prepare_custom_filter(custom_filter)

    lib = _libhandle()
    fbp_sym = Libdl.dlsym(lib, :astra_julia_cuda_fbp2d_parallel_vec_f32)
    ok = ccall(
        fbp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Cint, Cfloat, Cfloat, Cint),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
        filter_type,
        custom_filter_ptr,
        custom_filter_width,
        custom_filter_height,
        filter_d,
        filter_parameter,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_fbp2d_parallel_vec_f32 failed: $(last_error())")
    return volume
end

function cuda_fbp2d_fanflat_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
    filter_type::Integer = FILTER_RAMLAK,
    filter_d::Float32 = 1.0f0,
    filter_parameter::Float32 = -1.0f0,
    custom_filter = nothing,
    short_scan::Bool = false,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    custom_filter_ptr, custom_filter_width, custom_filter_height = _prepare_custom_filter(custom_filter)

    lib = _libhandle()
    fbp_sym = Libdl.dlsym(lib, :astra_julia_cuda_fbp2d_fanflat_f32)
    ok = ccall(
        fbp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Cint, Cfloat, Cfloat, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        origin_source_distance,
        origin_detector_distance,
        angles,
        sinogram,
        volume,
        filter_type,
        custom_filter_ptr,
        custom_filter_width,
        custom_filter_height,
        filter_d,
        filter_parameter,
        short_scan ? 1 : 0,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_fbp2d_fanflat_f32 failed: $(last_error())")
    return volume
end

function cuda_fbp2d_fanflat_vec_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32};
    filter_type::Integer = FILTER_RAMLAK,
    filter_d::Float32 = 1.0f0,
    filter_parameter::Float32 = -1.0f0,
    custom_filter = nothing,
    short_scan::Bool = false,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))
    custom_filter_ptr, custom_filter_width, custom_filter_height = _prepare_custom_filter(custom_filter)

    lib = _libhandle()
    fbp_sym = Libdl.dlsym(lib, :astra_julia_cuda_fbp2d_fanflat_vec_f32)
    ok = ccall(
        fbp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Cint, Cfloat, Cfloat, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
        filter_type,
        custom_filter_ptr,
        custom_filter_width,
        custom_filter_height,
        filter_d,
        filter_parameter,
        short_scan ? 1 : 0,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_fbp2d_fanflat_vec_f32 failed: $(last_error())")
    return volume
end

function cuda_sirt2d_parallel_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    iterations::Integer,
    relaxation::Float32 = 1.0f0,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)
    use_min_constraint, min_constraint_value, use_max_constraint, max_constraint_value =
        _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_sirt2d_parallel_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        sinogram,
        volume,
        iterations,
        relaxation,
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        residual_norm_ptr,
        use_min_constraint,
        min_constraint_value,
        use_max_constraint,
        max_constraint_value,
        detector_supersampling,
        pixel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_sirt2d_parallel_f32 failed: $(last_error())")
    return volume
end

function _projection_order_args(projection_order, projection_order_list, angle_count::Integer)
    if projection_order === :random
        projection_order_list === nothing || throw(ArgumentError("projection_order_list is only valid with projection_order = :custom"))
        return (0, Ptr{Cint}(C_NULL), 0)
    elseif projection_order === :sequential
        projection_order_list === nothing || throw(ArgumentError("projection_order_list is only valid with projection_order = :custom"))
        return (1, Ptr{Cint}(C_NULL), 0)
    elseif projection_order === :custom
        projection_order_list isa Vector{Int32} || throw(ArgumentError("projection_order_list must be `Vector{Int32}` when projection_order = :custom"))
        length(projection_order_list) > 0 || throw(ArgumentError("projection_order_list must be non-empty"))
        all(0 .<= projection_order_list .< angle_count) || throw(ArgumentError("projection_order_list entries must be in 0:(angle_count-1)"))
        return (2, projection_order_list, length(projection_order_list))
    else
        throw(ArgumentError("projection_order must be :random, :sequential, or :custom"))
    end
end

function cuda_sart2d_parallel_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    iterations::Integer,
    relaxation::Float32 = 1.0f0,
    reconstruction_mask = nothing,
    residual_norm = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
    projection_order::Symbol = :random,
    projection_order_list = nothing,
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)
    use_min_constraint, min_constraint_value, use_max_constraint, max_constraint_value =
        _constraint_args(min_constraint, max_constraint)
    projection_order_mode, projection_order_ptr, projection_order_length =
        _projection_order_args(projection_order, projection_order_list, length(angles))

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_sart2d_parallel_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat, Cint, Ptr{Cint}, Cint, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        sinogram,
        volume,
        iterations,
        relaxation,
        reconstruction_mask_ptr,
        residual_norm_ptr,
        use_min_constraint,
        min_constraint_value,
        use_max_constraint,
        max_constraint_value,
        projection_order_mode,
        projection_order_ptr,
        projection_order_length,
        detector_supersampling,
        pixel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_sart2d_parallel_f32 failed: $(last_error())")
    return volume
end

function cuda_cgls2d_parallel_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    iterations::Integer,
    reconstruction_mask = nothing,
    residual_norm = nothing,
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_cgls2d_parallel_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        sinogram,
        volume,
        iterations,
        reconstruction_mask_ptr,
        residual_norm_ptr,
        detector_supersampling,
        pixel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_cgls2d_parallel_f32 failed: $(last_error())")
    return volume
end

function cuda_em2d_parallel_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    iterations::Integer,
    residual_norm = nothing,
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_em2d_parallel_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        sinogram,
        volume,
        iterations,
        residual_norm_ptr,
        detector_supersampling,
        pixel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_em2d_parallel_f32 failed: $(last_error())")
    return volume
end

function cuda_sirt2d_parallel_vec_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32};
    iterations::Integer,
    relaxation::Float32 = 1.0f0,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)
    use_min_constraint, min_constraint_value, use_max_constraint, max_constraint_value =
        _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_sirt2d_parallel_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
        iterations,
        relaxation,
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        residual_norm_ptr,
        use_min_constraint,
        min_constraint_value,
        use_max_constraint,
        max_constraint_value,
        detector_supersampling,
        pixel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_sirt2d_parallel_vec_f32 failed: $(last_error())")
    return volume
end

function cuda_sart2d_parallel_vec_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32};
    iterations::Integer,
    relaxation::Float32 = 1.0f0,
    reconstruction_mask = nothing,
    residual_norm = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
    projection_order::Symbol = :random,
    projection_order_list = nothing,
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)
    use_min_constraint, min_constraint_value, use_max_constraint, max_constraint_value =
        _constraint_args(min_constraint, max_constraint)
    projection_order_mode, projection_order_ptr, projection_order_length =
        _projection_order_args(projection_order, projection_order_list, size(sinogram, 2))

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_sart2d_parallel_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat, Cint, Ptr{Cint}, Cint, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
        iterations,
        relaxation,
        reconstruction_mask_ptr,
        residual_norm_ptr,
        use_min_constraint,
        min_constraint_value,
        use_max_constraint,
        max_constraint_value,
        projection_order_mode,
        projection_order_ptr,
        projection_order_length,
        detector_supersampling,
        pixel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_sart2d_parallel_vec_f32 failed: $(last_error())")
    return volume
end

function cuda_sirt2d_fanflat_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
    iterations::Integer,
    relaxation::Float32 = 1.0f0,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)
    use_min_constraint, min_constraint_value, use_max_constraint, max_constraint_value =
        _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_sirt2d_fanflat_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        origin_source_distance,
        origin_detector_distance,
        angles,
        sinogram,
        volume,
        iterations,
        relaxation,
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        residual_norm_ptr,
        use_min_constraint,
        min_constraint_value,
        use_max_constraint,
        max_constraint_value,
        detector_supersampling,
        pixel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_sirt2d_fanflat_f32 failed: $(last_error())")
    return volume
end

function cuda_sart2d_fanflat_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
    iterations::Integer,
    relaxation::Float32 = 1.0f0,
    reconstruction_mask = nothing,
    residual_norm = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
    projection_order::Symbol = :random,
    projection_order_list = nothing,
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)
    use_min_constraint, min_constraint_value, use_max_constraint, max_constraint_value =
        _constraint_args(min_constraint, max_constraint)
    projection_order_mode, projection_order_ptr, projection_order_length =
        _projection_order_args(projection_order, projection_order_list, length(angles))

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_sart2d_fanflat_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat, Cint, Ptr{Cint}, Cint, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        origin_source_distance,
        origin_detector_distance,
        angles,
        sinogram,
        volume,
        iterations,
        relaxation,
        reconstruction_mask_ptr,
        residual_norm_ptr,
        use_min_constraint,
        min_constraint_value,
        use_max_constraint,
        max_constraint_value,
        projection_order_mode,
        projection_order_ptr,
        projection_order_length,
        detector_supersampling,
        pixel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_sart2d_fanflat_f32 failed: $(last_error())")
    return volume
end

function cuda_sirt2d_fanflat_vec_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32};
    iterations::Integer,
    relaxation::Float32 = 1.0f0,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)
    use_min_constraint, min_constraint_value, use_max_constraint, max_constraint_value =
        _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_sirt2d_fanflat_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
        iterations,
        relaxation,
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        residual_norm_ptr,
        use_min_constraint,
        min_constraint_value,
        use_max_constraint,
        max_constraint_value,
        detector_supersampling,
        pixel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_sirt2d_fanflat_vec_f32 failed: $(last_error())")
    return volume
end

function cuda_sart2d_fanflat_vec_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32};
    iterations::Integer,
    relaxation::Float32 = 1.0f0,
    reconstruction_mask = nothing,
    residual_norm = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
    projection_order::Symbol = :random,
    projection_order_list = nothing,
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)
    use_min_constraint, min_constraint_value, use_max_constraint, max_constraint_value =
        _constraint_args(min_constraint, max_constraint)
    projection_order_mode, projection_order_ptr, projection_order_length =
        _projection_order_args(projection_order, projection_order_list, size(sinogram, 2))

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_sart2d_fanflat_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat, Cint, Ptr{Cint}, Cint, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
        iterations,
        relaxation,
        reconstruction_mask_ptr,
        residual_norm_ptr,
        use_min_constraint,
        min_constraint_value,
        use_max_constraint,
        max_constraint_value,
        projection_order_mode,
        projection_order_ptr,
        projection_order_length,
        detector_supersampling,
        pixel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_sart2d_fanflat_vec_f32 failed: $(last_error())")
    return volume
end

function cuda_cgls2d_parallel_vec_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32};
    iterations::Integer,
    reconstruction_mask = nothing,
    residual_norm = nothing,
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_cgls2d_parallel_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
        iterations,
        reconstruction_mask_ptr,
        residual_norm_ptr,
        detector_supersampling,
        pixel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_cgls2d_parallel_vec_f32 failed: $(last_error())")
    return volume
end

function cuda_cgls2d_fanflat_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
    iterations::Integer,
    reconstruction_mask = nothing,
    residual_norm = nothing,
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_cgls2d_fanflat_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        origin_source_distance,
        origin_detector_distance,
        angles,
        sinogram,
        volume,
        iterations,
        reconstruction_mask_ptr,
        residual_norm_ptr,
        detector_supersampling,
        pixel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_cgls2d_fanflat_f32 failed: $(last_error())")
    return volume
end

function cuda_cgls2d_fanflat_vec_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32};
    iterations::Integer,
    reconstruction_mask = nothing,
    residual_norm = nothing,
    detector_supersampling::Integer = 1,
    pixel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_cgls2d_fanflat_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
        iterations,
        reconstruction_mask_ptr,
        residual_norm_ptr,
        detector_supersampling,
        pixel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_cgls2d_fanflat_vec_f32 failed: $(last_error())")
    return volume
end


# --- CPU Projection Kernels ---

function fp2d_parallel_linear_f32!(
    sinogram::Array{Float32,2},
    volume::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_fp2d_parallel_linear_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        volume,
        sinogram,
    )
    ok != 0 || error("astra_julia_fp2d_parallel_linear_f32 failed: $(last_error())")
    return sinogram
end

function bp2d_parallel_linear_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_bp2d_parallel_linear_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        sinogram,
        volume,
    )
    ok != 0 || error("astra_julia_bp2d_parallel_linear_f32 failed: $(last_error())")
    return volume
end

function fp2d_parallel_vec_linear_f32!(
    sinogram::Array{Float32,2},
    volume::Array{Float32,2},
    vectors::Matrix{Float32},
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_fp2d_parallel_vec_linear_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        volume,
        sinogram,
    )
    ok != 0 || error("astra_julia_fp2d_parallel_vec_linear_f32 failed: $(last_error())")
    return sinogram
end

function bp2d_parallel_vec_linear_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32},
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_bp2d_parallel_vec_linear_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
    )
    ok != 0 || error("astra_julia_bp2d_parallel_vec_linear_f32 failed: $(last_error())")
    return volume
end

function fp2d_parallel_strip_f32!(
    sinogram::Array{Float32,2},
    volume::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_fp2d_parallel_strip_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        volume,
        sinogram,
    )
    ok != 0 || error("astra_julia_fp2d_parallel_strip_f32 failed: $(last_error())")
    return sinogram
end

function bp2d_parallel_strip_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_bp2d_parallel_strip_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        sinogram,
        volume,
    )
    ok != 0 || error("astra_julia_bp2d_parallel_strip_f32 failed: $(last_error())")
    return volume
end

function fp2d_parallel_vec_strip_f32!(
    sinogram::Array{Float32,2},
    volume::Array{Float32,2},
    vectors::Matrix{Float32},
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_fp2d_parallel_vec_strip_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        volume,
        sinogram,
    )
    ok != 0 || error("astra_julia_fp2d_parallel_vec_strip_f32 failed: $(last_error())")
    return sinogram
end

function bp2d_parallel_vec_strip_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32},
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_bp2d_parallel_vec_strip_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
    )
    ok != 0 || error("astra_julia_bp2d_parallel_vec_strip_f32 failed: $(last_error())")
    return volume
end

function fp2d_fanflat_strip_f32!(
    sinogram::Array{Float32,2},
    volume::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_fp2d_fanflat_strip_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        origin_source_distance,
        origin_detector_distance,
        angles,
        volume,
        sinogram,
    )
    ok != 0 || error("astra_julia_fp2d_fanflat_strip_f32 failed: $(last_error())")
    return sinogram
end

function bp2d_fanflat_strip_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_bp2d_fanflat_strip_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        origin_source_distance,
        origin_detector_distance,
        angles,
        sinogram,
        volume,
    )
    ok != 0 || error("astra_julia_bp2d_fanflat_strip_f32 failed: $(last_error())")
    return volume
end

function fp2d_parallel_distance_driven_f32!(
    sinogram::Array{Float32,2},
    volume::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_fp2d_parallel_distance_driven_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        volume,
        sinogram,
    )
    ok != 0 || error("astra_julia_fp2d_parallel_distance_driven_f32 failed: $(last_error())")
    return sinogram
end

function bp2d_parallel_distance_driven_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_bp2d_parallel_distance_driven_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        sinogram,
        volume,
    )
    ok != 0 || error("astra_julia_bp2d_parallel_distance_driven_f32 failed: $(last_error())")
    return volume
end

function fp2d_parallel_vec_distance_driven_f32!(
    sinogram::Array{Float32,2},
    volume::Array{Float32,2},
    vectors::Matrix{Float32},
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_fp2d_parallel_vec_distance_driven_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        volume,
        sinogram,
    )
    ok != 0 || error("astra_julia_fp2d_parallel_vec_distance_driven_f32 failed: $(last_error())")
    return sinogram
end

function bp2d_parallel_vec_distance_driven_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32},
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_bp2d_parallel_vec_distance_driven_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
    )
    ok != 0 || error("astra_julia_bp2d_parallel_vec_distance_driven_f32 failed: $(last_error())")
    return volume
end

# --- CPU Iterative Reconstruction Algorithms ---

function sirt2d_parallel_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    iterations::Integer,
    relaxation::Real = 1.0f0,
    projector_kernel::Union{Symbol,Integer} = :line,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_sirt2d_parallel_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        sinogram,
        volume,
        iterations,
        Float32(relaxation),
        _kernel_code(projector_kernel),
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_sirt2d_parallel_f32 failed: $(last_error())")
    return volume
end

function sirt2d_parallel_vec_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32};
    iterations::Integer,
    relaxation::Real = 1.0f0,
    projector_kernel::Union{Symbol,Integer} = :line,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_sirt2d_parallel_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
        iterations,
        Float32(relaxation),
        _kernel_code(projector_kernel),
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_sirt2d_parallel_vec_f32 failed: $(last_error())")
    return volume
end

function sirt2d_fanflat_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
    iterations::Integer,
    relaxation::Real = 1.0f0,
    projector_kernel::Union{Symbol,Integer} = :line,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_sirt2d_fanflat_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        origin_source_distance,
        origin_detector_distance,
        angles,
        sinogram,
        volume,
        iterations,
        Float32(relaxation),
        _kernel_code(projector_kernel),
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_sirt2d_fanflat_f32 failed: $(last_error())")
    return volume
end

function sirt2d_fanflat_vec_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32};
    iterations::Integer,
    relaxation::Real = 1.0f0,
    projector_kernel::Union{Symbol,Integer} = :line,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_sirt2d_fanflat_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
        iterations,
        Float32(relaxation),
        _kernel_code(projector_kernel),
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_sirt2d_fanflat_vec_f32 failed: $(last_error())")
    return volume
end

function sart2d_parallel_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    iterations::Integer,
    relaxation::Real = 1.0f0,
    projection_order::Symbol = :random,
    projection_order_list = nothing,
    projector_kernel::Union{Symbol,Integer} = :line,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)
    order_mode, order_ptr, order_len = _projection_order_args(projection_order, projection_order_list, length(angles))

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_sart2d_parallel_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Ptr{Cint}, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        sinogram,
        volume,
        iterations,
        Float32(relaxation),
        order_mode,
        order_ptr,
        order_len,
        _kernel_code(projector_kernel),
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_sart2d_parallel_f32 failed: $(last_error())")
    return volume
end

function sart2d_parallel_vec_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32};
    iterations::Integer,
    relaxation::Real = 1.0f0,
    projection_order::Symbol = :random,
    projection_order_list = nothing,
    projector_kernel::Union{Symbol,Integer} = :line,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)
    order_mode, order_ptr, order_len = _projection_order_args(projection_order, projection_order_list, size(sinogram, 2))

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_sart2d_parallel_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Ptr{Cint}, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
        iterations,
        Float32(relaxation),
        order_mode,
        order_ptr,
        order_len,
        _kernel_code(projector_kernel),
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_sart2d_parallel_vec_f32 failed: $(last_error())")
    return volume
end

function sart2d_fanflat_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
    iterations::Integer,
    relaxation::Real = 1.0f0,
    projection_order::Symbol = :random,
    projection_order_list = nothing,
    projector_kernel::Union{Symbol,Integer} = :line,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)
    order_mode, order_ptr, order_len = _projection_order_args(projection_order, projection_order_list, length(angles))

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_sart2d_fanflat_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Ptr{Cint}, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        origin_source_distance,
        origin_detector_distance,
        angles,
        sinogram,
        volume,
        iterations,
        Float32(relaxation),
        order_mode,
        order_ptr,
        order_len,
        _kernel_code(projector_kernel),
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_sart2d_fanflat_f32 failed: $(last_error())")
    return volume
end

function sart2d_fanflat_vec_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32};
    iterations::Integer,
    relaxation::Real = 1.0f0,
    projection_order::Symbol = :random,
    projection_order_list = nothing,
    projector_kernel::Union{Symbol,Integer} = :line,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)
    order_mode, order_ptr, order_len = _projection_order_args(projection_order, projection_order_list, size(sinogram, 2))

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_sart2d_fanflat_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Ptr{Cint}, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
        iterations,
        Float32(relaxation),
        order_mode,
        order_ptr,
        order_len,
        _kernel_code(projector_kernel),
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_sart2d_fanflat_vec_f32 failed: $(last_error())")
    return volume
end

function cgls2d_parallel_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    iterations::Integer,
    projector_kernel::Union{Symbol,Integer} = :line,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cgls2d_parallel_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        sinogram,
        volume,
        iterations,
        _kernel_code(projector_kernel),
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_cgls2d_parallel_f32 failed: $(last_error())")
    return volume
end

function cgls2d_parallel_vec_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32};
    iterations::Integer,
    projector_kernel::Union{Symbol,Integer} = :line,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cgls2d_parallel_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
        iterations,
        _kernel_code(projector_kernel),
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_cgls2d_parallel_vec_f32 failed: $(last_error())")
    return volume
end

function cgls2d_fanflat_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
    iterations::Integer,
    projector_kernel::Union{Symbol,Integer} = :line,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cgls2d_fanflat_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        origin_source_distance,
        origin_detector_distance,
        angles,
        sinogram,
        volume,
        iterations,
        _kernel_code(projector_kernel),
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_cgls2d_fanflat_f32 failed: $(last_error())")
    return volume
end

function cgls2d_fanflat_vec_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32};
    iterations::Integer,
    projector_kernel::Union{Symbol,Integer} = :line,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cgls2d_fanflat_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
        iterations,
        _kernel_code(projector_kernel),
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_cgls2d_fanflat_vec_f32 failed: $(last_error())")
    return volume
end

function art2d_parallel_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    iterations::Integer,
    relaxation::Real = 1.0f0,
    projector_kernel::Union{Symbol,Integer} = :line,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_art2d_parallel_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        angles,
        sinogram,
        volume,
        iterations,
        Float32(relaxation),
        _kernel_code(projector_kernel),
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_art2d_parallel_f32 failed: $(last_error())")
    return volume
end

function art2d_parallel_vec_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32};
    iterations::Integer,
    relaxation::Real = 1.0f0,
    projector_kernel::Union{Symbol,Integer} = :line,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_art2d_parallel_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
        iterations,
        Float32(relaxation),
        _kernel_code(projector_kernel),
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_art2d_parallel_vec_f32 failed: $(last_error())")
    return volume
end

function art2d_fanflat_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    angles::Vector{Float32};
    detector_width::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
    iterations::Integer,
    relaxation::Real = 1.0f0,
    projector_kernel::Union{Symbol,Integer} = :line,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_art2d_fanflat_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        length(angles),
        size(sinogram, 1),
        detector_width,
        origin_source_distance,
        origin_detector_distance,
        angles,
        sinogram,
        volume,
        iterations,
        Float32(relaxation),
        _kernel_code(projector_kernel),
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_art2d_fanflat_f32 failed: $(last_error())")
    return volume
end

function art2d_fanflat_vec_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    vectors::Matrix{Float32};
    iterations::Integer,
    relaxation::Real = 1.0f0,
    projector_kernel::Union{Symbol,Integer} = :line,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors2d("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_art2d_fanflat_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        size(sinogram, 2),
        size(sinogram, 1),
        vectors,
        sinogram,
        volume,
        iterations,
        Float32(relaxation),
        _kernel_code(projector_kernel),
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_art2d_fanflat_vec_f32 failed: $(last_error())")
    return volume
end

# --- 2D Sparse Matrix Operations ---

function fp2d_sparse_matrix_f32!(
    sinogram::Array{Float32,2},
    volume::Array{Float32,2},
    row_starts::Vector{UInt64},
    col_indices::Vector{UInt32},
    values::Vector{Float32};
    angle_count::Integer = size(sinogram, 2),
    detector_count::Integer = size(sinogram, 1),
    nnz::Integer = length(values),
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 1) == detector_count || throw(ArgumentError("size(sinogram, 1) must equal detector_count"))
    size(sinogram, 2) == angle_count || throw(ArgumentError("size(sinogram, 2) must equal angle_count"))
    length(row_starts) == angle_count * detector_count + 1 ||
        throw(ArgumentError("length(row_starts) must equal angle_count * detector_count + 1"))
    length(col_indices) == nnz || throw(ArgumentError("length(col_indices) must equal nnz"))
    length(values) == nnz || throw(ArgumentError("length(values) must equal nnz"))

    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_fp2d_sparse_matrix_f32)
    ok = ccall(
        sym,
        Cint,
        (Cint, Cint, Cint, Cint, Culonglong, Ptr{Culonglong}, Ptr{Cuint}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        angle_count,
        detector_count,
        UInt64(nnz),
        row_starts,
        col_indices,
        values,
        volume,
        sinogram,
    )
    ok != 0 || error("astra_julia_fp2d_sparse_matrix_f32 failed: $(last_error())")
    return sinogram
end

function bp2d_sparse_matrix_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    row_starts::Vector{UInt64},
    col_indices::Vector{UInt32},
    values::Vector{Float32};
    angle_count::Integer = size(sinogram, 2),
    detector_count::Integer = size(sinogram, 1),
    nnz::Integer = length(values),
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 1) == detector_count || throw(ArgumentError("size(sinogram, 1) must equal detector_count"))
    size(sinogram, 2) == angle_count || throw(ArgumentError("size(sinogram, 2) must equal angle_count"))
    length(row_starts) == angle_count * detector_count + 1 ||
        throw(ArgumentError("length(row_starts) must equal angle_count * detector_count + 1"))
    length(col_indices) == nnz || throw(ArgumentError("length(col_indices) must equal nnz"))
    length(values) == nnz || throw(ArgumentError("length(values) must equal nnz"))

    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_bp2d_sparse_matrix_f32)
    ok = ccall(
        sym,
        Cint,
        (Cint, Cint, Cint, Cint, Culonglong, Ptr{Culonglong}, Ptr{Cuint}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
        size(volume, 1),
        size(volume, 2),
        angle_count,
        detector_count,
        UInt64(nnz),
        row_starts,
        col_indices,
        values,
        sinogram,
        volume,
    )
    ok != 0 || error("astra_julia_bp2d_sparse_matrix_f32 failed: $(last_error())")
    return volume
end

function sirt2d_sparse_matrix_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    row_starts::Vector{UInt64},
    col_indices::Vector{UInt32},
    values::Vector{Float32};
    angle_count::Integer = size(sinogram, 2),
    detector_count::Integer = size(sinogram, 1),
    nnz::Integer = length(values),
    iterations::Integer,
    relaxation::Real = 1.0f0,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 1) == detector_count || throw(ArgumentError("size(sinogram, 1) must equal detector_count"))
    size(sinogram, 2) == angle_count || throw(ArgumentError("size(sinogram, 2) must equal angle_count"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_sirt2d_sparse_matrix_f32)
    ok = ccall(
        sym,
        Cint,
        (Cint, Cint, Cint, Cint, Culonglong, Ptr{Culonglong}, Ptr{Cuint}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        angle_count,
        detector_count,
        UInt64(nnz),
        row_starts,
        col_indices,
        values,
        sinogram,
        volume,
        iterations,
        Float32(relaxation),
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_sirt2d_sparse_matrix_f32 failed: $(last_error())")
    return volume
end

function sart2d_sparse_matrix_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    row_starts::Vector{UInt64},
    col_indices::Vector{UInt32},
    values::Vector{Float32};
    angle_count::Integer = size(sinogram, 2),
    detector_count::Integer = size(sinogram, 1),
    nnz::Integer = length(values),
    iterations::Integer,
    relaxation::Real = 1.0f0,
    projection_order::Symbol = :random,
    projection_order_list::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 1) == detector_count || throw(ArgumentError("size(sinogram, 1) must equal detector_count"))
    size(sinogram, 2) == angle_count || throw(ArgumentError("size(sinogram, 2) must equal angle_count"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)

    order_mode, order_ptr, order_len = _projection_order_args(
        projection_order,
        projection_order_list,
        angle_count,
    )

    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_sart2d_sparse_matrix_f32)
    ok = ccall(
        sym,
        Cint,
        (Cint, Cint, Cint, Cint, Culonglong, Ptr{Culonglong}, Ptr{Cuint}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Ptr{Cint}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        angle_count,
        detector_count,
        UInt64(nnz),
        row_starts,
        col_indices,
        values,
        sinogram,
        volume,
        iterations,
        Float32(relaxation),
        order_mode,
        order_ptr,
        order_len,
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_sart2d_sparse_matrix_f32 failed: $(last_error())")
    return volume
end

function cgls2d_sparse_matrix_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    row_starts::Vector{UInt64},
    col_indices::Vector{UInt32},
    values::Vector{Float32};
    angle_count::Integer = size(sinogram, 2),
    detector_count::Integer = size(sinogram, 1),
    nnz::Integer = length(values),
    iterations::Integer,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 1) == detector_count || throw(ArgumentError("size(sinogram, 1) must equal detector_count"))
    size(sinogram, 2) == angle_count || throw(ArgumentError("size(sinogram, 2) must equal angle_count"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_cgls2d_sparse_matrix_f32)
    ok = ccall(
        sym,
        Cint,
        (Cint, Cint, Cint, Cint, Culonglong, Ptr{Culonglong}, Ptr{Cuint}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        angle_count,
        detector_count,
        UInt64(nnz),
        row_starts,
        col_indices,
        values,
        sinogram,
        volume,
        iterations,
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_cgls2d_sparse_matrix_f32 failed: $(last_error())")
    return volume
end

function art2d_sparse_matrix_f32!(
    volume::Array{Float32,2},
    sinogram::Array{Float32,2},
    row_starts::Vector{UInt64},
    col_indices::Vector{UInt32},
    values::Vector{Float32};
    angle_count::Integer = size(sinogram, 2),
    detector_count::Integer = size(sinogram, 1),
    nnz::Integer = length(values),
    iterations::Integer,
    relaxation::Real = 1.0f0,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 1) == detector_count || throw(ArgumentError("size(sinogram, 1) must equal detector_count"))
    size(sinogram, 2) == angle_count || throw(ArgumentError("size(sinogram, 2) must equal angle_count"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    use_min, min_val, use_max, max_val = _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_art2d_sparse_matrix_f32)
    ok = ccall(
        sym,
        Cint,
        (Cint, Cint, Cint, Cint, Culonglong, Ptr{Culonglong}, Ptr{Cuint}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat),
        size(volume, 1),
        size(volume, 2),
        angle_count,
        detector_count,
        UInt64(nnz),
        row_starts,
        col_indices,
        values,
        sinogram,
        volume,
        iterations,
        Float32(relaxation),
        reconstruction_mask_ptr,
        sinogram_mask_ptr,
        use_min,
        min_val,
        use_max,
        max_val,
    )
    ok != 0 || error("astra_julia_art2d_sparse_matrix_f32 failed: $(last_error())")
    return volume
end
