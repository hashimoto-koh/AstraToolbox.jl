function cuda_fp3d_parallel_f32!(
    sinogram::Array{Float32,3},
    volume::Array{Float32,3},
    angles::Vector{Float32};
    detector_spacing_u::Float32 = 1.0f0,
    detector_spacing_v::Float32 = 1.0f0,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_cuda_fp3d_parallel_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        length(angles),
        size(sinogram, 1),
        size(sinogram, 3),
        detector_spacing_u,
        detector_spacing_v,
        angles,
        volume,
        sinogram,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_fp3d_parallel_f32 failed: $(last_error())")
    return sinogram
end

function cuda_bp3d_parallel_f32!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    angles::Vector{Float32};
    detector_spacing_u::Float32 = 1.0f0,
    detector_spacing_v::Float32 = 1.0f0,
    voxel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_cuda_bp3d_parallel_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        length(angles),
        size(sinogram, 1),
        size(sinogram, 3),
        detector_spacing_u,
        detector_spacing_v,
        angles,
        sinogram,
        volume,
        voxel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_bp3d_parallel_f32 failed: $(last_error())")
    return volume
end

function cuda_fp3d_parallel_vec_f32!(
    sinogram::Array{Float32,3},
    volume::Array{Float32,3},
    vectors::Matrix{Float32};
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_cuda_fp3d_parallel_vec_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        size(sinogram, 2),
        size(sinogram, 1),
        size(sinogram, 3),
        vectors,
        volume,
        sinogram,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_fp3d_parallel_vec_f32 failed: $(last_error())")
    return sinogram
end

function cuda_bp3d_parallel_vec_f32!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    vectors::Matrix{Float32};
    voxel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_cuda_bp3d_parallel_vec_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        size(sinogram, 2),
        size(sinogram, 1),
        size(sinogram, 3),
        vectors,
        sinogram,
        volume,
        voxel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_bp3d_parallel_vec_f32 failed: $(last_error())")
    return volume
end

function cuda_fp3d_cone_f32!(
    sinogram::Array{Float32,3},
    volume::Array{Float32,3},
    angles::Vector{Float32};
    detector_spacing_u::Float32 = 1.0f0,
    detector_spacing_v::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
    detector_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_cuda_fp3d_cone_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        length(angles),
        size(sinogram, 1),
        size(sinogram, 3),
        detector_spacing_u,
        detector_spacing_v,
        origin_source_distance,
        origin_detector_distance,
        angles,
        volume,
        sinogram,
        detector_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_fp3d_cone_f32 failed: $(last_error())")
    return sinogram
end

function cuda_bp3d_cone_f32!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    angles::Vector{Float32};
    detector_spacing_u::Float32 = 1.0f0,
    detector_spacing_v::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
    voxel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_cuda_bp3d_cone_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        length(angles),
        size(sinogram, 1),
        size(sinogram, 3),
        detector_spacing_u,
        detector_spacing_v,
        origin_source_distance,
        origin_detector_distance,
        angles,
        sinogram,
        volume,
        voxel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_bp3d_cone_f32 failed: $(last_error())")
    return volume
end

function cuda_fp3d_cone_vec_f32!(
    sinogram::Array{Float32,3},
    volume::Array{Float32,3},
    vectors::Matrix{Float32};
    detector_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_cuda_fp3d_cone_vec_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        size(sinogram, 2),
        size(sinogram, 1),
        size(sinogram, 3),
        vectors,
        volume,
        sinogram,
        detector_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_fp3d_cone_vec_f32 failed: $(last_error())")
    return sinogram
end

function cuda_bp3d_cone_vec_f32!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    vectors::Matrix{Float32};
    voxel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_cuda_bp3d_cone_vec_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        size(sinogram, 2),
        size(sinogram, 1),
        size(sinogram, 3),
        vectors,
        sinogram,
        volume,
        voxel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_bp3d_cone_vec_f32 failed: $(last_error())")
    return volume
end

function cuda_fp3d_cyl_cone_vec_f32!(
    sinogram::Array{Float32,3},
    volume::Array{Float32,3},
    vectors::Matrix{Float32};
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_cyl_vectors("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    fp_sym = Libdl.dlsym(lib, :astra_julia_cuda_fp3d_cyl_cone_vec_f32)
    ok = ccall(
        fp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        size(sinogram, 2),
        size(sinogram, 1),
        size(sinogram, 3),
        vectors,
        volume,
        sinogram,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_fp3d_cyl_cone_vec_f32 failed: $(last_error())")
    return sinogram
end

function cuda_bp3d_cyl_cone_vec_f32!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    vectors::Matrix{Float32};
    voxel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_cyl_vectors("vectors", vectors, size(sinogram, 2))

    lib = _libhandle()
    bp_sym = Libdl.dlsym(lib, :astra_julia_cuda_bp3d_cyl_cone_vec_f32)
    ok = ccall(
        bp_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        size(sinogram, 2),
        size(sinogram, 1),
        size(sinogram, 3),
        vectors,
        sinogram,
        volume,
        voxel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_bp3d_cyl_cone_vec_f32 failed: $(last_error())")
    return volume
end

function cuda_sirt3d_parallel_f32!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    angles::Vector{Float32};
    detector_spacing_u::Float32 = 1.0f0,
    detector_spacing_v::Float32 = 1.0f0,
    iterations::Integer,
    relaxation::Float32 = 1.0f0,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
    voxel_supersampling::Integer = 1,
    detector_supersampling::Integer = 1,
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
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_sirt3d_parallel_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        length(angles),
        size(sinogram, 1),
        size(sinogram, 3),
        detector_spacing_u,
        detector_spacing_v,
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
        voxel_supersampling,
        detector_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_sirt3d_parallel_f32 failed: $(last_error())")
    return volume
end

function cuda_sirt3d_parallel_vec_f32!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    vectors::Matrix{Float32};
    iterations::Integer,
    relaxation::Float32 = 1.0f0,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
    voxel_supersampling::Integer = 1,
    detector_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)
    use_min_constraint, min_constraint_value, use_max_constraint, max_constraint_value =
        _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_sirt3d_parallel_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        size(sinogram, 2),
        size(sinogram, 1),
        size(sinogram, 3),
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
        voxel_supersampling,
        detector_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_sirt3d_parallel_vec_f32 failed: $(last_error())")
    return volume
end

function cuda_sirt3d_cone_f32!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    angles::Vector{Float32};
    detector_spacing_u::Float32 = 1.0f0,
    detector_spacing_v::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
    iterations::Integer,
    relaxation::Float32 = 1.0f0,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
    voxel_supersampling::Integer = 1,
    detector_supersampling::Integer = 1,
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
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_sirt3d_cone_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        length(angles),
        size(sinogram, 1),
        size(sinogram, 3),
        detector_spacing_u,
        detector_spacing_v,
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
        voxel_supersampling,
        detector_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_sirt3d_cone_f32 failed: $(last_error())")
    return volume
end

function cuda_sirt3d_cone_vec_f32!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    vectors::Matrix{Float32};
    iterations::Integer,
    relaxation::Float32 = 1.0f0,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
    voxel_supersampling::Integer = 1,
    detector_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)
    use_min_constraint, min_constraint_value, use_max_constraint, max_constraint_value =
        _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_sirt3d_cone_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        size(sinogram, 2),
        size(sinogram, 1),
        size(sinogram, 3),
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
        voxel_supersampling,
        detector_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_sirt3d_cone_vec_f32 failed: $(last_error())")
    return volume
end

function cuda_sirt3d_cyl_cone_vec_f32!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    vectors::Matrix{Float32};
    iterations::Integer,
    relaxation::Float32 = 1.0f0,
    reconstruction_mask = nothing,
    sinogram_mask = nothing,
    residual_norm = nothing,
    min_constraint = nothing,
    max_constraint = nothing,
    voxel_supersampling::Integer = 1,
    detector_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_cyl_vectors("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    sinogram_mask_ptr = _maybe_mask_ptr(sinogram_mask, "sinogram_mask", size(sinogram))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)
    use_min_constraint, min_constraint_value, use_max_constraint, max_constraint_value =
        _constraint_args(min_constraint, max_constraint)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_sirt3d_cyl_cone_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Cint, Cfloat, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        size(sinogram, 2),
        size(sinogram, 1),
        size(sinogram, 3),
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
        voxel_supersampling,
        detector_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_sirt3d_cyl_cone_vec_f32 failed: $(last_error())")
    return volume
end

function cuda_cgls3d_parallel_f32!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    angles::Vector{Float32};
    detector_spacing_u::Float32 = 1.0f0,
    detector_spacing_v::Float32 = 1.0f0,
    iterations::Integer,
    reconstruction_mask = nothing,
    residual_norm = nothing,
    voxel_supersampling::Integer = 1,
    detector_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_cgls3d_parallel_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        length(angles),
        size(sinogram, 1),
        size(sinogram, 3),
        detector_spacing_u,
        detector_spacing_v,
        angles,
        sinogram,
        volume,
        iterations,
        reconstruction_mask_ptr,
        residual_norm_ptr,
        voxel_supersampling,
        detector_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_cgls3d_parallel_f32 failed: $(last_error())")
    return volume
end

function cuda_cgls3d_parallel_vec_f32!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    vectors::Matrix{Float32};
    iterations::Integer,
    reconstruction_mask = nothing,
    residual_norm = nothing,
    voxel_supersampling::Integer = 1,
    detector_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_cgls3d_parallel_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        size(sinogram, 2),
        size(sinogram, 1),
        size(sinogram, 3),
        vectors,
        sinogram,
        volume,
        iterations,
        reconstruction_mask_ptr,
        residual_norm_ptr,
        voxel_supersampling,
        detector_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_cgls3d_parallel_vec_f32 failed: $(last_error())")
    return volume
end

function cuda_cgls3d_cone_f32!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    angles::Vector{Float32};
    detector_spacing_u::Float32 = 1.0f0,
    detector_spacing_v::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
    iterations::Integer,
    reconstruction_mask = nothing,
    residual_norm = nothing,
    voxel_supersampling::Integer = 1,
    detector_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_cgls3d_cone_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        length(angles),
        size(sinogram, 1),
        size(sinogram, 3),
        detector_spacing_u,
        detector_spacing_v,
        origin_source_distance,
        origin_detector_distance,
        angles,
        sinogram,
        volume,
        iterations,
        reconstruction_mask_ptr,
        residual_norm_ptr,
        voxel_supersampling,
        detector_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_cgls3d_cone_f32 failed: $(last_error())")
    return volume
end

function cuda_cgls3d_cone_vec_f32!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    vectors::Matrix{Float32};
    iterations::Integer,
    reconstruction_mask = nothing,
    residual_norm = nothing,
    voxel_supersampling::Integer = 1,
    detector_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_cgls3d_cone_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        size(sinogram, 2),
        size(sinogram, 1),
        size(sinogram, 3),
        vectors,
        sinogram,
        volume,
        iterations,
        reconstruction_mask_ptr,
        residual_norm_ptr,
        voxel_supersampling,
        detector_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_cgls3d_cone_vec_f32 failed: $(last_error())")
    return volume
end

function cuda_cgls3d_cyl_cone_vec_f32!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    vectors::Matrix{Float32};
    iterations::Integer,
    reconstruction_mask = nothing,
    residual_norm = nothing,
    voxel_supersampling::Integer = 1,
    detector_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_cyl_vectors("vectors", vectors, size(sinogram, 2))
    reconstruction_mask_ptr = _maybe_mask_ptr(reconstruction_mask, "reconstruction_mask", size(volume))
    residual_norm_ptr = _maybe_residual_ptr(residual_norm)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_cgls3d_cyl_cone_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        size(sinogram, 2),
        size(sinogram, 1),
        size(sinogram, 3),
        vectors,
        sinogram,
        volume,
        iterations,
        reconstruction_mask_ptr,
        residual_norm_ptr,
        voxel_supersampling,
        detector_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_cgls3d_cyl_cone_vec_f32 failed: $(last_error())")
    return volume
end

function cuda_fdk3d_cone_f32!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    angles::Vector{Float32};
    detector_spacing_u::Float32 = 1.0f0,
    detector_spacing_v::Float32 = 1.0f0,
    origin_source_distance::Float32,
    origin_detector_distance::Float32,
    filter_type::Integer = FILTER_RAMLAK,
    custom_filter = nothing,
    filter_d::Float32 = 1.0f0,
    filter_parameter::Float32 = -1.0f0,
    short_scan::Bool = false,
    voxel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    size(sinogram, 2) == length(angles) || throw(ArgumentError("size(sinogram, 2) must equal length(angles)"))
    custom_filter_ptr, custom_filter_width, custom_filter_height = _prepare_custom_filter(custom_filter)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_fdk3d_cone_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Cint, Cfloat, Cfloat, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        length(angles),
        size(sinogram, 1),
        size(sinogram, 3),
        detector_spacing_u,
        detector_spacing_v,
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
        voxel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_fdk3d_cone_f32 failed: $(last_error())")
    return volume
end

function cuda_fdk3d_cone_vec_f32!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    vectors::Matrix{Float32};
    filter_type::Integer = FILTER_RAMLAK,
    custom_filter = nothing,
    filter_d::Float32 = 1.0f0,
    filter_parameter::Float32 = -1.0f0,
    short_scan::Bool = false,
    voxel_supersampling::Integer = 1,
    gpu_index::Integer = -1,
)
    _check_array("volume", volume)
    _check_array("sinogram", sinogram)
    _check_vectors("vectors", vectors, size(sinogram, 2))
    custom_filter_ptr, custom_filter_width, custom_filter_height = _prepare_custom_filter(custom_filter)

    lib = _libhandle()
    alg_sym = Libdl.dlsym(lib, :astra_julia_cuda_fdk3d_cone_vec_f32)
    ok = ccall(
        alg_sym,
        Cint,
        (Cint, Cint, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Cint, Cfloat, Cfloat, Cint, Cint, Cint),
        size(volume, 1),
        size(volume, 2),
        size(volume, 3),
        size(sinogram, 2),
        size(sinogram, 1),
        size(sinogram, 3),
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
        voxel_supersampling,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_fdk3d_cone_vec_f32 failed: $(last_error())")
    return volume
end

struct VolumePart3DC
    data::Ptr{Float32}
    grid_col_count::Int32
    grid_row_count::Int32
    grid_slice_count::Int32
    min_x::Float32
    max_x::Float32
    min_y::Float32
    max_y::Float32
    min_z::Float32
    max_z::Float32
end

struct ProjectionPart3DC
    data::Ptr{Float32}
    geom_type::Int32
    proj_count::Int32
    det_row_count::Int32
    det_col_count::Int32
    det_width::Float32
    det_height::Float32
    src_origin::Float32
    origin_det::Float32
    cyl_radius::Float32
    angles::Ptr{Float32}
    vectors::Ptr{Float32}
end

function cuda_composite_fp3d_f32!(
    volumes::Vector{VolumePart3DC},
    projections::Vector{ProjectionPart3DC};
    mode::Integer = 1,
    gpu_index::Integer = -1,
)
    isempty(volumes) && throw(ArgumentError("volumes must not be empty"))
    isempty(projections) && throw(ArgumentError("projections must not be empty"))

    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_cuda_composite_fp3d_f32)
    ok = ccall(
        sym,
        Cint,
        (Cint, Ptr{VolumePart3DC}, Cint, Ptr{ProjectionPart3DC}, Cint, Cint),
        length(volumes),
        volumes,
        length(projections),
        projections,
        mode,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_composite_fp3d_f32 failed: $(last_error())")
    return projections
end

function cuda_composite_bp3d_f32!(
    volumes::Vector{VolumePart3DC},
    projections::Vector{ProjectionPart3DC};
    mode::Integer = 1,
    gpu_index::Integer = -1,
)
    isempty(volumes) && throw(ArgumentError("volumes must not be empty"))
    isempty(projections) && throw(ArgumentError("projections must not be empty"))

    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_cuda_composite_bp3d_f32)
    ok = ccall(
        sym,
        Cint,
        (Cint, Ptr{VolumePart3DC}, Cint, Ptr{ProjectionPart3DC}, Cint, Cint),
        length(volumes),
        volumes,
        length(projections),
        projections,
        mode,
        gpu_index,
    )
    ok != 0 || error("astra_julia_cuda_composite_bp3d_f32 failed: $(last_error())")
    return volumes
end

function project_point_3d(proj::ProjectionPart3DC, x::Real, y::Real, z::Real, angle_idx::Integer)
    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_project_point_3d)
    out_u = Ref{Float64}(0.0)
    out_v = Ref{Float64}(0.0)
    ok = ccall(
        sym,
        Cint,
        (Ref{ProjectionPart3DC}, Cdouble, Cdouble, Cdouble, Cint, Ptr{Cdouble}, Ptr{Cdouble}),
        proj, Float64(x), Float64(y), Float64(z), Cint(angle_idx), out_u, out_v,
    )
    ok != 0 || error("astra_julia_project_point_3d failed: $(last_error())")
    return (out_u[], out_v[])
end

function projected_bbox_3d(
    proj::ProjectionPart3DC,
    min_x::Real, max_x::Real,
    min_y::Real, max_y::Real,
    min_z::Real, max_z::Real,
    angle_idx::Integer = -1,
)
    lib = _libhandle()
    sym = Libdl.dlsym(lib, :astra_julia_projected_bbox_3d)
    out_min_u = Ref{Float64}(0.0)
    out_max_u = Ref{Float64}(0.0)
    out_min_v = Ref{Float64}(0.0)
    out_max_v = Ref{Float64}(0.0)
    ok = ccall(
        sym,
        Cint,
        (Ref{ProjectionPart3DC}, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
        proj, Float64(min_x), Float64(max_x), Float64(min_y), Float64(max_y), Float64(min_z), Float64(max_z), Cint(angle_idx), out_min_u, out_max_u, out_min_v, out_max_v,
    )
    ok != 0 || error("astra_julia_projected_bbox_3d failed: $(last_error())")
    return (min_u = out_min_u[], max_u = out_max_u[], min_v = out_min_v[], max_v = out_max_v[])
end


