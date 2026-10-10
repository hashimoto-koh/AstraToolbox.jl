module LowLevel

using Libdl

const _deps_path = joinpath(@__DIR__, "..", "deps", "deps.jl")
if isfile(_deps_path)
    include(_deps_path)
end

export set_library!,
       library_path,
       FILTER_ERROR,
       FILTER_NONE,
       FILTER_RAMLAK,
       FILTER_SHEPPLOGAN,
       FILTER_COSINE,
       FILTER_HAMMING,
       FILTER_HANN,
       FILTER_TUKEY,
       FILTER_LANCZOS,
       FILTER_TRIANGULAR,
       FILTER_GAUSSIAN,
       FILTER_BARTLETTHANN,
       FILTER_BLACKMAN,
       FILTER_NUTTALL,
       FILTER_BLACKMANHARRIS,
       FILTER_BLACKMANNUTTALL,
       FILTER_FLATTOP,
       FILTER_KAISER,
       FILTER_PARZEN,
       FILTER_PROJECTION,
       FILTER_SINOGRAM,
       FILTER_RPROJECTION,
       FILTER_RSINOGRAM,
       fp2d_parallel_line_f32!,
       cuda_fp2d_parallel_line_f32!,
       bp2d_parallel_line_f32!,
       cuda_bp2d_parallel_f32!,
       fp2d_parallel_vec_line_f32!,
       cuda_fp2d_parallel_vec_f32!,
       bp2d_parallel_vec_line_f32!,
       cuda_bp2d_parallel_vec_f32!,
       fp2d_fanflat_line_f32!,
       cuda_fp2d_fanflat_f32!,
       bp2d_fanflat_line_f32!,
       cuda_bp2d_fanflat_f32!,
       fp2d_fanflat_vec_line_f32!,
       cuda_fp2d_fanflat_vec_f32!,
       bp2d_fanflat_vec_line_f32!,
       cuda_bp2d_fanflat_vec_f32!,
       fbp2d_parallel_f32!,
       cuda_fbp2d_parallel_f32!,
       cuda_fbp2d_parallel_vec_f32!,
       cuda_fbp2d_fanflat_f32!,
       cuda_fbp2d_fanflat_vec_f32!,
       cuda_sart2d_parallel_f32!,
       cuda_sirt2d_parallel_f32!,
       cuda_sart2d_parallel_vec_f32!,
       cuda_sirt2d_parallel_vec_f32!,
       cuda_sart2d_fanflat_f32!,
       cuda_sirt2d_fanflat_f32!,
       cuda_sart2d_fanflat_vec_f32!,
       cuda_sirt2d_fanflat_vec_f32!,
       cuda_cgls2d_parallel_f32!,
       cuda_cgls2d_parallel_vec_f32!,
       cuda_cgls2d_fanflat_f32!,
       cuda_cgls2d_fanflat_vec_f32!,
       cuda_em2d_parallel_f32!,
       cuda_fp3d_parallel_f32!,
       cuda_bp3d_parallel_f32!,
       cuda_sirt3d_parallel_f32!,
       cuda_cgls3d_parallel_f32!,
       cuda_fp3d_parallel_vec_f32!,
       cuda_bp3d_parallel_vec_f32!,
       cuda_sirt3d_parallel_vec_f32!,
       cuda_cgls3d_parallel_vec_f32!,
       cuda_fp3d_cone_f32!,
       cuda_bp3d_cone_f32!,
       cuda_sirt3d_cone_f32!,
       cuda_cgls3d_cone_f32!,
       cuda_fdk3d_cone_f32!,
       cuda_fp3d_cone_vec_f32!,
       cuda_bp3d_cone_vec_f32!,
       cuda_fp3d_cyl_cone_vec_f32!,
       cuda_bp3d_cyl_cone_vec_f32!,
       cuda_sirt3d_cone_vec_f32!,
       cuda_sirt3d_cyl_cone_vec_f32!,
       cuda_cgls3d_cone_vec_f32!,
       cuda_cgls3d_cyl_cone_vec_f32!,
       cuda_fdk3d_cone_vec_f32!,
       has_cuda,
       has_feature,
       get_gpu_count,
       get_gpu_info,
       set_gpu_index,
       get_gpu_index,
       fp2d_parallel_linear_f32!,
       bp2d_parallel_linear_f32!,
       fp2d_parallel_vec_linear_f32!,
       bp2d_parallel_vec_linear_f32!,
       fp2d_parallel_strip_f32!,
       bp2d_parallel_strip_f32!,
       fp2d_parallel_vec_strip_f32!,
       bp2d_parallel_vec_strip_f32!,
       fp2d_fanflat_strip_f32!,
       bp2d_fanflat_strip_f32!,
       fp2d_parallel_distance_driven_f32!,
       bp2d_parallel_distance_driven_f32!,
       fp2d_parallel_vec_distance_driven_f32!,
       bp2d_parallel_vec_distance_driven_f32!,
       sirt2d_parallel_f32!,
       sirt2d_parallel_vec_f32!,
       sirt2d_fanflat_f32!,
       sirt2d_fanflat_vec_f32!,
       sart2d_parallel_f32!,
       sart2d_parallel_vec_f32!,
       sart2d_fanflat_f32!,
       sart2d_fanflat_vec_f32!,
       cgls2d_parallel_f32!,
       cgls2d_parallel_vec_f32!,
       cgls2d_fanflat_f32!,
       cgls2d_fanflat_vec_f32!,
       art2d_parallel_f32!,
       art2d_parallel_vec_f32!,
       art2d_fanflat_f32!,
       art2d_fanflat_vec_f32!,
       fp2d_sparse_matrix_f32!,
       bp2d_sparse_matrix_f32!,
       sirt2d_sparse_matrix_f32!,
       sart2d_sparse_matrix_f32!,
       cgls2d_sparse_matrix_f32!,
       art2d_sparse_matrix_f32!,
       set_volume_window_2d!,
       clear_volume_window_2d!,
       get_volume_window_2d,
       set_volume_window_3d!,
       clear_volume_window_3d!,
       get_volume_window_3d,
       cuda_dart_mask_2d_f32!,
       cuda_dart_smoothing_2d_f32!,
       cuda_roi_select_2d_f32!,
       cuda_dart_mask_3d_f32!,
       cuda_dart_smoothing_3d_f32!,
       VolumePart3DC,
       ProjectionPart3DC,
       cuda_composite_fp3d_f32!,
       cuda_composite_bp3d_f32!,
       project_point_3d,
       projected_bbox_3d,
       last_error

const _library_path = Ref{String}("")

const FILTER_ERROR = 0
const FILTER_NONE = 1
const FILTER_RAMLAK = 2
const FILTER_SHEPPLOGAN = 3
const FILTER_COSINE = 4
const FILTER_HAMMING = 5
const FILTER_HANN = 6
const FILTER_TUKEY = 7
const FILTER_LANCZOS = 8
const FILTER_TRIANGULAR = 9
const FILTER_GAUSSIAN = 10
const FILTER_BARTLETTHANN = 11
const FILTER_BLACKMAN = 12
const FILTER_NUTTALL = 13
const FILTER_BLACKMANHARRIS = 14
const FILTER_BLACKMANNUTTALL = 15
const FILTER_FLATTOP = 16
const FILTER_KAISER = 17
const FILTER_PARZEN = 18
const FILTER_PROJECTION = 19
const FILTER_SINOGRAM = 20
const FILTER_RPROJECTION = 21
const FILTER_RSINOGRAM = 22


include("lowlevel_common.jl")
include("lowlevel_2d.jl")
include("lowlevel_3d.jl")

end # module LowLevel
