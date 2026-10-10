module AstraToolbox

using Random
using Distributions
using LinearAlgebra
using SparseArrays

include("LowLevel.jl")
using .LowLevel

export set_library!,
       library_path,
       last_error,
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
       AbstractBackend,
       CPUBackend,
       CUDABackend,
       cpu,
       cuda,
       AbstractProjectionOptions,
       ProjectionOptions2D,
       ProjectionOptions3D,
       AbstractBackprojectionOptions,
       BackprojectionOptions2D,
       BackprojectionOptions3D,
       AbstractGeometry,
       AbstractGeometry2D,
       AbstractGeometry3D,
       Parallel2DGeometry,
       ParallelVec2DGeometry,
       FanFlat2DGeometry,
       FanFlatVec2DGeometry,
       SparseMatrix2DGeometry,
       Sparse2DGeometry,
       Parallel3DGeometry,
       ParallelVec3DGeometry,
       Cone3DGeometry,
       ConeVec3DGeometry,
       CylConeVec3DGeometry,
       AbstractVolumeGeometry,
       VolumeGeometry2D,
       VolumeGeometry3D,
       create_vol_geom,
       pixel_size,
       voxel_size,
       center,
       AbstractMethod,
       AbstractReconstructionOptions,
       ReconstructionOptions2D,
       ReconstructionOptions3D,
       FBPMethod,
       SIRTMethod,
       SARTMethod,
       CGLSMethod,
       ARTMethod,
       EMMethod,
       SIRT3DMethod,
       CGLS3DMethod,
       FDK3DMethod,
       project,
       project!,
       backproject,
       backproject!,
       reconstruct,
       reconstruct!,
       add_noise_to_sino,
       add_noise_to_sino!,
       shepp_logan,
       shepp_logan!,
       shepp_logan_3d,
       geom_2vec,
       geom_postalignment,
       move_vol_geom,
       geom_size,
       sinogram_size,
       project_point,
       projected_bbox,
       AstraProjectorMap,
       clip_circle,
       clip_circle!,
       has_cuda,
       has_feature,
       get_gpu_count,
       get_gpu_info,
       set_gpu_index,
       get_gpu_index,
       dart_segment,
       dart_segment!,
       dart_mask,
       dart_mask!,
       dart_smoothing,
       dart_smoothing!,
       reconstruct_dart,
       CompositeVolumePart,
       CompositeProjectionPart,
       do_composite_FP,
       do_composite_FP!,
       do_composite_BP,
       do_composite_BP!,
       accumulate_FP!,
       accumulate_BP!,
       composite_project,
       composite_project!,
       composite_backproject,
       composite_backproject!

"""Error indicator for invalid filter type."""
const FILTER_ERROR = LowLevel.FILTER_ERROR
"""No filter (pure backprojection)."""
const FILTER_NONE = LowLevel.FILTER_NONE
"""Ram-Lak filter (standard for FBP)."""
const FILTER_RAMLAK = LowLevel.FILTER_RAMLAK
"""Shepp-Logan filter."""
const FILTER_SHEPPLOGAN = LowLevel.FILTER_SHEPPLOGAN
"""Cosine filter."""
const FILTER_COSINE = LowLevel.FILTER_COSINE
"""Hamming filter."""
const FILTER_HAMMING = LowLevel.FILTER_HAMMING
"""Hann filter."""
const FILTER_HANN = LowLevel.FILTER_HANN
"""Tukey filter."""
const FILTER_TUKEY = LowLevel.FILTER_TUKEY
"""Lanczos filter."""
const FILTER_LANCZOS = LowLevel.FILTER_LANCZOS
"""Triangular filter."""
const FILTER_TRIANGULAR = LowLevel.FILTER_TRIANGULAR
"""Gaussian filter."""
const FILTER_GAUSSIAN = LowLevel.FILTER_GAUSSIAN
"""Bartlett-Hann filter."""
const FILTER_BARTLETTHANN = LowLevel.FILTER_BARTLETTHANN
"""Blackman filter."""
const FILTER_BLACKMAN = LowLevel.FILTER_BLACKMAN
"""Nuttall filter."""
const FILTER_NUTTALL = LowLevel.FILTER_NUTTALL
"""Blackman-Harris filter."""
const FILTER_BLACKMANHARRIS = LowLevel.FILTER_BLACKMANHARRIS
"""Blackman-Nuttall filter."""
const FILTER_BLACKMANNUTTALL = LowLevel.FILTER_BLACKMANNUTTALL
"""Flat-top filter."""
const FILTER_FLATTOP = LowLevel.FILTER_FLATTOP
"""Kaiser filter."""
const FILTER_KAISER = LowLevel.FILTER_KAISER
"""Parzen filter."""
const FILTER_PARZEN = LowLevel.FILTER_PARZEN
"""Projection-domain filter."""
const FILTER_PROJECTION = LowLevel.FILTER_PROJECTION
"""Sinogram-domain filter."""
const FILTER_SINOGRAM = LowLevel.FILTER_SINOGRAM
"""Radial projection-domain filter."""
const FILTER_RPROJECTION = LowLevel.FILTER_RPROJECTION
"""Radial sinogram-domain filter."""
const FILTER_RSINOGRAM = LowLevel.FILTER_RSINOGRAM

"""
Abstract supertype for all computational backends.
"""

include("types.jl")
include("system.jl")
include("geometries.jl")
include("methods.jl")
include("project.jl")
include("backproject.jl")
include("reconstruct.jl")
include("noise.jl")
include("phantoms.jl")
include("geometry_utils.jl")
include("operators.jl")
include("dart.jl")
include("composite.jl")

end # module AstraToolbox
