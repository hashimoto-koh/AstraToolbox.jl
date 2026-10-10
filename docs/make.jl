using Documenter
using AstraToolbox

# Automatically generate markdown tutorials from demos/*.ipynb
run(`python3 $(joinpath(@__DIR__, "generate_tutorials.py"))`)

makedocs(
    sitename = "AstraToolbox.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://hashimoto-koh.github.io/AstraToolbox.jl",
        size_threshold = nothing,
        size_threshold_warn = 500 * 1024,
    ),
    modules = [AstraToolbox],
    pages = [
        "Home" => "index.md",
        "Quick Start" => "quickstart.md",
        "Installation" => "installation.md",
        "Toolbox contents" => "toolbox_contents.md",
        "Data Objects" => "data_objects.md",
        "Geometries" => [
            "2D Geometries" => "geometries_2d.md",
            "3D Geometries" => "geometries_3d.md",
        ],
        "Projectors" => "projectors.md",
        "Algorithms" => "algorithms.md",
        "Utilities" => "utilities.md",
        "Tutorials" => [
            "Tutorial 1: 2D Parallel-Beam CT" => "tutorials/01_2d_parallel_basic.md",
            "Tutorial 2: Linear Operators & Adjointness" => "tutorials/02_projector_and_adjoint.md",
            "Tutorial 3: Custom FISTA Reconstruction" => "tutorials/03_custom_iterative_reconstruction.md",
            "Tutorial 4: 3D Cone-Beam FDK" => "tutorials/04_3d_conebeam_fdk.md",
            "Tutorial 5: 3D Vector Geometry & Alignment" => "tutorials/05_3d_geometry_and_cone_vec.md",
            "Tutorial 6: Computed Laminography" => "tutorials/06_conebeam_laminography.md",
            "Tutorial 7: Spatial Sampling & Accuracy" => "tutorials/07_sampling_and_accuracy.md",
            "Tutorial 8: Limited-Angle & Sparse-View CT" => "tutorials/08_limited_angle_and_sparse_view.md",
            "Tutorial 9: GPU Memory & Large Scale" => "tutorials/09_large_scale_and_gpu.md",
        ],
        "Interactive Demonstrations" => "demos.md",
        "Miscellaneous" => "misc.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/hashimoto-koh/AstraToolbox.jl.git",
    devbranch = "master",
)
