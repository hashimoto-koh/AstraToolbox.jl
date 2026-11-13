const PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const WORK_ROOT = joinpath(@__DIR__, "work")
const SOURCE_ROOT = joinpath(WORK_ROOT, "astra-toolbox")
const BUILD_ROOT = joinpath(SOURCE_ROOT, "build", "linux")
const INSTALL_LIBDIR = joinpath(@__DIR__, "usr", "lib")
const DEPS_FILE = joinpath(@__DIR__, "deps.jl")

const WRAPPER_HEADER = joinpath(@__DIR__, "wrapper", "include", "astra", "julia_c_api.h")
const WRAPPER_SOURCE = joinpath(@__DIR__, "wrapper", "src", "julia_c_api.cpp")

const DEFAULT_ASTRA_SOURCE = "https://github.com/astra-toolbox/astra-toolbox.git"
const DEFAULT_ASTRA_REF = "934920a46781142c2b8f0177007c395eae601783" # v2.5.0

function env_or(names::Vector{String}, default::String = "")
    for name in names
        value = get(ENV, name, "")
        isempty(value) || return value
    end
    return default
end

function detect_cuda_root()
    configured = env_or(["ASTRATOOLBOXJL_CUDA", "CUDA_ROOT", "CUDA_HOME"], "")
    if !isempty(configured)
        isdir(configured) || error("configured CUDA root does not exist: $(configured)")
        return configured
    end

    if isdir("/usr/local/cuda")
        return "/usr/local/cuda"
    end

    nvcc = Sys.which("nvcc")
    nvcc === nothing && error("CUDA toolkit was not found; set ASTRATOOLBOXJL_CUDA or ensure nvcc is on PATH")
    return normpath(joinpath(dirname(nvcc), ".."))
end

function run_logged(cmd::Cmd; dir::Union{Nothing, String} = nothing)
    shown = dir === nothing ? cmd : Cmd(cmd; dir = dir)
    println(">> ", shown)
    run(shown)
end

function reset_workdir(path::String)
    rm(path; recursive = true, force = true)
    mkpath(dirname(path))
end

function acquire_astra_source()
    source = env_or(["ASTRATOOLBOXJL_ASTRA_SOURCE"], DEFAULT_ASTRA_SOURCE)
    ref = env_or(["ASTRATOOLBOXJL_ASTRA_REF"], DEFAULT_ASTRA_REF)

    if !isdir(SOURCE_ROOT) || !isdir(joinpath(SOURCE_ROOT, ".git"))
        reset_workdir(SOURCE_ROOT)
        run_logged(`git clone $source $SOURCE_ROOT`)
        run_logged(`git checkout $ref`; dir = SOURCE_ROOT)
        rm(DEPS_FILE; force = true)
    else
        current_rev = try
            readchomp(`git -C $SOURCE_ROOT rev-parse HEAD`)
        catch
            ""
        end
        target_rev = try
            readchomp(`git -C $SOURCE_ROOT rev-parse $ref`)
        catch
            ""
        end

        if isempty(target_rev)
            println("Fetching latest tags and commits from $source...")
            run_logged(`git fetch --tags origin`; dir = SOURCE_ROOT)
            target_rev = try
                readchomp(`git -C $SOURCE_ROOT rev-parse $ref`)
            catch
                error("Specified ASTRA ref '$ref' could not be resolved in $SOURCE_ROOT after git fetch")
            end
        end

        if current_rev != target_rev
            println("Upstream work tree ref mismatch (current: $current_rev, target: $target_rev). Updating...")
            run_logged(`git checkout -f $ref`; dir = SOURCE_ROOT)
            run_logged(`git clean -fdx build/linux`; dir = SOURCE_ROOT)
            rm(DEPS_FILE; force = true)
        end
    end
end

function patch_upstream_sources()
    darthelper = joinpath(SOURCE_ROOT, "cuda", "2d", "darthelper.cu")
    if isfile(darthelper)
        content = read(darthelper, String)
        if occursin("d[o3 + 1] != d[o2] ||", content) && occursin("d[o3] != d[o2]", content)
            content = replace(content, "d[o2 - 1] != d[o2] ||                  d[o3 + 1] != d[o2] ||" => "d[o2 - 1] != d[o2] ||                  d[o2 + 1] != d[o2] ||")
            write(darthelper, content)
        end
    end
end

function install_wrapper_files()
    mkpath(joinpath(SOURCE_ROOT, "include", "astra"))
    cp(WRAPPER_HEADER, joinpath(SOURCE_ROOT, "include", "astra", "julia_c_api.h"); force = true)
    cp(WRAPPER_SOURCE, joinpath(SOURCE_ROOT, "src", "julia_c_api.cpp"); force = true)
end

function patch_makefile()
    makefile = joinpath(BUILD_ROOT, "Makefile.in")
    content = read(makefile, String)
    occursin("src/julia_c_api.lo", content) && return

    marker = "src/Globals.lo \\\n"
    replacement = marker * "\tsrc/julia_c_api.lo \\\n"
    occursin(marker, content) || error("failed to patch Makefile.in: marker not found")
    write(makefile, replace(content, marker => replacement))
end

function build_astra()
    cuda_root = detect_cuda_root()
    if !isfile(joinpath(BUILD_ROOT, "Makefile"))
        run_logged(`./autogen.sh`; dir = BUILD_ROOT)
        run_logged(`./configure --with-cuda=$cuda_root --with-python=no`; dir = BUILD_ROOT)
    end
    jobs = string(min(Sys.CPU_THREADS, 8))
    run_logged(`make -j$jobs libastra.la`; dir = BUILD_ROOT)
end

function install_library()
    mkpath(INSTALL_LIBDIR)
    built_lib = joinpath(BUILD_ROOT, ".libs", "libastra.so")
    isfile(built_lib) || error("built libastra.so was not found at $(built_lib)")
    target = joinpath(INSTALL_LIBDIR, "libastra.so")
    run_logged(`cp -L $built_lib $target`)
    write(DEPS_FILE, "const libastra = $(repr(target))\n")
end

if get(ENV, "ASTRATOOLBOXJL_SOURCE_ONLY", "") == "1"
    acquire_astra_source()
    println("ok: AstraToolbox source acquisition completed (ASTRATOOLBOXJL_SOURCE_ONLY=1)")
    exit(0)
end

try
    acquire_astra_source()
    patch_upstream_sources()
    install_wrapper_files()
    patch_makefile()
    build_astra()
    install_library()
    println("ok: AstraToolbox build completed")
catch err
    rm(DEPS_FILE; force = true)
    rethrow(err)
end
