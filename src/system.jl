"""
    has_cuda() -> Bool

Check if CUDA acceleration is compiled into the native ASTRA library and available on the system.

# Returns
- `Bool`: `true` if CUDA GPU support is ready for use, `false` otherwise.

# Examples
```julia
if has_cuda()
    println("CUDA is enabled!")
end
```
"""
function has_cuda()
    return LowLevel.has_cuda()
end

"""
    has_feature(feature::AbstractString) -> Bool

Check if an ASTRA feature (e.g., `"cuda"`, `"projectors_scaled_as_line_integrals"`, `"unpadded_GPULink"`) is available.

# Arguments
- `feature`: Name of the ASTRA feature string to test.

# Returns
- `Bool`: `true` if the feature is supported, `false` otherwise.

# Examples
```julia
is_available = has_feature("cuda")
```
"""
function has_feature(feature::AbstractString)
    return LowLevel.has_feature(feature)
end

"""
    get_gpu_count() -> Int

Return the number of available CUDA GPUs detected on the host system.

# Returns
- `Int`: Number of accessible CUDA GPU devices (0 if CUDA is unavailable).

# Examples
```julia
n_gpus = get_gpu_count()
```
"""
function get_gpu_count()
    return LowLevel.get_gpu_count()
end

"""
    get_gpu_info(gpu_index::Integer = -1) -> String

Return a descriptive string detailing CUDA device properties and memory capacity.
If `gpu_index == -1`, returns information for the currently active GPU device.

# Arguments
- `gpu_index`: 0-based index of the target GPU, or `-1` for the active device. Defaults to `-1`.

# Returns
- `String`: Formatted summary of GPU model, compute capability, and memory stats.

# Examples
```julia
println(get_gpu_info())
```
"""
function get_gpu_info(gpu_index::Integer = -1)
    return LowLevel.get_gpu_info(gpu_index)
end

"""
    set_gpu_index(gpu_index::Integer) -> Bool

Set the active CUDA GPU index for subsequent GPU operations.

# Arguments
- `gpu_index`: 0-based index of the CUDA GPU device to activate.

# Returns
- `Bool`: `true` if device selection was successful, `false` otherwise.

# Examples
```julia
set_gpu_index(0)
```
"""
function set_gpu_index(gpu_index::Integer)
    return LowLevel.set_gpu_index(gpu_index)
end

"""
    get_gpu_index() -> Int

Return the index of the currently active CUDA GPU device.

# Returns
- `Int`: 0-based device index of the currently active CUDA GPU.

# Examples
```julia
idx = get_gpu_index()
```
"""
function get_gpu_index()
    return LowLevel.get_gpu_index()
end
