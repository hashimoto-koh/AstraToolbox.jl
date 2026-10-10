# Miscellaneous

This page includes additional notes and information about `AstraToolbox.jl`.

## Backend Management

### `cpu()` and `cuda()`

These functions are used to select the computational backend.

- `cpu()`: Use the CPU-based implementation.
- `cuda(; gpu_index = -1)`: Use the GPU-accelerated implementation via CUDA.

### `last_error()`

Returns the message of the last error that occurred in the native ASTRA library.

```julia
err = last_error()
println(err)
```

## Internal Bridge

The package uses a small C++ wrapper (`deps/wrapper/src/julia_c_api.cpp`) to bridge Julia with the native ASTRA library. This allows for a more direct and efficient interaction compared to the Python or MATLAB wrappers.

## Known Limitations

- **Platform Support**: Currently only tested and supported on Linux.
- **CUDA Requirement**: The package currently builds with CUDA enabled by default. CPU-only builds are not yet officially supported.

## Licensing

- **`AstraToolbox.jl`**: Licensed under the GNU General Public License v3 (GPLv3), same as the ASTRA Toolbox core.
- **ASTRA Toolbox Core**: Copyright imec-Vision Lab and CWI.

For more details, see the `LICENSE` and `THIRD_PARTY_NOTICES.md` files in the repository.
