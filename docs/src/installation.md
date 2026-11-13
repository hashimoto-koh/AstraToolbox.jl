# Installation Instructions

This page explains how to install and configure `AstraToolbox.jl`.

## Prerequisites

The current version of `AstraToolbox.jl` is primarily developed and tested on Linux systems. It builds the ASTRA Toolbox from source during the package installation process.

### Software Requirements

- **Julia 1.11** or higher.
- **C/C++ compiler toolchain**: C++17 compliant compiler (`g++ >= 7` or `clang++ >= 5`), `make`, `autoconf`, `automake`, and `libtool`.
- **NVIDIA CUDA Toolkit**: Required for building the native library (`nvcc` >= 11.0 recommended). Ensure `nvcc` is on your path or set the `ASTRATOOLBOXJL_CUDA` environment variable. *Note*: At runtime, an NVIDIA CUDA-capable GPU is required for `backend=cuda()`, while `backend=cpu()` can be used on systems without a physical GPU as long as the native library was built with the toolkit.

## Standard Installation

To install `AstraToolbox.jl` from its Git repository:

```julia
using Pkg
Pkg.add(url = "https://github.com/your-org/AstraToolbox.jl.git")
```

During installation, Julia will run the `deps/build.jl` script, which will:
1.  Clone the upstream ASTRA Toolbox source code (pinned to release `v2.5.0`, commit `934920a46781142c2b8f0177007c395eae601783`).
2.  Apply necessary patches for Julia compatibility.
3.  Configure and compile the native library.

## Environment Variables

You can customize the build process by setting the following environment variables before running `Pkg.build("AstraToolbox")`:

- `ASTRATOOLBOXJL_CUDA`: Path to the CUDA toolkit root directory (e.g., `/usr/local/cuda`).
- `ASTRATOOLBOXJL_ASTRA_SOURCE`: Git URL or local path to the upstream ASTRA source tree.
- `ASTRATOOLBOXJL_ASTRA_REF`: Git ref or commit hash to check out (default is `934920a46781142c2b8f0177007c395eae601783`, corresponding to `v2.5.0`).
- `ASTRATOOLBOXJL_SOURCE_ONLY`: Set to `1` to fetch/prepare upstream sources only without compiling the library.
- `ASTRA_JULIA_LIB`: Path to an existing `libastra.so` if you want to skip the build process and use a pre-built library.

## Troubleshooting

- **CUDA not found**: If the build fails because it cannot find CUDA, ensure `nvcc` is accessible or set `ASTRATOOLBOXJL_CUDA`.
- **Missing dependencies**: Ensure all compiler tools (`autoconf`, `libtool`, etc.) are installed on your system.
- **Non-Linux platforms**: Windows and macOS are currently not supported by the automated build script.
