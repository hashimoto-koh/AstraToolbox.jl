# 2D and 3D Data Objects

In `AstraToolbox.jl`, data objects are simply Julia arrays of type `Array{Float32}`.

This is a significant departure from the original Python or MATLAB interfaces, where you have to explicitly create and manage data objects using unique IDs.

## Julia Arrays as Data Objects

Every function in `AstraToolbox.jl` that requires data (like a volume or a sinogram) accepts standard Julia arrays.

### Advantages of the Julia-native Approach

- **Automatic Memory Management**: Julia's garbage collector automatically handles memory allocation and deallocation. You don't need to manually delete data objects when you're done.
- **Ease of Use**: You can use all of Julia's built-in array manipulation functions, such as indexing, slicing, and broadcasting, directly on your ASTRA data.
- **Zero-Copy Interoperability**: When calling ASTRA functions, the package passes pointers to the underlying memory of your Julia arrays, avoiding unnecessary data copying.

## Data Layout Conventions

To ensure consistent results, `AstraToolbox.jl` follows these axis conventions:

### 2D Data

- **Volume**: `A[x, y]`
    - `x` is the horizontal coordinate (columns).
    - `y` is the vertical coordinate (rows).
- **Sinogram**: `S[u, angle]`
    - `u` is the detector column index.
    - `angle` is the index of the projection angle.

### 3D Data

- **Volume**: `A[x, y, z]`
    - `x` and `y` are the horizontal coordinates.
    - `z` is the vertical coordinate (slice).
- **Sinogram**: `S[u, angle, v]`
    - `u` is the detector column index (horizontal).
    - `angle` is the index of the projection angle.
    - `v` is the detector row index (vertical).

## Precision

ASTRA's core computations are performed in single precision (`Float32`).

- **Bang Functions (`!`)**: Functions like `project!` and `reconstruct!` require input arrays to be explicitly typed as `Array{Float32}`.
- **Non-Bang Functions**: Functions like `project` and `reconstruct` are more flexible. They accept any numeric array (e.g., `Array{Float64}`) and internally convert it to `Float32` if necessary.

## Allocation and Mutation

- **`project`**: Allocates a new `Array{Float32}` for the output sinogram.
- **`project!`**: Writes the output sinogram directly into a pre-allocated `Array{Float32}`. Use this to minimize memory allocations in performance-critical loops.
