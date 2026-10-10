# Toolbox contents

`AstraToolbox.jl` provides a high-level Julia interface to the ASTRA Toolbox core.

## Main Functionality

The toolbox is organized into several key components:

### 2D and 3D Geometries

Define the physical setup of your tomography system, including source positions, detector pixel sizes, and rotation angles.

- **2D Geometries**: `Parallel2DGeometry`, `ParallelVec2DGeometry`, `FanFlat2DGeometry`, `FanFlatVec2DGeometry`.
- **3D Geometries**: `Parallel3DGeometry`, `ParallelVec3DGeometry`, `Cone3DGeometry`, `ConeVec3DGeometry`, `CylConeVec3DGeometry`.

### Forward and Backward Projection

Apply the physical model of X-ray projection and its adjoint operator (backprojection).

- **`project` / `project!`**: Computes the forward projection (sinogram) of a volume.
- **`backproject` / `backproject!`**: Computes the backprojection of a sinogram onto a volume.

### Reconstruction Algorithms

State-of-the-art algorithms for reconstructing images from projection data.

- **Analytical**: `FBPMethod` (2D), `FDK3DMethod` (3D).
- **Iterative**: `SIRTMethod`, `SARTMethod`, `CGLSMethod`, `EMMethod` (2D), `SIRT3DMethod`, `CGLS3DMethod` (3D).

### Backend Support

- **`cpu()`**: Runs computations on the host CPU.
- **`cuda()`**: Accelerates computations using NVIDIA GPUs via CUDA.

## Data Layout and Conventions

The package uses standard Julia `Array{Float32}` for all data.

- **2D volume**: `A[x, y]` (size $N_x \times N_y$)
- **3D volume**: `A[x, y, z]` (size $N_x \times N_y \times N_z$)
- **2D sinogram**: `S[u, angle]` (size $N_{det} \times N_{angles}$)
- **3D sinogram**: `S[u, angle, v]` (size $N_{det,u} \times N_{angles} \times N_{det,v}$)

Refer to the [Data Objects](data_objects.md) page for more details.
