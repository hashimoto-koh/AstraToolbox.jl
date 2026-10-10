# Third-Party Notices

This repository, `AstraToolbox.jl`, is distributed under the GNU General Public License v3.0 or later (GPL-3.0-or-later). See [LICENSE](LICENSE).

---

## Upstream Project

`AstraToolbox.jl` is an unofficial Julia port and binding for the upstream **ASTRA Toolbox** project:

This project is **not** affiliated with, endorsed by, or maintained by imec-Vision Lab, the University of Antwerp, CWI (Centrum Wiskunde & Informatica), or the upstream ASTRA Toolbox contributors.

- **Upstream repository**: <https://github.com/astra-toolbox/astra-toolbox>
- **Upstream documentation & website**: <https://www.astra-toolbox.com/>
- **Upstream copyright**:
  - Copyright: 2010–2024, imec Vision Lab, University of Antwerp (<https://visielab.uantwerpen.be/>)
  - Copyright: 2014–2024, CWI, Centrum Wiskunde & Informatica, Amsterdam (<https://www.cwi.nl/>)
- **Upstream license**: GNU General Public License v3.0 or later (GPL-3.0-or-later)

This repository exists because the original `ASTRA Toolbox` is an outstanding, highly performant, and widely used platform in tomographic reconstruction research. `AstraToolbox.jl` is developed with explicit respect for that foundational work, and the maintainers express their sincere gratitude to the original authors, maintainers, and contributors at imec-Vision Lab and CWI for creating and sustaining the ASTRA Toolbox as a world-class open-source tomographic reconstruction platform.

---

## Academic Citations & References

If you use `AstraToolbox.jl` or the underlying ASTRA Toolbox in scientific publications or research projects, please cite the following original upstream papers:

1. **General Architecture and 2D/3D Toolbox**:
   > W. van Aarle, W. J. Palenstijn, J. Cant, E. Janssens, F. Bleichrodt, A. Dabravolski, J. De Beenhouwer, K. J. Batenburg, and J. Sijbers,  
   > *"Fast and Flexible X-ray Tomography Using the ASTRA Toolbox"*,  
   > **Optics Express**, vol. 24, no. 22, pp. 25129–25147, 2016.  
   > DOI: [10.1364/OE.24.025129](https://doi.org/10.1364/OE.24.025129)

2. **Electron Tomography & Advanced Algorithms**:
   > W. van Aarle, W. J. Palenstijn, J. De Beenhouwer, T. Altantzis, S. Bals, K. J. Batenburg, and J. Sijbers,  
   > *"The ASTRA Toolbox: A platform for advanced algorithm development in electron tomography"*,  
   > **Ultramicroscopy**, vol. 157, pp. 35–47, 2015.  
   > DOI: [10.1016/j.ultramic.2015.05.002](https://doi.org/10.1016/j.ultramic.2015.05.002)

3. **GPU Acceleration & Iterative Solvers**:
   > W. J. Palenstijn, K. J. Batenburg, and J. Sijbers,  
   > *"Performance improvements for iterative electron tomography reconstruction using graphics processing units (GPUs)"*,  
   > **Journal of Structural Biology**, vol. 176, issue 2, pp. 250–253, 2011.  
   > DOI: [10.1016/j.jsb.2011.07.017](https://doi.org/10.1016/j.jsb.2011.07.017)

---

## How This Repository Uses Upstream Material

1. **No Source Vendoring in Git History**:
   This repository does not vendor the upstream ASTRA Toolbox source tree directly in version control.
2. **Reproducible Build-Time Fetching**:
   During `Pkg.build("AstraToolbox")`, the build script ([`deps/build.jl`](deps/build.jl)) fetches upstream source code pinned to a verified release commit (`bd0323e9b4b243ee90bca41341927b05e6e816c8`, ASTRA 2.4.1+) and builds the native C++/CUDA shared library (`libastra.so`).
3. **Zero-Copy Native Bridge**:
   This repository provides a lightweight C API bridge ([`deps/wrapper/include/astra/julia_c_api.h`](deps/wrapper/include/astra/julia_c_api.h) and [`deps/wrapper/src/julia_c_api.cpp`](deps/wrapper/src/julia_c_api.cpp)) that interacts with ASTRA's native C++ object managers (`CAstraObjectManager`, `CCompositeGeometryManager`).
4. **Native Julia Integration**:
   The Julia package interfaces directly with the compiled native library via Julia's `ccall` mechanism without depending on Python, `PyCall`, or Cython, borrowing Julia `Array{Float32}` pointers with zero memory copies.

---

## Attribution and Redistribution Notes

Because `AstraToolbox.jl` includes C++ wrapper code that directly includes ASTRA Toolbox headers and links with the ASTRA C++/CUDA library in the same process space, this entire package and any derived or combined distributions are licensed under the **GNU General Public License v3.0 or later (GPL-3.0-or-later)**.

If you redistribute this repository or derivative works:
- You must preserve all original copyright notices, license texts, and attribution information.
- You must provide the corresponding source code of any modifications under the terms of the GNU General Public License v3.0 (or later).
- Review [LICENSE](LICENSE) and [deps/build.jl](deps/build.jl) for detailed legal terms and reproducible build instructions.
