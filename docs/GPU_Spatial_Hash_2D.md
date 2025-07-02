# GPU Spatial Hash Neighbour Search (2-D)

This document outlines how the optional CUDA implementation integrates
with the rest of the repository. The architecture follows a clean
separation of domain logic and GPU infrastructure.

## Structure

- **Domain layer (`src/sph/core`)** – physics, data structures and CPU fallback.
- **Infrastructure layer (`src/sph/gpu`)** – CUDA kernels for hashing and neighbour search.
- **Interface layer (`bindings`)** – Python bindings and example applications.

The CUDA path requires devices with compute capability 8.0 or higher and
uses features introduced in newer architectures. NVIDIA's programming
guide states that operations such as warp reductions are supported only
on devices of compute capability 8.0 and above【86cec1†L3-L21】.
Cluster-wide synchronization is guaranteed on compute capability 9.0
GPUs, where thread blocks share distributed memory【abb017†L3-L11】.

## Build Option

Enable GPU support when configuring CMake:

```console
cmake -DUSE_CUDA=ON ..
```

When `SPH_ENABLE_HASH2D` is defined, the CUDA sources in
`src/sph/gpu` are compiled and the `World` class dispatches to the
GPU neighbour search.

## Workflow

1. The host collects particle buffers and launches
   `computeHashKernel` and `findCellStartKernel`.
2. Sorted indices are fed to `neighbourSearchKernel` which runs in a
   persistent grid.
3. Subsequent density and force kernels reuse the neighbour lists.

The default build targets compute capability 9.0 (Hopper) but can run
on any GPU of capability 8.0 or newer.

