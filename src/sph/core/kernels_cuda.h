#ifndef SPH_CORE_KERNELS_CUDA_H
#define SPH_CORE_KERNELS_CUDA_H

#include "kernels.h"
#include <vector>

#ifdef USE_CUDA
#include "cuda_utils.h"
#include <cuda_runtime.h>
#endif

namespace sph {

#ifdef USE_CUDA

// CUDA implementation of the smoothing kernel lives in a separate translation
// unit. When compiling this header with a host compiler (i.e. when __CUDACC__ is
// not defined) only the declarations are visible.
#ifdef __CUDACC__
__global__ void calcSmoothingKernelKernel(const float* dist, float* out, float radius, int n);
#endif

// Calculate the kernel on the GPU. The input and output pointers must already
// reside on the device; this function performs no memory copies.
void calcSmoothingKernelGPU(const float* d_dist, float* d_out, float radius, int n);

#endif // USE_CUDA

// Pure CPU implementation used when CUDA is unavailable or when running on the
// host even in CUDA builds.
inline void calcSmoothingKernelCPU(const float* dist, float* out, float radius, int n)
{
    for (int i = 0; i < n; ++i) {
        out[i] = calcSmoothingKernel(dist[i], radius);
    }
}


} // namespace sph

#endif // SPH_CORE_KERNELS_CUDA_H
