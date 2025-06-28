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
void calcSmoothingKernelCUDA(const float* dist, float* out, float radius, int n,
                             float* d_in, float* d_out);

#else // !USE_CUDA

inline void calcSmoothingKernelCUDA(const float* dist, float* out, float radius, int n,
                                    float*, float*)
{
    for (int i = 0; i < n; ++i) {
        out[i] = calcSmoothingKernel(dist[i], radius);
    }
}

#endif // USE_CUDA

} // namespace sph

#endif // SPH_CORE_KERNELS_CUDA_H
