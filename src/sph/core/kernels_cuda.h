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

__global__ void calcSmoothingKernelKernel(const float* dist, float* out, float radius, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float d = dist[idx];
        float val = 0.0f;
        if (d < radius) {
            float volume = (float)(M_PI * radius * radius * radius * radius) / 6.0f;
            float t = radius - d;
            val = t * t / volume;
        }
        out[idx] = val;
    }
}

inline void calcSmoothingKernelCUDA(const float* dist, float* out, float radius, int n)
{
    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, dist, n * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    calcSmoothingKernelKernel<<<blocks, threads>>>(d_in, d_out, radius, n);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

#else

inline void calcSmoothingKernelCUDA(const float* dist, float* out, float radius, int n)
{
    for (int i = 0; i < n; ++i) {
        out[i] = calcSmoothingKernel(dist[i], radius);
    }
}

#endif // USE_CUDA

} // namespace sph

#endif // SPH_CORE_KERNELS_CUDA_H
