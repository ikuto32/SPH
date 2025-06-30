#include "kernels_cuda.h"

#ifdef USE_CUDA

#include <cmath>          // 必須
#ifndef M_PI              // まだ無ければ自前で定義
#define M_PI 3.14159265358979323846
#endif

namespace sph {

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

void calcSmoothingKernelGPU(const float* d_dist, float* d_out, float radius, int n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    calcSmoothingKernelKernel<<<blocks, threads>>>(d_dist, d_out, radius, n);
    CUDA_KERNEL_CHECK();
}

} // namespace sph

#endif // USE_CUDA
