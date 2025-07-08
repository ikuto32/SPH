#include "kernels_cuda.h"

#ifdef USE_CUDA

#include <cmath>          // 必須
#include <cstdlib>
#ifndef M_PI              // まだ無ければ自前で定義
#define M_PI 3.14159265358979323846
#endif

namespace sph {

namespace {
    // Cached device buffers and CUDA availability flag
    bool             initialized   = false;
    bool             gpu_available = false;
    float*           d_in          = nullptr;
    float*           d_out         = nullptr;
    int              buffer_size   = 0;

    void freeBuffers() {
        if (d_in)  cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        d_in  = nullptr;
        d_out = nullptr;
        buffer_size = 0;
    }
}

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

void calcSmoothingKernelCUDA(const float* dist, float* out, float radius, int n)
{
    if (!initialized) {
        int deviceCount = 0;
        cudaError_t st = cudaGetDeviceCount(&deviceCount);
        gpu_available = (st == cudaSuccess && deviceCount > 0);
        initialized = true;
        if (gpu_available) std::atexit(freeBuffers);
    }

    if (!gpu_available) {
        for (int i = 0; i < n; ++i) {
            out[i] = calcSmoothingKernel(dist[i], radius);
        }
        return;
    }

    if (buffer_size != n) {
        freeBuffers();
        cudaError_t st1 = cudaMalloc(&d_in, n * sizeof(float));
        if (st1 != cudaSuccess) {
            gpu_available = false;
            for (int i = 0; i < n; ++i) out[i] = calcSmoothingKernel(dist[i], radius);
            return;
        }
        st1 = cudaMalloc(&d_out, n * sizeof(float));
        if (st1 != cudaSuccess) {
            cudaFree(d_in);
            d_in = nullptr;
            gpu_available = false;
            for (int i = 0; i < n; ++i) out[i] = calcSmoothingKernel(dist[i], radius);
            return;
        }
        buffer_size = n;
    }

    cudaError_t status = cudaMemcpy(d_in, dist, n * sizeof(float), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        freeBuffers();
        gpu_available = false;
        for (int i = 0; i < n; ++i) out[i] = calcSmoothingKernel(dist[i], radius);
        return;
    }

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    calcSmoothingKernelKernel<<<blocks, threads>>>(d_in, d_out, radius, n);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        freeBuffers();
        gpu_available = false;
        for (int i = 0; i < n; ++i) out[i] = calcSmoothingKernel(dist[i], radius);
        return;
    }
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        freeBuffers();
        gpu_available = false;
        for (int i = 0; i < n; ++i) out[i] = calcSmoothingKernel(dist[i], radius);
        return;
    }

    status = cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        freeBuffers();
        gpu_available = false;
        for (int i = 0; i < n; ++i) out[i] = calcSmoothingKernel(dist[i], radius);
        return;
    }
}

} // namespace sph

#endif // USE_CUDA
