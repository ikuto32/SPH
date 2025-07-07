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

void calcSmoothingKernelCUDA(const float* dist, float* out, float radius, int n)
{
    int deviceCount = 0;
    cudaError_t status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess || deviceCount == 0) {
        // Fallback to CPU implementation when CUDA is not available
        for (int i = 0; i < n; ++i) {
            out[i] = calcSmoothingKernel(dist[i], radius);
        }
        return;
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    status = cudaMalloc(&d_in, n * sizeof(float));
    if (status != cudaSuccess) {
        for (int i = 0; i < n; ++i) out[i] = calcSmoothingKernel(dist[i], radius);
        return;
    }
    status = cudaMalloc(&d_out, n * sizeof(float));
    if (status != cudaSuccess) {
        cudaFree(d_in);
        for (int i = 0; i < n; ++i) out[i] = calcSmoothingKernel(dist[i], radius);
        return;
    }
    status = cudaMemcpy(d_in, dist, n * sizeof(float), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        for (int i = 0; i < n; ++i) out[i] = calcSmoothingKernel(dist[i], radius);
        return;
    }

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    calcSmoothingKernelKernel<<<blocks, threads>>>(d_in, d_out, radius, n);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        for (int i = 0; i < n; ++i) out[i] = calcSmoothingKernel(dist[i], radius);
        return;
    }
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        for (int i = 0; i < n; ++i) out[i] = calcSmoothingKernel(dist[i], radius);
        return;
    }

    status = cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        for (int i = 0; i < n; ++i) out[i] = calcSmoothingKernel(dist[i], radius);
        return;
    }

    cudaFree(d_in);
    cudaFree(d_out);
}

} // namespace sph

#endif // USE_CUDA
