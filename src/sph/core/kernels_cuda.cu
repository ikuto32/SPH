#include "kernels_cuda.h"
#include "../debug_gpu.hpp"

#ifdef USE_CUDA

#include <cmath>          // 必須
#include <cstdlib>
#include <mutex>
#ifndef M_PI              // まだ無ければ自前で定義
#define M_PI 3.14159265358979323846
#endif

namespace sph {

namespace {
    // Cached device buffers and CUDA availability flag
    bool             gpu_available = false;
    float*           d_in          = nullptr;
    float*           d_out         = nullptr;
    int              buffer_size   = 0;
    std::once_flag   init_flag;
    std::mutex       kernel_mutex;

    void freeBuffers() {
        if (d_in)  CUDA_TRY(cudaFree(d_in));
        if (d_out) CUDA_TRY(cudaFree(d_out));
        d_in  = nullptr;
        d_out = nullptr;
        buffer_size = 0;
    }

    void initialize() {
        int deviceCount = 0;
        cudaError_t e = cudaGetDeviceCount(&deviceCount);
        if (e != cudaSuccess) {
#ifdef DEBUG_GPU
            fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(e));
#endif
            gpu_available = false;
        } else {
            gpu_available = (deviceCount > 0);
        }

        if (gpu_available) {
            // Query the first device and verify the compute capability
            cudaDeviceProp prop{};
            CUDA_TRY(cudaGetDeviceProperties(&prop, 0));
#ifdef DEBUG_GPU
            fprintf(stderr, "CUDA device 0: %s (compute %d.%d)\n", prop.name, prop.major, prop.minor);
#endif
            // Require compute capability 8.0 or newer
            constexpr int requiredCap = 80;
            int capability = prop.major * 10 + prop.minor;
            if (capability < requiredCap) {
#ifdef DEBUG_GPU
                fprintf(stderr,
                        "Compute capability %d.%d below required 8.0, falling back to CPU\n",
                        prop.major, prop.minor);
#endif
                gpu_available = false;
            }
        } else {
#ifdef DEBUG_GPU
            fprintf(stderr, "No CUDA devices available or CUDA initialization failed\n");
#endif
        }

        if (gpu_available) std::atexit(freeBuffers);
    }
}

__global__ void calcSmoothingKernelKernel(const float* dist, float* out, float radius, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
#ifdef DEBUG_GPU
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("kernel alive\n");
    }
#endif
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
    if (n <= 0) {
        return;
    }

    std::call_once(init_flag, initialize);

    if (!gpu_available) {
        for (int i = 0; i < n; ++i) {
            out[i] = calcSmoothingKernel(dist[i], radius);
        }
        return;
    }

    std::lock_guard<std::mutex> guard(kernel_mutex);

    if (buffer_size < n) {
        freeBuffers();
        CUDA_TRY(cudaMalloc(&d_in, n * sizeof(float)));
        CUDA_TRY(cudaMalloc(&d_out, n * sizeof(float)));
        buffer_size = n;
    }

    CUDA_TRY(cudaMemcpy(d_in, dist, n * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    calcSmoothingKernelKernel<<<blocks, threads>>>(d_in, d_out, radius, n);
    CUDA_KERNEL_CHECK();

    CUDA_TRY(cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
}

} // namespace sph

#endif // USE_CUDA
