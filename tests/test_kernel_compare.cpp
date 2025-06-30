#include <cassert>
#include <cmath>
#include <vector>
#include "sph/core/kernels.h"
#include "sph/core/kernels_cuda.h"

int main() {
    const float radius = 1.5f;
    std::vector<float> distances;
    for (int i = 0; i <= 20; ++i) {
        distances.push_back(radius * i / 20.0f);
    }
    const int n = static_cast<int>(distances.size());
    std::vector<float> cpu(n), gpu(n);
    for (int i = 0; i < n; ++i) {
        cpu[i] = sph::calcSmoothingKernel(distances[i], radius);
    }
#ifdef USE_CUDA
    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, distances.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    sph::calcSmoothingKernelGPU(d_in, d_out, radius, n);
    CUDA_CHECK(cudaMemcpy(gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
#else
    sph::calcSmoothingKernelCPU(distances.data(), gpu.data(), radius, n);
#endif
    for (int i = 0; i < n; ++i) {
        assert(std::abs(cpu[i] - gpu[i]) < 1e-5f);
    }
    return 0;
}
