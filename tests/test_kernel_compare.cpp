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
    sph::calcSmoothingKernelCUDA(distances.data(), gpu.data(), radius, n);
    for (int i = 0; i < n; ++i) {
        assert(std::abs(cpu[i] - gpu[i]) < 1e-5f);
    }

    // Ensure calling the CUDA kernel with zero length does not crash
    std::vector<float> dummy_in(1), dummy_out(1);
    sph::calcSmoothingKernelCUDA(dummy_in.data(), dummy_out.data(), radius, 0);

    return 0;
}
