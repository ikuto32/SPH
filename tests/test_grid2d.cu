#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>

#include "sph/gpu/hash_grid_2d.hpp"

#ifdef USE_CUDA
#ifdef SPH_ENABLE_HASH2D

int main() {
    constexpr int NX = 256;
    constexpr int NY = 256;
    constexpr int N = NX * NY;
    const float dx = 1.0f;
    const float dy = std::sqrt(3.0f) / 2.0f;
    const float margin = 1.0f;
    const float radius = 1.01f;
    const float width = NX + 2.0f + 1.0f;  // margin on both sides + offset
    const float height = NY * dy + 2.0f;

    std::vector<float> h_posX(N);
    std::vector<float> h_posY(N);
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = y * NX + x;
            h_posX[idx] = margin + x + (y & 1) * 0.5f;
            h_posY[idx] = margin + y * dy;
        }
    }

    float* d_posX = nullptr;
    float* d_posY = nullptr;
    CUDA_CHECK(cudaMalloc(&d_posX, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_posY, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_posX, h_posX.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posY, h_posY.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    sph::HashGrid2D grid(width, height, radius);
    grid.build(d_posX, d_posY, N);

    int* d_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counts, N * sizeof(int)));
    sph::launchNeighbourSearch(d_posX, d_posY, grid, radius, N, d_counts);

    std::vector<int> counts(N);
    CUDA_CHECK(cudaMemcpy(counts.data(), d_counts, N * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_posX));
    CUDA_CHECK(cudaFree(d_posY));
    CUDA_CHECK(cudaFree(d_counts));

    auto mm = std::minmax_element(counts.begin(), counts.end());
    assert(*mm.first == 6);
    assert(*mm.second == 6);
    return 0;
}

#else
int main() { return 0; }
#endif
#else
int main() { return 0; }
#endif
