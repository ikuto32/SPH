#include "sph/gpu/hash_grid_2d.hpp"
#include <cuda_runtime.h>
#include <vector>
#include "../src/sph/debug_gpu.hpp"
#include <cassert>

using namespace sph;

int main() {
#ifdef SPH_ENABLE_HASH2D
    const uint32_t Nx = 32;
    const uint32_t Ny = 32;
    const uint32_t N = Nx * Ny;

    std::vector<float2> hPos(N);
    for (uint32_t i = 0; i < N; ++i) {
        hPos[i] = make_float2(static_cast<float>(i % Nx), static_cast<float>(i / Nx));
    }

    ParticleSoA p{};
    CUDA_TRY(cudaMalloc(&p.pos, N * sizeof(float2)));
    CUDA_TRY(cudaMemcpy(p.pos, hPos.data(), N * sizeof(float2), cudaMemcpyHostToDevice));

    HashGrid2D grid{};
    grid.gridDim = make_uint2(Nx, Ny);
    grid.invCell = 1.0f;
    grid.gridCells = Nx * Ny;
    grid.particles = p;
    CUDA_TRY(cudaMallocManaged(&grid.hashBuf, N * sizeof(uint32_t)));
    CUDA_TRY(cudaMallocManaged(&grid.idxBuf, N * sizeof(uint32_t)));
    CUDA_TRY(cudaMallocManaged(&grid.cellStart, grid.gridCells * sizeof(uint32_t)));
    CUDA_TRY(cudaMallocManaged(&grid.cellEnd, grid.gridCells * sizeof(uint32_t)));

    grid.build(N, 0);
    CUDA_TRY(cudaDeviceSynchronize());

    uint32_t* neigh;
    uint32_t* count;
    CUDA_TRY(cudaMallocManaged(&neigh, N * MAX_NEIGHBORS * sizeof(uint32_t)));
    CUDA_TRY(cudaMallocManaged(&count, N * sizeof(uint32_t)));

    float hh = 1.1f;
    grid.findNeighbors(N, hh, neigh, count);
    CUDA_TRY(cudaDeviceSynchronize());

    std::vector<uint32_t> cpuCount(N, 0);
    float hh2 = hh * hh;
    for (uint32_t i = 0; i < N; ++i) {
        float2 pi = hPos[i];
        for (uint32_t j = 0; j < N && cpuCount[i] < MAX_NEIGHBORS; ++j) {
            float2 pj = hPos[j];
            float dx = pi.x - pj.x;
            float dy = pi.y - pj.y;
            if (dx * dx + dy * dy <= hh2) {
                cpuCount[i]++;
            }
        }
    }

    for (uint32_t i = 0; i < N; ++i) {
        assert(count[i] == cpuCount[i]);
    }

    CUDA_TRY(cudaFree(neigh));
    CUDA_TRY(cudaFree(count));
    CUDA_TRY(cudaFree(p.pos));
    CUDA_TRY(cudaFree(grid.hashBuf));
    CUDA_TRY(cudaFree(grid.idxBuf));
    CUDA_TRY(cudaFree(grid.cellStart));
    CUDA_TRY(cudaFree(grid.cellEnd));
#endif
    return 0;
}
