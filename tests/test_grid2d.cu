#include "sph/gpu/hash_grid_2d.hpp"
#include <cuda_runtime.h>
#include <vector>
#include "../src/sph/debug_gpu.hpp"
#include <cassert>

using namespace sph;

int main() {
#ifdef SPH_ENABLE_HASH2D
    const int cells = 256 * 256;
    const uint32_t N = 66000; // approx 256x256 hex lattice
    std::vector<float2> hPos(N);
    for(uint32_t i=0;i<N;++i){
        hPos[i] = make_float2((float)(i%256), (float)(i/256));
    }
    ParticleSoA p{};
    CUDA_TRY(cudaMalloc(&p.pos, N * sizeof(float2)));
    CUDA_TRY(cudaMemcpy(p.pos, hPos.data(), N * sizeof(float2), cudaMemcpyHostToDevice));
    HashGrid2D grid{};
    CUDA_TRY(cudaMallocManaged(&grid.hashBuf, N * sizeof(uint32_t)));
    CUDA_TRY(cudaMallocManaged(&grid.idxBuf,  N * sizeof(uint32_t)));
    CUDA_TRY(cudaMallocManaged(&grid.cellStart, cells * sizeof(uint32_t)));
    CUDA_TRY(cudaMallocManaged(&grid.cellEnd,   cells * sizeof(uint32_t)));
    grid.gridDim = make_uint2(256,256);
    grid.invCell = 1.0f;
    grid.gridCells = cells;
    grid.particles = p;
    grid.build(N, 0);
    CUDA_TRY(cudaDeviceSynchronize());
    // success if no crash
    CUDA_TRY(cudaFree(p.pos));
    CUDA_TRY(cudaFree(grid.hashBuf));
    CUDA_TRY(cudaFree(grid.idxBuf));
    CUDA_TRY(cudaFree(grid.cellStart));
    CUDA_TRY(cudaFree(grid.cellEnd));
#endif
    return 0;
}
