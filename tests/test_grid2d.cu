#include "sph/gpu/hash_grid_2d.hpp"
#include <cuda_runtime.h>
#include <vector>
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
    cudaMalloc(&p.pos, N * sizeof(float2));
    cudaMemcpy(p.pos, hPos.data(), N * sizeof(float2), cudaMemcpyHostToDevice);
    HashGrid2D grid{};
    grid.hashBuf = (uint32_t*)cudaMallocManaged(N * sizeof(uint32_t));
    grid.idxBuf  = (uint32_t*)cudaMallocManaged(N * sizeof(uint32_t));
    grid.cellStart = (uint32_t*)cudaMallocManaged(cells * sizeof(uint32_t));
    grid.cellEnd   = (uint32_t*)cudaMallocManaged(cells * sizeof(uint32_t));
    grid.gridDim = make_uint2(256,256);
    grid.invCell = 1.0f;
    grid.gridCells = cells;
    grid.particles = p;
    grid.build(N, 0);
    cudaDeviceSynchronize();
    // success if no crash
    cudaFree(p.pos);
    cudaFree(grid.hashBuf);
    cudaFree(grid.idxBuf);
    cudaFree(grid.cellStart);
    cudaFree(grid.cellEnd);
#endif
    return 0;
}
