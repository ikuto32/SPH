#ifndef SPH_GPU_HASH_GRID_2D_HPP
#define SPH_GPU_HASH_GRID_2D_HPP

#include <cuda_runtime.h>
#include <cstdint>

namespace sph {

inline constexpr uint32_t MAX_NEIGHBORS = 64;

struct ParticleSoA {
    float2* pos;
    float2* vel;
    float*  rho;
    float*  prs;
};

struct HashGrid2D {
    uint32_t* hashBuf;
    uint32_t* idxBuf;
    uint32_t* cellStart;
    uint32_t* cellEnd;
    uint2      gridDim;
    float      invCell;
    uint32_t   gridCells;

    ParticleSoA particles;

    void build(uint32_t N, cudaStream_t s = 0);
    void findNeighbors(uint32_t N,
                       float hh,
                       uint32_t* neighborsOut,
                       uint32_t* outCount,
                       cudaStream_t s = 0);
};

} // namespace sph

#endif // SPH_GPU_HASH_GRID_2D_HPP
