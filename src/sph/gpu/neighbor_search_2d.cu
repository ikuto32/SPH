#include "hash_grid_2d.hpp"
#include <cuda_runtime.h>

namespace sph {

#ifndef MAX_NEIGHBORS
#define MAX_NEIGHBORS 64
#endif

__global__
void neighbourSearchKernel(const float2* pos,
                           const uint32_t* cellStart,
                           const uint32_t* cellEnd,
                           const uint32_t* idxBuf,
                           float hh,
                           uint2 gridDim,
                           float invCell,
                           uint32_t N,
                           uint32_t* outCount) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // Placeholder: zero neighbours
    outCount[i] = 0;
}

} // namespace sph
