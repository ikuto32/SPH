#include "hash_grid_2d.hpp"
#include <cuda_runtime.h>
#include "../debug_gpu.hpp"

namespace sph {


__device__ __forceinline__
uint32_t cantorHash(int x, int y) {
    uint32_t a = static_cast<uint32_t>(x + ((x < 0) * 0x7fffffff));
    uint32_t b = static_cast<uint32_t>(y + ((y < 0) * 0x7fffffff));
    uint32_t c = (a + b) * (a + b + 1u) / 2u + b;
    return c * 2654435761u;
}

__global__
void neighbourSearchKernel(const float2* pos,
                           const uint32_t* cellStart,
                           const uint32_t* cellEnd,
                           const uint32_t* idxBuf,
                           float hh,
                           uint2 gridDim,
                           float invCell,
                           uint32_t N,
                           uint32_t* neighborsOut,
                           uint32_t* outCount) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
#ifdef DEBUG_GPU
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("kernel alive\n");
    }
#endif
    if (i >= N) return;

    float2 pi = pos[i];
    int gx = static_cast<int>(pi.x * invCell);
    int gy = static_cast<int>(pi.y * invCell);

    uint32_t count = 0;
    float hh2 = hh * hh;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int cx = gx + dx;
            int cy = gy + dy;
            if (cx < 0 || cy < 0 || cx >= gridDim.x || cy >= gridDim.y)
                continue;
            uint32_t hash = cantorHash(cx, cy);
            uint32_t start = cellStart[hash];
            if (start == 0xffffffff)
                continue;
            uint32_t end = cellEnd[hash];
            for (uint32_t t = start; t < end; ++t) {
                uint32_t j = idxBuf[t];
                float2 pj = pos[j];
                float dxp = pj.x - pi.x;
                float dyp = pj.y - pi.y;
                float dist2 = dxp * dxp + dyp * dyp;
                if (dist2 <= hh2 && count < MAX_NEIGHBORS) {
                    neighborsOut[i * MAX_NEIGHBORS + count] = j;
                    ++count;
                }
            }
        }
    }
    outCount[i] = count;
}

void HashGrid2D::findNeighbors(uint32_t N,
                               float hh,
                               uint32_t* neighborsOut,
                               uint32_t* outCount,
                               cudaStream_t s) {
    constexpr int BLK = 256;
    neighbourSearchKernel<<<(N + BLK - 1) / BLK, BLK, 0, s>>>(
        particles.pos,
        cellStart,
        cellEnd,
        idxBuf,
        hh,
        gridDim,
        invCell,
        N,
        neighborsOut,
        outCount);
    CUDA_KERNEL_CHECK();
}

} // namespace sph
