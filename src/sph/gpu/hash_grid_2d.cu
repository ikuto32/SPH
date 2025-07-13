#include "hash_grid_2d.hpp"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
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
void computeHashKernel(const float2* pos,
                       uint32_t* hashOut,
                       uint32_t* idxOut,
                       float invCell,
                       uint32_t N) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float2 p = pos[i];
    int gx = static_cast<int>(p.x * invCell);
    int gy = static_cast<int>(p.y * invCell);
    hashOut[i] = cantorHash(gx, gy);
    idxOut[i]  = i;
}

__global__
void findCellStartKernel(const uint32_t* hashBuf,
                         uint32_t* cellStart,
                         uint32_t* cellEnd,
                         uint32_t N) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint32_t hash = hashBuf[i];
    if (i == 0 || hash != hashBuf[i - 1]) {
        cellStart[hash] = i;
        if (i > 0) cellEnd[hashBuf[i - 1]] = i;
    }
    if (i == N - 1) {
        cellEnd[hash] = N;
    }
}

void HashGrid2D::build(uint32_t N, cudaStream_t s) {
    constexpr int BLK = 256;
    computeHashKernel<<<(N + BLK - 1) / BLK, BLK, 0, s>>>(particles.pos, hashBuf, idxBuf, invCell, N);
    CUDA_KERNEL_CHECK();
    thrust::device_ptr<uint32_t> dH(hashBuf), dI(idxBuf);
    thrust::sort_by_key(thrust::cuda::par.on(s), dH, dH + N, dI);
    CUDA_TRY(cudaMemsetAsync(cellStart, 0xFF, gridCells * sizeof(uint32_t), s));
    CUDA_TRY(cudaMemsetAsync(cellEnd, 0x00, gridCells * sizeof(uint32_t), s));
    findCellStartKernel<<<(N + BLK - 1) / BLK, BLK, 0, s>>>(hashBuf, cellStart, cellEnd, N);
    CUDA_KERNEL_CHECK();
}

} // namespace sph

