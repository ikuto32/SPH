#include "hash_grid_2d.hpp"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cmath>

namespace sph {

__global__ void neighbourSearchKernel(const float* posX, const float* posY,
                                      const int* particleIndex, const int* particleHash,
                                      const int* cellStart, const int* cellEnd,
                                      float cellSize, int gridWidth, int gridHeight,
                                      float radius, int n, int* neighbourCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int p = particleIndex[idx];
    float x = posX[p];
    float y = posY[p];
    int hash = particleHash[idx];
    int cx = hash % gridWidth;
    int cy = hash / gridWidth;
    int count = 0;
    float r2 = radius * radius;
    for (int dy=-1; dy<=1; ++dy) {
        int ny = cy + dy;
        if (ny < 0 || ny >= gridHeight) continue;
        for (int dx=-1; dx<=1; ++dx) {
            int nx = cx + dx;
            if (nx < 0 || nx >= gridWidth) continue;
            int h = ny * gridWidth + nx;
            int start = cellStart[h];
            int end = cellEnd[h];
            if (start == -1) continue;
            for (int j=start; j<end; ++j) {
                int pj = particleIndex[j];
                if (pj == p) continue;
                float dxv = posX[pj] - x;
                float dyv = posY[pj] - y;
                float d2 = dxv*dxv + dyv*dyv;
                if (d2 <= r2) ++count;
            }
        }
    }
    neighbourCount[p] = count;
}


void launchNeighbourSearch(const float* d_posX, const float* d_posY,
                           const HashGrid2D& grid, float radius, int n,
                           int* d_neighborCount)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    neighbourSearchKernel<<<blocks, threads>>>(d_posX, d_posY,
                                              grid.d_particleIndex, grid.d_particleHash,
                                              grid.d_cellStart, grid.d_cellEnd,
                                              grid.cellSize, grid.gridWidth, grid.gridHeight,
                                              radius, n, d_neighborCount);
    CUDA_KERNEL_CHECK();
}

} // namespace sph

#endif // USE_CUDA

