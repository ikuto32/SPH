#include "hash_grid_2d.hpp"
#ifdef USE_CUDA
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cmath>

namespace sph {

__global__ void computeHashKernel(const float* posX, const float* posY,
                                  int* hash, int* index,
                                  float cellSize, int gridWidth, int gridHeight,
                                  int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = posX[idx];
    float y = posY[idx];
    int gx = (int)floorf(x / cellSize);
    int gy = (int)floorf(y / cellSize);
    gx = max(0, min(gx, gridWidth - 1));
    gy = max(0, min(gy, gridHeight - 1));
    int h = gy * gridWidth + gx;
    hash[idx] = h;
    index[idx] = idx;
}

__global__ void clearGridKernel(int* cellStart, int* cellEnd, int numCells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numCells) {
        cellStart[idx] = -1;
        cellEnd[idx] = -1;
    }
}

__global__ void findCellStartEndKernel(const int* hash, const int* index,
                                       int* cellStart, int* cellEnd, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int curHash = hash[idx];
    if (idx == 0) {
        cellStart[curHash] = 0;
    } else {
        int prevHash = hash[idx - 1];
        if (curHash != prevHash) {
            cellStart[curHash] = idx;
            cellEnd[prevHash] = idx;
        }
    }
    if (idx == n - 1) {
        cellEnd[curHash] = n;
    }
}

HashGrid2D::HashGrid2D(float width, float height, float cell)
    : cellSize(cell)
{
    gridWidth = static_cast<int>(std::ceil(width / cell));
    gridHeight = static_cast<int>(std::ceil(height / cell));
    numCells = gridWidth * gridHeight;
    maxParticles = 0;
#ifdef USE_CUDA
    CUDA_CHECK(cudaMalloc(&d_cellStart, numCells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cellEnd, numCells * sizeof(int)));
#endif
}

HashGrid2D::~HashGrid2D()
{
#ifdef USE_CUDA
    if (d_cellStart) cudaFree(d_cellStart);
    if (d_cellEnd) cudaFree(d_cellEnd);
    if (d_particleHash) cudaFree(d_particleHash);
    if (d_particleIndex) cudaFree(d_particleIndex);
#endif
}

void HashGrid2D::build(const float* d_posX, const float* d_posY, int n)
{
    if (n > maxParticles) {
        if (d_particleHash) cudaFree(d_particleHash);
        if (d_particleIndex) cudaFree(d_particleIndex);
        CUDA_CHECK(cudaMalloc(&d_particleHash, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_particleIndex, n * sizeof(int)));
        maxParticles = n;
    }
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    computeHashKernel<<<blocks, threads>>>(d_posX, d_posY,
                                           d_particleHash, d_particleIndex,
                                           cellSize, gridWidth, gridHeight, n);
    CUDA_KERNEL_CHECK();

    thrust::device_ptr<int> hashPtr(d_particleHash);
    thrust::device_ptr<int> indexPtr(d_particleIndex);
    thrust::sort_by_key(hashPtr, hashPtr + n, indexPtr);

    blocks = (numCells + threads - 1) / threads;
    clearGridKernel<<<blocks, threads>>>(d_cellStart, d_cellEnd, numCells);
    CUDA_KERNEL_CHECK();

    blocks = (n + threads - 1) / threads;
    findCellStartEndKernel<<<blocks, threads>>>(d_particleHash, d_particleIndex,
                                               d_cellStart, d_cellEnd, n);
    CUDA_KERNEL_CHECK();
}

} // namespace sph

#endif // USE_CUDA
