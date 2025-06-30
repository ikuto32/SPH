#pragma once
#include <vector>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "../core/cuda_utils.h"
#endif

namespace sph {

struct HashGrid2D {
    float cellSize;
    int gridWidth;
    int gridHeight;
    int numCells;
    int maxParticles;

    // host arrays for debugging (unused currently)
    std::vector<int> h_cellStart;
    std::vector<int> h_cellEnd;
#ifdef USE_CUDA
    int* d_cellStart = nullptr;
    int* d_cellEnd = nullptr;
    int* d_particleHash = nullptr;
    int* d_particleIndex = nullptr;
#endif

    HashGrid2D(float width, float height, float cell);
    ~HashGrid2D();

#ifdef USE_CUDA
    void build(const float* d_posX, const float* d_posY, int n);
#endif
};

#ifdef USE_CUDA
void launchNeighbourSearch(const float* d_posX, const float* d_posY,
                           const HashGrid2D& grid, float radius, int n,
                           int* d_neighborCount);
#endif

} // namespace sph

