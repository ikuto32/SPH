#include "world.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "cuda_utils.h"

namespace sph {

__global__ void predictedPosKernel(float* posX, float* posY,
                                   float* velX, float* velY,
                                   float* predX, float* predY,
                                   float gravity, float dt, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        velY[idx] += gravity * dt;
        predX[idx] = posX[idx] + velX[idx] * dt;
        predY[idx] = posY[idx] + velY[idx] * dt;
    }
}

void predictedPosCUDA(float* d_posX, float* d_posY,
                      float* d_velX, float* d_velY,
                      float* d_predX, float* d_predY,
                      float gravity, float dt, int n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    predictedPosKernel<<<blocks, threads>>>(d_posX, d_posY,
                                            d_velX, d_velY,
                                            d_predX, d_predY,
                                            gravity, dt, n);
    CUDA_KERNEL_CHECK();
}

__global__ void updatePositionKernel(float* posX, float* posY,
                                     float* velX, float* velY,
                                     const float* pressureX,
                                     const float* pressureY,
                                     const float* interactionX,
                                     const float* interactionY,
                                     float drag, float dt, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        velX[idx] += (pressureX[idx] + interactionX[idx]) * dt;
        velY[idx] += (pressureY[idx] + interactionY[idx]) * dt;
        posX[idx] += velX[idx] * dt;
        posY[idx] += velY[idx] * dt;
        velX[idx] *= drag;
        velY[idx] *= drag;
    }
}

void updatePositionCUDA(float* d_posX, float* d_posY,
                        float* d_velX, float* d_velY,
                        float* d_pressureX, float* d_pressureY,
                        float* d_interactionX, float* d_interactionY,
                        float drag, float dt, int n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    updatePositionKernel<<<blocks, threads>>>(d_posX, d_posY,
                                             d_velX, d_velY,
                                             d_pressureX, d_pressureY,
                                             d_interactionX, d_interactionY,
                                             drag, dt, n);
    CUDA_KERNEL_CHECK();
}

__global__ void fixPositionKernel(float* posX, float* posY,
                                  float* velX, float* velY,
                                  float width, float height,
                                  float damping, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = posX[idx];
        float y = posY[idx];
        float vx = velX[idx];
        float vy = velY[idx];
        if (x < 0) { posX[idx] = 0; velX[idx] = -vx * damping; }
        if (width < x) { posX[idx] = width; velX[idx] = -vx * damping; }
        if (y < 0) { posY[idx] = 0; velY[idx] = -vy * damping; }
        if (height < y) { posY[idx] = height; velY[idx] = -vy * damping; }
    }
}

void fixPositionCUDA(float* d_posX, float* d_posY,
                     float* d_velX, float* d_velY,
                     float width, float height,
                     float damping, int n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fixPositionKernel<<<blocks, threads>>>(d_posX, d_posY,
                                          d_velX, d_velY,
                                          width, height,
                                          damping, n);
    CUDA_KERNEL_CHECK();
}

} // namespace sph

#endif // USE_CUDA
