#include "world.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "cuda_utils.h"

namespace sph {

__global__ void predictedPosKernel(float* pos, float* vel, float* predpos,
                                   float gravity, float dt, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        vel[idx*2 + 1] += gravity * dt;
        predpos[idx*2 + 0] = pos[idx*2 + 0] + vel[idx*2 + 0] * (1.0f/120.0f);
        predpos[idx*2 + 1] = pos[idx*2 + 1] + vel[idx*2 + 1] * (1.0f/120.0f);
    }
}

void predictedPosCUDA(float* d_pos, float* d_vel, float* d_predpos,
                      float gravity, float dt, int n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    predictedPosKernel<<<blocks, threads>>>(d_pos, d_vel, d_predpos,
                                            gravity, dt, n);
    CUDA_KERNEL_CHECK();
}

__global__ void updatePositionKernel(float* pos, float* vel,
                                     const float* pressure,
                                     const float* interaction,
                                     float drag, float dt, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        vel[idx*2 + 0] += (pressure[idx*2 + 0] + interaction[idx*2 + 0]) * dt;
        vel[idx*2 + 1] += (pressure[idx*2 + 1] + interaction[idx*2 + 1]) * dt;
        pos[idx*2 + 0] += vel[idx*2 + 0] * dt;
        pos[idx*2 + 1] += vel[idx*2 + 1] * dt;
        vel[idx*2 + 0] *= drag;
        vel[idx*2 + 1] *= drag;
    }
}

void updatePositionCUDA(float* d_pos, float* d_vel, float* d_pressure,
                        float* d_interaction, float drag, float dt, int n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    updatePositionKernel<<<blocks, threads>>>(d_pos, d_vel,
                                             d_pressure, d_interaction,
                                             drag, dt, n);
    CUDA_KERNEL_CHECK();
}

__global__ void fixPositionKernel(float* pos, float* vel,
                                  float width, float height,
                                  float damping, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = pos[idx*2 + 0];
        float y = pos[idx*2 + 1];
        float vx = vel[idx*2 + 0];
        float vy = vel[idx*2 + 1];
        if (x < 0) { pos[idx*2 + 0] = 0; vel[idx*2 + 0] = -vx * damping; }
        if (width < x) { pos[idx*2 + 0] = width; vel[idx*2 + 0] = -vx * damping; }
        if (y < 0) { pos[idx*2 + 1] = 0; vel[idx*2 + 1] = -vy * damping; }
        if (height < y) { pos[idx*2 + 1] = height; vel[idx*2 + 1] = -vy * damping; }
    }
}

void fixPositionCUDA(float* d_pos, float* d_vel, float width, float height,
                     float damping, int n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fixPositionKernel<<<blocks, threads>>>(d_pos, d_vel, width, height,
                                          damping, n);
    CUDA_KERNEL_CHECK();
}

} // namespace sph

#endif // USE_CUDA
