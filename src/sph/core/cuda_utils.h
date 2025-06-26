#pragma once
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    } \
} while (0)

#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())
#endif
