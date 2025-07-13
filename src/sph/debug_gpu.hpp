#pragma once
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifdef DEBUG_GPU
#undef CUDA_KERNEL_CHECK
#define CUDA_TRY(x) do { \
  cudaError_t _e = (x); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA-error %s at %s:%d \xE2\x80\x94 %s\n", \
            #x, __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)
#define CUDA_KERNEL_CHECK() do { \
  CUDA_TRY(cudaGetLastError()); \
  CUDA_TRY(cudaDeviceSynchronize()); \
} while (0)
#else
#define CUDA_TRY(x) (x)
#define CUDA_KERNEL_CHECK() CUDA_TRY(cudaGetLastError())
#endif // DEBUG_GPU

#endif // USE_CUDA
