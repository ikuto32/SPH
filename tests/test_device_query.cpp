#include <cstdio>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

int main(){
#ifdef USE_CUDA
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(e));
        return 0;
    }
    printf("device count: %d\n", count);
    if (count > 0) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        printf("device[0] compute capability: %d.%d\n", prop.major, prop.minor);
    }
#else
    printf("compiled without CUDA support\n");
#endif
    return 0;
}
