#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>
#include <iostream>

#ifndef MAX_THREADS
#define MAX_THREADS 512
#endif

#if  __CUDA_ARCH__ < 600 and defined(__CUDA_ARCH__)
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

#define GPU_ERROR_CHECK(ans) {gpu_assert((ans), __FILE__, __LINE__);}
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"\nGPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

inline int64_t gpu_blocks(int64_t total_threads, int64_t threads_per_block) {
    return (total_threads + threads_per_block - 1) / threads_per_block;
}
