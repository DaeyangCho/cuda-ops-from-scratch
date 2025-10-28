#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                           \
    do {                                                           \
        cudaError_t _err = (expr);                                 \
        if (_err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA Error: %s (%s:%d)\n",            \
                    cudaGetErrorString(_err), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    } while (0)

struct CudaTimer {
    cudaEvent_t start_;
    cudaEvent_t stop_;

    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        CUDA_CHECK(cudaEventDestroy(start_));
        CUDA_CHECK(cudaEventDestroy(stop_));
    }

    void start() { CUDA_CHECK(cudaEventRecord(start_)); }

    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};
