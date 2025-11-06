#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <vector>
#include "../utils/timer.h"

#define BN 1024

__global__ void sum_reduction(float* input, float* output, int N) {
    __shared__ float sdata[BN];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[idx];
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + stride];
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}


int main() {
    const int N = BN;

    // Host init
    std::vector<float> ha(N), hc(1);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < N; ++i) ha[i] = dist(rng);

    // Device alloc
    float *da, *dc;
    CUDA_CHECK(cudaMalloc(&da, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, 1 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(da, ha.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block(BN);
    dim3 grid(1);

    // Warm-up
    sum_reduction<<<grid, block>>>(da, dc, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer t;
    t.start();
    for (int r = 0; r < 100; r++)
        sum_reduction<<<grid, block>>>(da, dc, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = t.stop_ms() / 100.0f;

    CUDA_CHECK(cudaMemcpy(hc.data(), dc, 1 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Verify
    double max_err = 0.0;
    float expected = 0.0f;
    for (int i=0; i<N; i++) {
        expected += ha[i];
    }
    max_err = std::max(max_err, static_cast<double>(std::fabs(hc[0] - expected)));

    printf(
        "cuda naive sum reduction: N=%d, time=%.4f ms, expected: %.4f, result: %.4f, "
        "max_err=%.3g\n", N, ms, expected, hc[0], max_err);

    cudaFree(da);
    cudaFree(dc);
    return 0;
}