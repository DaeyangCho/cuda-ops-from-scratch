#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <vector>
#include "../utils/timer.h"

__global__ void vector_add_kernel(const float *__restrict__ a,
                                  const float *__restrict__ b,
                                  float *__restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 200'000'000;
    const size_t bytes = N * sizeof(float);

    // Host init
    std::vector<float> ha(N), hb(N), hc(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < N; ++i) {
        ha[i] = dist(rng);
        hb[i] = dist(rng);
    }

    // Device alloc
    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));

    CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // Warm-up
    vector_add_kernel<<<grid, block>>>(da, db, dc, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer t;
    t.start();
    for (int r = 0; r < 100; r++)
        vector_add_kernel<<<grid, block>>>(da, db, dc, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = t.stop_ms() / 100.0f;

    CUDA_CHECK(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));

    // Verify
    double max_err = 0.0;
    for (int i = 0; i < N; ++i)
        max_err = std::max(max_err, (double)std::abs(hc[i] - (ha[i] + hb[i])));

    printf("vector_add: N=%d, time=%.4f ms, max_err=%.3g\n", N, ms, max_err);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;
}