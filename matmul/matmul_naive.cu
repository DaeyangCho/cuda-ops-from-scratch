#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "../utils/timer.h"

__global__ void matrix_multiplication_kernel(const float* A, const float* B,
                                             float* C, int M, int N, int K) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < M && c < K) {
        float v = 0.0;
        for (int i = 0; i < N; i++) {
            v += A[(r * N) + i] * B[(i * K) + c];
        }
        C[(r * K) + c] = v;
    }
}

int main() {
    // Matrix dimensions: A(M×N), B(N×K), C(M×K)
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    // Host init
    std::vector<float> ha(M * N), hb(N * K), hc(M * K);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < M * N; ++i) ha[i] = dist(rng);
    for (int i = 0; i < N * K; ++i) hb[i] = dist(rng);

    // Device alloc
    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, M * K * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(da, ha.data(), M * N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), N * K * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Warm-up
    matrix_multiplication_kernel<<<grid, block>>>(da, db, dc, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer t;
    t.start();
    for (int r = 0; r < 100; r++)
        matrix_multiplication_kernel<<<grid, block>>>(da, db, dc, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = t.stop_ms() / 100.0f;

    CUDA_CHECK(cudaMemcpy(hc.data(), dc, M * K * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Verify
    double max_err = 0.0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            float expected = 0.0f;
            for (int k = 0; k < N; ++k) {
                expected += ha[i * N + k] * hb[k * K + j];
            }
            max_err = std::max(
                max_err,
                static_cast<double>(std::fabs(hc[i * K + j] - expected)));
        }
    }

    printf(
        "cuda naive matrix_multiply: M=%d, N=%d, K=%d, time=%.4f ms, "
        "max_err=%.3g\n",
        M, N, K, ms, max_err);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;
}