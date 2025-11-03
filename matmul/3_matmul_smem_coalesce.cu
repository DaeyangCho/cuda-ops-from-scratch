#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "../utils/timer.h"

#define BLOCK_SIZE 32

__global__ void matrix_multiplication_kernel(const float* A, const float* B,
                                             float* C, int M, int N, int K) {
    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;

    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    const int threadCol = threadIdx.x % BLOCK_SIZE;
    const int threadRow = threadIdx.x / BLOCK_SIZE;

    A += cRow * BLOCK_SIZE * K;                      // row=cRow, col=0
    B += cCol * BLOCK_SIZE;                          // row=0, col=cCol
    C += cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE;  // row=cRow, col=cCol

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE) {
        As[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * N + threadCol];

        __syncthreads();
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;

        for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCK_SIZE + dotIdx] *
                   Bs[dotIdx * BLOCK_SIZE + threadCol];
        }
        __syncthreads();
    }
    C[threadRow * N + threadCol] = tmp;
}

int main() {
    // Matrix dimensions: A(M×K), B(K×N), C(M×N)
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    // Host init
    std::vector<float> ha(M * K), hb(K * N), hc(M * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < M * K; ++i) ha[i] = dist(rng);
    for (int i = 0; i < K * N; ++i) hb[i] = dist(rng);

    // Device alloc
    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(da, ha.data(), M * K * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), K * N * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE * BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Warm-up
    matrix_multiplication_kernel<<<grid, block>>>(da, db, dc, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer t;
    t.start();
    for (int r = 0; r < 100; r++)
        matrix_multiplication_kernel<<<grid, block>>>(da, db, dc, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = t.stop_ms() / 100.0f;

    CUDA_CHECK(cudaMemcpy(hc.data(), dc, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Verify
    double max_err = 0.0;
    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         float expected = 0.0f;
    //         for (int k = 0; k < K; ++k) {
    //             expected += ha[i * M + k] * hb[k * N + j];
    //         }
    //         max_err = std::max(
    //             max_err,
    //             static_cast<double>(std::fabs(hc[i * N + j] - expected)));
    //     }
    // }

    printf(
        "cuda shared mem matrix_multiply: M=%d, N=%d, K=%d, time=%.4f ms, "
        "max_err=%.3g\n",
        M, N, K, ms, max_err);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;
}