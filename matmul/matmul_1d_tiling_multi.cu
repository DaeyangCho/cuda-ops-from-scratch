#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "../utils/timer.h"

#define BM 64
#define BN 64
#define BK 8
#define TM 8

__global__ void matrix_multiplication_kernel(const float* A, const float* B,
                                             float* C, int M, int N, int K) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;              // row=cRow, col=0
    B += cCol * BN;                  // row=0, col=cCol
    C += cRow * BM * N + cCol * BN;  // row=cRow, col=cCol

    assert(BM * BK == blockDim.x);  // Need to ensure each thread loads one
                                    // element of A
    assert(BN * BK == blockDim.x);  // Need to ensure each thread loads one
                                    // element of B

    const int innerColA = threadIdx.x % BK;
    const int innerRowA = threadIdx.x / BK;
    const int innerColB = threadIdx.x % BN;
    const int innerRowB = threadIdx.x / BN;

    float threadResults[TM] = {0.0};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            float tmpB = Bs[dotIdx * BN + threadCol];
            for (int resIdx = 0; resIdx < TM; resIdx++) {
                threadResults[resIdx] +=
                    As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    for (int resIdx = 0; resIdx < TM; resIdx++) {
        C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
    }
}

int main() {
    // Matrix dimensions: A(M×K), B(K×N), C(M×N)
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

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

    dim3 block((BM * BN) / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

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
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k) {
                expected += ha[i * M + k] * hb[k * N + j];
            }
            max_err = std::max(
                max_err,
                static_cast<double>(std::fabs(hc[i * N + j] - expected)));
        }
    }

    printf(
        "cuda shared mem matrix_multiply: M=%d, N=%d, K=%d, time=%.4f ms, "
        "max_err=%.3g\n",
        M, N, K, ms, max_err);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;
}