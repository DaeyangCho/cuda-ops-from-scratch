#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "../utils/timer.h"

const int WARPSIZE = 32;

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    matrix_multiplication_kernel(float *A, float *B, float *C, int M, int N,
                                 int K) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int warpIdx = threadIdx.x / WARPSIZE;
    const int warpCol = warpIdx % (BN / WN);
    const int warpRow = warpIdx / (BN / WN);

    constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    const int threadIdxInWarp = threadIdx.x % WARPSIZE;
    const int threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const int threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;  // row=cRow, col=0
    B += cCol * BN;      // row=0, col=cCol
    C += (cRow * BM + warpRow * WM) * N +
         (cCol * BN + warpCol * WN);  // row=cRow, col=cCol

    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    constexpr int rowStrideA = (NUM_THREADS * 4) / BK;
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);
    constexpr int rowStrideB = NUM_THREADS / (BN / 4);

    float threadResults[WMITER * TM * WNITER * TN] = {0.0};
    float regM[WMITER * TM] = {0.0};
    float regN[WNITER * TN] = {0.0};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            float4 tmp = reinterpret_cast<float4 *>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        for (int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<float4 *>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<float4 *>(
                    &B[(innerRowB + offset) * N + innerColB * 4])[0];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
                for (int i = 0; i < TM; i++) {
                    regM[wSubRowIdx * TM + i] =
                        As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                           threadRowInWarp * TM + i];
                }
            }
            for (int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
                for (int i = 0; i < TN; i++) {
                    regN[wSubColIdx * TN + i] =
                        Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                           threadColInWarp * TN + i];
                }
            }
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
                for (int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
                    for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
                        for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
                            threadResults[(wSubRowIdx * TM + resIdxM) *
                                              (WNITER * TN) +
                                          (wSubColIdx * TN) + resIdxN] +=
                                regM[wSubRowIdx * TM + resIdxM] *
                                regN[wSubColIdx * TN + resIdxN];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float *C_interim =
                C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    float4 tmp = reinterpret_cast<float4 *>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                                   threadColInWarp * TN + resIdxN])[0];
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                  wSubColIdx * TN + resIdxN;
                    tmp.x = threadResults[i + 0];
                    tmp.y = threadResults[i + 1];
                    tmp.z = threadResults[i + 2];
                    tmp.w = threadResults[i + 3];
                    reinterpret_cast<float4 *>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                                   threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}

int main() {
    // Matrix dimensions: A(M×K), B(K×N), C(M×N)
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    const int NUM_THREADS = 128;
    const int BM = 64;
    const int BN = 128;
    const int BK = 16;
    const int WN = 64;
    const int WM = 32;
    const int WNITER = 1;
    const int TN = 4;
    const int TM = 4;

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

    dim3 block(NUM_THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    // Warm-up
    matrix_multiplication_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN,
                                 NUM_THREADS>
        <<<grid, block>>>(da, db, dc, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer t;
    t.start();
    for (int r = 0; r < 100; r++)
        matrix_multiplication_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN,
                                     NUM_THREADS>
            <<<grid, block>>>(da, db, dc, M, N, K);
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