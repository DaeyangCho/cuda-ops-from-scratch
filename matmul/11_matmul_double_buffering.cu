#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "../utils/timer.h"

#define NUM_THREADS 256
#define BN 256
#define BM 128
#define BK 16
#define WN 32
#define WM 128
#define WNITER 1
#define TN 8
#define TM 8
#define WARPSIZE 32


template <int rowStrideA, int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
        const float4 tmp = reinterpret_cast<const float4 *>(
            &A[(innerRowA + offset) * K + innerColA * 4])[0];
        // float4 tmp;
        // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
        //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
        //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
        As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
        reinterpret_cast<float4 *>(
            &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4 *>(
                &B[(innerRowB + offset) * N + innerColB * 4])[0];
        // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
        //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
        //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
        //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
        //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
        //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
    }
}

template <const int WMITER, const int WSUBM, const int WSUBN>
__device__ void processFromSmem(float *regM, float *regN, float *threadResults,
                                const float *As, const float *Bs,
                                const uint warpRow, const uint warpCol,
                                const uint threadRowInWarp,
                                const uint threadColInWarp) {
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        // populate registers for whole warptile
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint i = 0; i < TM; ++i) {
                regM[wSubRowIdx * TM + i] =
                    As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                       threadRowInWarp * TM + i];
            }
        }
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            for (uint i = 0; i < TN; ++i) {
                regN[wSubColIdx * TN + i] =
                    Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                       threadColInWarp * TN + i];
            }
        }

        // execute warptile matmul
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                // calculate per-thread results
                for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
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
}

__global__ void __launch_bounds__(NUM_THREADS)
    matrix_multiplication_kernel(float *A, float *B, float *C, int M, int N,
                                 int K) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // Placement of the warp in the threadblock tile
    const uint warpIdx = threadIdx.x / WARPSIZE;  // the warp this thread is in
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);

    // size of the warp subtile
    const uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    const uint WSUBM = WM / WMITER;  // 64/2=32
    const uint WSUBN = WN / WNITER;  // 32/2=16

    // Placement of the thread in the warp subtile
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;          // [0, 31]
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);  // i%(16/4)
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);  // i/4

    // allocate space for the current blocktile in SMEM
    __shared__ float As[2 * BM * BK];
    __shared__ float Bs[2 * BK * BN];

    // setup double buffering split
    bool doubleBufferIdx = threadIdx.x >= (NUM_THREADS / 2);

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    // Move C_ptr to warp's output tile
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // calculating the indices that this thread will load into SMEM
    // for the loading, we're pretending like there's half as many threads
    // as there actually are
    const uint innerRowA = (threadIdx.x % (NUM_THREADS / 2)) / (BK / 4);
    const uint innerColA = (threadIdx.x % (NUM_THREADS / 2)) % (BK / 4);
    const uint rowStrideA = ((NUM_THREADS / 2) * 4) / BK;
    const uint innerRowB = (threadIdx.x % (NUM_THREADS / 2)) / (BN / 4);
    const uint innerColB = (threadIdx.x % (NUM_THREADS / 2)) % (BN / 4);
    const uint rowStrideB = (NUM_THREADS / 2) / (BN / 4);

    // allocate thread-local cache for results in registerfile
    float threadResults[WMITER * TM * WNITER * TN] = {0.0};
    // we cache into registers on the warptile level
    float regM[WMITER * TM] = {0.0};
    float regN[WNITER * TN] = {0.0};

    if (doubleBufferIdx == 0) {
        // load first (B0)
        loadFromGmem<rowStrideA, rowStrideB>(
            N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    }
    __syncthreads();

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += 2 * BK) {
        if (doubleBufferIdx == 0) {
            // process current (B0)
            processFromSmem<WMITER, WSUBM, WSUBN>(regM, regN, threadResults, As, Bs, warpRow,
                                    warpCol, threadRowInWarp, threadColInWarp);
            __syncthreads();

            // process current+1 (B1)
            if (bkIdx + BK < K) {
                processFromSmem<WMITER, WSUBM, WSUBN>(
                    regM, regN, threadResults, As + (BM * BK), Bs + (BK * BN),
                    warpRow, warpCol, threadRowInWarp, threadColInWarp);
            }
            __syncthreads();

            // load current + 2 (B0)
            if (bkIdx + 2 * BK < K) {
                loadFromGmem<rowStrideA, rowStrideB>(
                    N, K, A + 2 * BK, B + 2 * BK * N, As, Bs, innerRowA,
                    innerColA, innerRowB, innerColB);
            }
        } else {
            // load current + 1 (B1)
            if (bkIdx + BK < K) {
                loadFromGmem<rowStrideA, rowStrideB>(
                    N, K, A + BK, B + BK * N, As + (BM * BK), Bs + (BK * BN),
                    innerRowA, innerColA, innerRowB, innerColB);
            }
            __syncthreads();

            // process current (B0)
            processFromSmem<WMITER, WSUBM, WSUBN>(regM, regN, threadResults, As, Bs, warpRow,
                                    warpCol, threadRowInWarp, threadColInWarp);
            __syncthreads();

            // process current+1 (B1)
            if (bkIdx + BK < K) {
                processFromSmem<WMITER, WSUBM, WSUBN>(
                    regM, regN, threadResults, As + (BM * BK), Bs + (BK * BN),
                    warpRow, warpCol, threadRowInWarp, threadColInWarp);
            }
        }

        A += 2 * BK;      // move BK columns to right
        B += 2 * BK * N;  // move BK rows down
        __syncthreads();
    }

    // write out the results
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            // move C pointer to current warp subtile
            float *C_interim =
                C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    // load C vector into registers
                    float4 tmp = reinterpret_cast<float4 *>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                                   threadColInWarp * TN + resIdxN])[0];
                    // perform GEMM update in reg
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                  wSubColIdx * TN + resIdxN;
                    tmp.x = threadResults[i + 0];
                    tmp.y = threadResults[i + 1];
                    tmp.z = threadResults[i + 2];
                    tmp.w = threadResults[i + 3];
                    // write back
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