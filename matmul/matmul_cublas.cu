#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "../utils/timer.h"

#define CUBLAS_CHECK(x)                                                        \
    do {                                                                       \
        cublasStatus_t st__ = (x);                                             \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS Error: %d (%s:%d)\n", int(st__), __FILE__, \
                    __LINE__);                                                 \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

int main() {
    // Matrix dimensions: A(M×K), B(K×N), C(M×N) in row-major
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    std::vector<float> ha(M * K), hb(K * N), hc(M * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < M * K; ++i) ha[i] = dist(rng);
    for (int i = 0; i < K * N; ++i) hb[i] = dist(rng);

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(da, ha.data(), M * K * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), K * N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // (Optional) Enable TF32: Uses Tensor Cores with FP32 input → faster but
    // may cause slightly higher numerical error.
    // CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    // Default: strict FP32 computation.
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    // cuBLAS assumes column-major order:
    // From a row-major perspective, C = A(M×K) * B(K×N)
    // corresponds to C^T = B^T (N×K) * A^T (K×M) in column-major form.
    // Therefore, for the sgemm call, set (m = N, n = M, k = K),
    // opA = N, opB = N, A = db (ldA = N), B = da (ldB = K), and C = dc (ldC =
    // N).
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             db, N, da, K, &beta, dc, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer t;
    t.start();
    for (int r = 0; r < 100; ++r) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                 &alpha, db, N, da, K, &beta, dc, N));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = t.stop_ms() / 100.0f;

    CUDA_CHECK(cudaMemcpy(hc.data(), dc, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float max_err = 0.0;
    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         float expected = 0.0f;
    //         for (int k = 0; k < K; ++k) {
    //             expected += ha[i * K + k] * hb[k * N + j];
    //         }
    //         max_err = std::max(max_err, std::fabs(hc[i * N + j] - expected));
    //     }
    // }

    printf("cuBLAS SGEMM: M=%d, N=%d, K=%d, time=%.4f ms, max_err=%.3g\n", M, N,
           K, ms, max_err);

    CUBLAS_CHECK(cublasDestroy(handle));
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;
}
