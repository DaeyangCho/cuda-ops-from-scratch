// lt_sgemm.cu
// FP32 GEMM benchmark with cuBLASLt on NVIDIA GPUs (row-major A,B,C).
// Usage: ./lt_sgemm M N K [--tf32]

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CHECK_CUDA(x)                                                     \
    do {                                                                  \
        cudaError_t err = (x);                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                             \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

#define CHECK_CUBLAS(x)                                                     \
    do {                                                                    \
        cublasStatus_t st = (x);                                            \
        if (st != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, \
                    int(st));                                               \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static void fill_random(float* p, size_t n) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (size_t i = 0; i < n; ++i) p[i] = dist(rng);
}

static float gflops(long long M, long long N, long long K, float ms) {
    // 2*M*N*K flops for GEMM
    double ops = 2.0 * double(M) * double(N) * double(K);
    return float(ops / (ms * 1e6));
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s M N K [--tf32]\n", argv[0]);
        return 0;
    }
    int64_t M = atoll(argv[1]);
    int64_t N = atoll(argv[2]);
    int64_t K = atoll(argv[3]);
    bool allow_tf32 = false;
    if (argc >= 5 && std::string(argv[4]) == "--tf32") allow_tf32 = true;

    printf(
        "GEMM: C[M=%lld,N=%lld] = A[M=%lld,K=%lld] * B[K=%lld,N=%lld]  "
        "(row-major)\n",
        (long long)M, (long long)N, (long long)M, (long long)K, (long long)K,
        (long long)N);
    printf("Compute: %s\n", allow_tf32
                                ? "CUBLAS_COMPUTE_32F_FAST_TF32 (TF32 allowed)"
                                : "CUBLAS_COMPUTE_32F (strict FP32)");

    // Host data
    std::vector<float> hA(M * K), hB(K * N), hC(M * N), hC_ref(M * N);
    fill_random(hA.data(), hA.size());
    fill_random(hB.data(), hB.size());
    std::fill(hC.begin(), hC.end(), 0.f);
    std::fill(hC_ref.begin(), hC_ref.end(), 0.f);

    // Device buffers
    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dCref = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(float) * hA.size()));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * hB.size()));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * hC.size()));
    CHECK_CUDA(cudaMalloc(&dCref, sizeof(float) * hC_ref.size()));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(float) * hA.size(),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(float) * hB.size(),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * hC.size()));
    CHECK_CUDA(cudaMemset(dCref, 0, sizeof(float) * hC_ref.size()));

    // cuBLASLt handle
    cublasLtHandle_t ltHandle;
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));

    // Matmul descriptor
    cublasLtMatmulDesc_t opDesc;
    cublasComputeType_t computeType =
        allow_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cudaDataType_t scaleType = CUDA_R_32F;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, computeType, scaleType));

    // A, B, C are row-major (no transposes)
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Matrix layouts (row-major)
    cublasLtMatrixLayout_t aDesc, bDesc, cDesc;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
        &aDesc, CUDA_R_32F, K, M,
        K));  // dims are columns, rows, ld? No—Lt uses m,n,ld with order
              // attribute below.
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_32F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_32F, N, M, N));

    // Set row-major order explicitly
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        aDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        bDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        cDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    // Leading dimensions for row-major: ld = number of columns
    int64_t lda = K, ldb = N, ldc = N;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        aDesc, CUBLASLT_MATRIX_LAYOUT_LD, &lda, sizeof(lda)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        bDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ldb, sizeof(ldb)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        cDesc, CUBLASLT_MATRIX_LAYOUT_LD, &ldc, sizeof(ldc)));

    // Heuristic search for good algorithms
    const size_t workspaceSize = 64 * 1024 * 1024;  // 64 MB
    void* dWorkspace = nullptr;
    CHECK_CUDA(cudaMalloc(&dWorkspace, workspaceSize));

    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
        sizeof(workspaceSize)));

    const int maxAlgos = 32;
    cublasLtMatmulHeuristicResult_t results[maxAlgos];
    int returnedResults = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, opDesc, aDesc, bDesc, cDesc, cDesc, preference, maxAlgos,
        results, &returnedResults));
    printf("Heuristic returned %d candidate algorithms\n", returnedResults);
    if (returnedResults == 0) {
        fprintf(stderr, "No algorithms found.\n");
        return 1;
    }

    float alpha = 1.f, beta = 0.f;

    // Warmup + timing each algo (few iterations)
    float bestMs = 1e9f;
    int bestIdx = -1;

    for (int i = 0; i < returnedResults; ++i) {
        // Quick functional run
        CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * hC.size()));
        CHECK_CUBLAS(cublasLtMatmul(
            ltHandle, opDesc, &alpha, dA, aDesc, dB, bDesc, &beta, dC, cDesc,
            dC, cDesc, &results[i].algo, dWorkspace, workspaceSize, 0));

        // Timing
        cudaEvent_t e0, e1;
        CHECK_CUDA(cudaEventCreate(&e0));
        CHECK_CUDA(cudaEventCreate(&e1));

        // Few warmups
        for (int w = 0; w < 3; ++w) {
            CHECK_CUBLAS(cublasLtMatmul(ltHandle, opDesc, &alpha, dA, aDesc, dB,
                                        bDesc, &beta, dC, cDesc, dC, cDesc,
                                        &results[i].algo, dWorkspace,
                                        workspaceSize, 0));
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        const int iters = 10;
        CHECK_CUDA(cudaEventRecord(e0));
        for (int it = 0; it < iters; ++it) {
            CHECK_CUBLAS(cublasLtMatmul(ltHandle, opDesc, &alpha, dA, aDesc, dB,
                                        bDesc, &beta, dC, cDesc, dC, cDesc,
                                        &results[i].algo, dWorkspace,
                                        workspaceSize, 0));
        }
        CHECK_CUDA(cudaEventRecord(e1));
        CHECK_CUDA(cudaEventSynchronize(e1));
        float ms = 0.f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        ms /= iters;

        float perf = gflops(M, N, K, ms);
        printf("Algo %2d: %.3f ms, %.1f GFLOP/s\n", i, ms, perf);

        if (ms < bestMs) {
            bestMs = ms;
            bestIdx = i;
        }

        CHECK_CUDA(cudaEventDestroy(e0));
        CHECK_CUDA(cudaEventDestroy(e1));
    }

    printf("\nBest: algo %d => %.3f ms  (%.1f GFLOP/s)\n", bestIdx, bestMs,
           gflops(M, N, K, bestMs));

    // Optional: compare with legacy cuBLAS sgemm for correctness (quick sanity)
    {
        cublasHandle_t handle;
        CHECK_CUBLAS(cublasCreate(&handle));
        // Legacy cuBLAS is column-major; emulate row-major by swapping &
        // transposing: C_row = A_row * B_row  ≡  C_col^T = B_col^T * A_col^T
        const float one = 1.f, zero = 0.f;
        CHECK_CUDA(cudaMemset(dCref, 0, sizeof(float) * hC_ref.size()));
        CHECK_CUBLAS(cublasSetMathMode(handle, allow_tf32
                                                   ? CUBLAS_TF32_TENSOR_OP_MATH
                                                   : CUBLAS_DEFAULT_MATH));
        // Use column-major view: lda_row=K -> in col-major treat as ldb_col=K
        // etc. Compute C^T (N x M) = B^T (N x K) * A^T (K x M)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                 &one, dB, N, dA, K, &zero, dCref, N));
        CHECK_CUBLAS(cublasDestroy(handle));

        // Compare dC (Lt result) vs dCref (legacy result) elementwise
        CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeof(float) * hC.size(),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hC_ref.data(), dCref,
                              sizeof(float) * hC_ref.size(),
                              cudaMemcpyDeviceToHost));
        double max_abs = 0.0, max_rel = 0.0, denom_eps = 1e-6;
        for (size_t i = 0; i < hC.size(); ++i) {
            double a = hC[i], b = hC_ref[i];
            double abs_err = std::abs(a - b);
            double rel_err = abs_err / (std::abs(b) + denom_eps);
            if (abs_err > max_abs) max_abs = abs_err;
            if (rel_err > max_rel) max_rel = rel_err;
        }
        printf("Check vs cuBLAS sgemm: max_abs=%.3e  max_rel=%.3e\n", max_abs,
               max_rel);
    }

    // Cleanup
    CHECK_CUDA(cudaFree(dWorkspace));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(aDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(bDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(cDesc));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(opDesc));
    CHECK_CUBLAS(cublasLtDestroy(ltHandle));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaFree(dCref));

    return 0;
}
