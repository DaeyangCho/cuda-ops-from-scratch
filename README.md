# CUDA Operations from Scratch

A collection of PyTorch operations reimplemented from scratch in CUDA C++
to explore GPU parallelism, memory hierarchy, and optimization techniques.

---

## Overview

| Operation | Highlights | Comparison |
|------------|-------------|-------------|
| **Vector Addition** |  | CUDA vs PyTorch |
| **Matrix Multiplication** | From Naive to Optimal | CDUA vs PyTorch |
---

## Optimize Matrix Multiplication

| Kernel | GFLOPS | Performace gain |
|------------|-------------|-------------|
| Naive          | 295.2   | 2.5% |
| GMEM Coalesing | 1069.2  | 9.1% |
| SMEM Coalesing | 1483.0  | 12.6% |
| 1D Tiling      | 4222.8  | 35.8% |
| 2D Tiling      | 7710.3  | 65.3% |
| Vectorize      | 8988.7  | 76.2% |
| PyTorch        | 11801.6 | 100.0% |
| cuBLAS         | 11801.6 | 98.0% |
| cuBLAS TF32    | 11801.6 | 142.8% |
---

## Test Environment

- **GPU:** NVIDIA RTX 3090 (24GB GDDR6X)
- **PCIe:** Gen3 x16
- **CUDA Toolkit:** 12.4
- **Driver Version:** 535.171
- **PyTorch:** 2.8.0 + cu124
- **OS:** Ubuntu 22.04 LTS
- **Compiler:** nvcc 12.4, gcc 11.4
- **CPU / RAM:** AMD Ryzen Threadripper PRO 5965WX 24-Cores, 256 GB DDR4

---

## Example: Matrix Multiplication

### Naive kernel
```cpp
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int k = 0; k < N; ++k)
        sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
}
