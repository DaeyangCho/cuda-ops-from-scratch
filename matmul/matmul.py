import time
import torch


def bench(M=1024, N=1024, K=1024, iters=100):
    # Create random matrices on GPU: A(M×N), B(N×K)
    A = torch.rand((M, N), device='cuda')
    B = torch.rand((N, K), device='cuda')

    # Warm-up
    for _ in range(10):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    dt = (time.time() - t0) * 1000 / iters

    print(f"PyTorch matrix multiply: M={M}, N={N}, K={K}, {dt:.4f} ms")


if __name__ == '__main__':
    bench()
