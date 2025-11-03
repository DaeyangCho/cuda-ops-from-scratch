import time
import torch


def bench(M=4096, N=4096, K=4096, iters=100):
    # Create random matrices on GPU: A(M×K), B(KxN)
    A = torch.rand((M, K), device='cuda')
    B = torch.rand((K, N), device='cuda')

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
