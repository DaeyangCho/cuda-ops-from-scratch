import time
import torch


def bench(n=100_000_000, iters=200):
    x = torch.rand(n, device='cuda')
    y = torch.rand(n, device='cuda')
    # Warm-up
    for _ in range(10):
        z = x + y
    torch.cuda.synchronize()
    t0=time.time()
    for _ in range(iters):
        z = x + y
    torch.cuda.synchronize()
    dt=(time.time()-t0)*1000/iters
    print(f"PyTorch vector add: N={n}, {dt:.4f} ms")

if __name__ == '__main__':
    bench()