"""Hardware-ceiling microbenchmarks: device-to-device copy bandwidth and FP32 FMA throughput."""

from __future__ import annotations

import numpy as np


def copy_bandwidth(nbytes: int = 2**30, iters: int = 20, repeats: int = 5) -> float:
    """Device-to-device ``copy_`` bandwidth in GB/s (read + write counted), median of repeats."""
    import torch

    n = nbytes // 4
    a = torch.empty(n, device="cuda", dtype=torch.float32).uniform_()
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            b.copy_(a)
        e.record()
        e.synchronize()
        out.append(2.0 * 4 * n * iters / (s.elapsed_time(e) * 1e-3) / 1e9)
    del a, b
    torch.cuda.empty_cache()
    return float(np.median(out))


_FMA_SRC = r"""
extern "C" __global__ void fma_peak(float* out, int iters, float a, float b) {
    float x0 = threadIdx.x, x1 = x0 + 1.f, x2 = x0 + 2.f, x3 = x0 + 3.f;
    float x4 = x0 + 4.f, x5 = x0 + 5.f, x6 = x0 + 6.f, x7 = x0 + 7.f;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            x0 = fmaf(x0, a, b); x1 = fmaf(x1, a, b); x2 = fmaf(x2, a, b); x3 = fmaf(x3, a, b);
            x4 = fmaf(x4, a, b); x5 = fmaf(x5, a, b); x6 = fmaf(x6, a, b); x7 = fmaf(x7, a, b);
        }
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
"""


def fma_tflops(iters: int = 4096, repeats: int = 5) -> float:
    """Peak FP32 FMA throughput in TFLOP/s (2 FLOPs per FMA), median of repeats."""
    from .backends.cupy_backend import cp

    k = cp.RawKernel(_FMA_SRC, "fma_peak")
    blocks, threads = 40 * 16, 256
    out = cp.empty(blocks * threads, dtype=cp.float32)
    args = (out, np.int32(iters), np.float32(0.999), np.float32(0.001))
    k((blocks,), (threads,), args)
    cp.cuda.runtime.deviceSynchronize()
    res = []
    for _ in range(repeats):
        s, e = cp.cuda.Event(), cp.cuda.Event()
        s.record()
        k((blocks,), (threads,), args)
        e.record()
        e.synchronize()
        ms = cp.cuda.get_elapsed_time(s, e)
        res.append(2.0 * 8 * 16 * iters * blocks * threads / (ms * 1e-3) / 1e12)
    return float(np.median(res))
