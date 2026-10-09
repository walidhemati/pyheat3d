"""Development helper: sweep tile/launch parameters of the CUDA C and Triton variants.

Not part of the benchmark contract; timings on a shared GPU are noisy. Usage:

    python bench/tune.py cuda-tb3d --n 256
    python bench/tune.py cuda-tb2d --n 4096
    python bench/tune.py cuda-zmarch | cuda-naive3d | triton-2d | triton-3d | triton-zmarch
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from pyheat3d.backends import cupy_backend as cb
from pyheat3d.benchmark import time_variant


def _run(key: str, u0: np.ndarray, label: str, min_time: float) -> float:
    try:
        r = time_variant(key, u0, min_time, 5)
    except Exception as e:  # invalid configs (too many threads / smem) just fail
        print(f"{label:48s} failed: {str(e).splitlines()[0][:80]}")
        return 0.0
    extra = f" equiv {r['single_step_equiv_GBs']:.0f} GB/s" if "single_step_equiv_GBs" in r else ""
    # best-of-5: on a shared GPU contention only ever slows a repeat down
    print(f"{label:48s} best {r['max'] / 1e3:7.2f} GLUPS (median {r['median'] / 1e3:6.2f})"
          f" {r['max'] * r['bytes_per_unit'] / 1e3:6.0f} GB/s{extra}", flush=True)
    return r["max"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--min-time", type=float, default=0.1)
    ap.add_argument("--K", type=int, nargs="*", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    w = a.what
    dims = 2 if w in ("cuda-tb2d", "triton-2d", "cuda-naive2d") else 3
    n = a.n or (4096 if dims == 2 else 256)
    u0 = rng.random((n,) * dims, dtype=np.float32)
    mt = a.min_time
    if w == "cuda-naive3d":
        for blk in [(32, 8, 1), (32, 4, 2), (32, 4, 4), (64, 4, 1), (128, 2, 1), (32, 16, 1)]:
            cb.NAIVE3D_BLOCK = blk
            _run("cupy-cuda-c/naive", u0, f"naive3d {blk}", mt)
    elif w == "cuda-naive2d":
        for blk in [(32, 8, 1), (32, 4, 1), (64, 4, 1), (128, 2, 1), (32, 16, 1), (256, 1, 1)]:
            cb.NAIVE2D_BLOCK = blk
            _run("cupy-cuda-c/naive", u0, f"naive2d {blk}", mt)
    elif w == "cuda-zmarch":
        for bx, by, zch in itertools.product([32, 64], [4, 8, 16], [16, 32, 64, 128]):
            cb.ZMARCH = {"BX": bx, "BY": by, "ZCH": zch}
            _run("cupy-cuda-c/zmarch", u0, f"zmarch BX={bx} BY={by} ZCH={zch}", mt)
    elif w == "cuda-tb3d":
        for K in a.K or [2, 3, 4]:
            for tyt, r, zch in itertools.product([4, 8, 16], [1, 2, 4, 8], [64, 128, 256]):
                if tyt * r <= 2 * K + 4 or tyt * r > 64:
                    continue
                cb.TB3D[K] = {"TYT": tyt, "R": r, "ZCH": zch}
                _run(f"cupy-cuda-c/tb{K}", u0, f"tb3d K={K} TYT={tyt} R={r} ZCH={zch}", mt)
    elif w == "cuda-tb2d":
        for K in a.K or [1, 2, 4, 8]:
            key = "cupy-cuda-c/ymarch" if K == 1 else f"cupy-cuda-c/tb{K}"
            for r, warps, zch in itertools.product([2, 4, 8], [2, 4, 8], [64, 128, 256, 512]):
                cb.TB2D[K] = {"R": r, "WARPS": warps, "ZCH": zch}
                _run(key, u0, f"tb2d K={K} R={r} WARPS={warps} ZCH={zch}", mt)
    elif w.startswith("triton"):
        from pyheat3d.backends import triton_backend as tb

        sts = ["", "-fp16"]
        if w == "triton-2d":
            for (by, bx), nw, st in itertools.product(
                [(4, 128), (8, 128), (16, 128), (8, 256), (16, 256), (32, 128), (4, 512)],
                [4, 8], sts):
                tb.TritonStepper.BY2, tb.TritonStepper.BX2, tb.TritonStepper.WARPS2 = by, bx, nw
                _run("triton/tile" + st, u0, f"triton2d BY={by} BX={bx} warps={nw} {st}", mt)
        elif w == "triton-3d":
            for (by, bx), nw, st in itertools.product(
                [(4, 64), (8, 64), (4, 128), (8, 128), (16, 64), (2, 256)], [4, 8], sts):
                tb.TritonStepper.BY3, tb.TritonStepper.BX3, tb.TritonStepper.WARPS3 = by, bx, nw
                _run("triton/tile" + st, u0, f"triton3d BY={by} BX={bx} warps={nw} {st}", mt)
        else:
            for (by, bx), nw, zch, st in itertools.product(
                [(8, 64), (16, 64), (8, 128), (4, 128)], [4, 8], [4, 8, 16, 32], sts):
                T = tb.TritonStepper
                T.BYZ, T.BXZ, T.WARPSZ, T.ZCH = by, bx, nw, zch
                _run("triton/zmarch" + st, u0, f"zmarch BY={by} BX={bx} w={nw} ZCH={zch} {st}",
                     mt)
    else:
        raise SystemExit(f"unknown target {w}")


if __name__ == "__main__":
    main()
