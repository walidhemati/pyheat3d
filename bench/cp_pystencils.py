"""pystencils worker for ``bench/counterparts.py`` (runs in its own venv, not in pyheat3d's).

    ~/.venvs/cp-pystencils/bin/python -I bench/cp_pystencils.py '<json job>'

Job keys: ``dims``, ``n`` (array edge incl. the Dirichlet layer), ``c0``, ``c1``, ``variant``
(``default`` | ``tuned``) and either ``verify`` (``{"infile", "outfile", "steps"}``) or
``min_time`` + ``repeats``. Prints one JSON line on stdout.

The update is pyheat3d's ``u' = c0 u + c1 S`` on the interior; pystencils' default ghost layer
of 1 leaves the outermost layer untouched, which is pyheat3d's Dirichlet boundary.

* ``default``: runtime-sized fields, pystencils' default GPU indexing and block size.
* ``tuned``: fields with the shape fixed at code-generation time (constant strides) and the
  block size picked from a short list by timing (same budget for every candidate).
"""

from __future__ import annotations

import ctypes
import glob
import json
import math
import os
import site
import sys
import time
import warnings


def _preload_nvrtc() -> None:
    for root in site.getsitepackages():
        for path in glob.glob(os.path.join(root, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so.12")):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            return


_preload_nvrtc()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # several cupy-cuda* wheels are visible via system site
    import cupy as cp
import numpy as np  # noqa: E402
import pystencils as ps  # noqa: E402

BLOCKS = {2: [(16, 16, 1), (32, 4, 1), (64, 4, 1), (128, 2, 1), (256, 1, 1)],
          3: [(8, 8, 4), (32, 4, 2), (64, 4, 1), (128, 2, 1), (32, 8, 1)]}


def build(dims: int, n: int, c0: float, c1: float, fixed: bool, block=None):
    """Generate and compile the kernel; returns the callable wrapper."""
    spec = f"float32[{','.join([str(n)] * dims)}]" if fixed else f"float32[{dims}D]"
    src, dst = ps.fields(f"src, dst: {spec}", layout="c")

    def off(ax: int, s: int) -> tuple[int, ...]:
        return tuple(s if a == ax else 0 for a in range(dims))

    # x (last, contiguous axis) first, as in pyheat3d's canonical order
    s = src[off(dims - 1, -1)] + src[off(dims - 1, 1)]
    for ax in range(dims - 2, -1, -1):
        s = s + (src[off(ax, -1)] + src[off(ax, 1)])
    upd = ps.Assignment(dst.center, c0 * src.center + c1 * s)
    cfg = ps.CreateKernelConfig(target=ps.Target.CUDA)
    if block is not None:
        cfg.gpu.default_block_size = block
    return ps.create_kernel(upd, cfg).compile()


def _run(f, a, b, steps: int):
    for _ in range(steps):
        f(src=a, dst=b)
        a, b = b, a
    return a, b


def _time(f, a, b, steps: int) -> tuple[float, object, object]:
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    a, b = _run(f, a, b, steps)
    cp.cuda.Device().synchronize()
    return time.perf_counter() - t0, a, b


def main() -> None:
    job = json.loads(sys.argv[1])
    dims, n, c0, c1 = job["dims"], job["n"], job["c0"], job["c1"]
    fixed = job["variant"] == "tuned"
    t0 = time.perf_counter()
    out: dict = {"pystencils": ps.__version__, "cupy": cp.__version__}
    if "verify" in job:
        v = job["verify"]
        f = build(dims, n, c0, c1, fixed)
        u0 = np.load(v["infile"])
        a = cp.asarray(u0)
        b = a.copy()  # both buffers carry the boundary values
        a, b = _run(f, a, b, v["steps"])
        np.save(v["outfile"], cp.asnumpy(a))
        print(json.dumps(out))
        return
    shape = (n,) * dims
    a = cp.asarray(np.random.default_rng(12345).random(shape, dtype=np.float32))
    b = a.copy()
    block = None
    if fixed:  # pick the block size: same short budget for every candidate
        best = math.inf
        for cand in BLOCKS[dims]:
            g = build(dims, n, c0, c1, True, cand)
            a, b = _run(g, a, b, 2)
            dt, a, b = _time(g, a, b, 10)
            if dt < best:
                best, block, f = dt, cand, g
        out["block"] = list(block)
    else:
        f = build(dims, n, c0, c1, False)
    a, b = _run(f, a, b, 4)  # warm-up
    out["warmup_s"] = time.perf_counter() - t0
    steps = 2
    while True:
        dt, a, b = _time(f, a, b, steps)
        if dt >= 0.2 * job["min_time"] or steps >= 1 << 20:
            break
        steps *= 2
    steps = max(2, int(math.ceil(job["min_time"] / (dt / steps))))
    secs = []
    for _ in range(job["repeats"]):
        dt, a, b = _time(f, a, b, steps)
        secs.append(dt)
    # (CuPy reductions need CCCL headers that NVRTC 12 cannot compile here: check on host)
    out.update(steps=steps, seconds=secs, finite=bool(np.isfinite(cp.asnumpy(a)).all()))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
