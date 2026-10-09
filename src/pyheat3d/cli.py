"""Command-line interface: ``pyheat3d {list, run, validate, bench}``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import analytic as an
from .problem import heat_stencil, jacobi_stencil, stability_limit
from .registry import VARIANTS, availability, make_stepper


def _cmd_list(_: argparse.Namespace) -> int:
    print(f"{'variant':30s} {'dims':6s} {'storage':9s} {'k':>2s}  status")
    for key, v in VARIANTS.items():
        ok, why = availability(key)
        dims = ",".join(str(d) for d in v.dims)
        print(f"{key:30s} {dims:6s} {v.storage:9s} {v.steps_per_pass:>2d}  "
              f"{'ok' if ok else 'unavailable (' + why + ')'}")
    return 0


def _cmd_run(a: argparse.Namespace) -> int:
    dims = a.dims
    shape = (a.n,) * dims
    if a.mode == "heat":
        r = a.r if a.r is not None else 0.9 * stability_limit(dims)
        st = heat_stencil(dims, r, a.bc)
        u0 = (an.fourier_mode(a.n, dims, a.bc) if a.init == "mode"
              else np.random.default_rng(0).random(shape)).astype(np.float32)
    else:
        st = jacobi_stencil(dims, a.bc)
        u0 = an.harmonic_sinh(a.n, dims).astype(np.float32)
        u0[(slice(1, -1),) * dims] = 0.0
    s = make_stepper(a.variant, u0, st)
    s.step(s.steps_per_pass)
    s.synchronize()
    t0 = time.perf_counter()
    s.step(a.steps)
    s.synchronize()
    dt = time.perf_counter() - t0
    cells = s.ncells
    print(f"{a.variant}: {dims}D n={a.n} {a.mode} {a.bc}, {a.steps} steps in {dt:.3f} s -> "
          f"{cells * a.steps / dt / 1e9:.2f} GLUPS (development timing, not a benchmark)")
    u = s.get()
    print(f"state: min {u.min():.4g} max {u.max():.4g} mean {u.mean():.4g}")
    return 0


def _cmd_validate(a: argparse.Namespace) -> int:
    from .validation import run_all

    res = run_all(quick=a.quick)
    for k in ("spatial", "temporal", "laplace", "poisson"):
        for x in res[k]:
            print(f"{k:9s} {x['dims']}D {x.get('bc', 'dirichlet'):9s} observed order "
                  f"{x['order']:.3f}")
    for x in res["laplace_quadratic"]:
        print(f"quadratic {x['dims']}D Laplace (stencil-exact) error {x['err']:.2e}")
    cb = res["cross_backend"]
    bad = [x for x in cb if not x["pass"]]
    print(f"cross-backend: {len(cb)} comparisons vs NumPy, {len(bad)} failures")
    for x in bad:
        print("  FAIL", x)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        print("wrote", a.out)
    return 1 if bad else 0


def _cmd_bench(a: argparse.Namespace) -> int:
    from .benchmark import run

    run("full" if a.full else "quick", a.out_dir, a.variants, tuple(a.dims))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pyheat3d", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list variants and availability").set_defaults(fn=_cmd_list)

    r = sub.add_parser("run", help="run one variant once and print a rough rate")
    r.add_argument("--variant", default="numpy/slicing", choices=sorted(VARIANTS))
    r.add_argument("--dims", type=int, default=3, choices=(2, 3))
    r.add_argument("--n", type=int, default=64)
    r.add_argument("--steps", type=int, default=100)
    r.add_argument("--bc", default="dirichlet", choices=("dirichlet", "periodic"))
    r.add_argument("--mode", default="heat", choices=("heat", "jacobi"))
    r.add_argument("--r", type=float, default=None, help="mesh Fourier number alpha*dt/h^2")
    r.add_argument("--init", default="mode", choices=("mode", "random"))
    r.set_defaults(fn=_cmd_run)

    v = sub.add_parser("validate", help="convergence studies and cross-backend checks")
    v.add_argument("--quick", action="store_true", help="small grids (CPU friendly)")
    v.add_argument("--out", default=None, help="write JSON (e.g. results/validation.json)")
    v.set_defaults(fn=_cmd_validate)

    b = sub.add_parser("bench", help="benchmark (same as bench/run.py)")
    g = b.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    b.add_argument("--variants", nargs="*", default=None)
    b.add_argument("--dims", type=int, nargs="*", default=[2, 3])
    b.add_argument("--out-dir", default="results")
    b.set_defaults(fn=_cmd_bench)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
