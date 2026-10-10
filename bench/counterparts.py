"""Same-GPU head-to-head against established stencil codes -> results/counterparts.json.

    python bench/counterparts.py --quick   # < 60 s, < 1 GB GPU: smoke run, numbers provisional
    python bench/counterparts.py --full    # publishable: idle GPU, 16384^2 and 512^3

Counterparts (sources pinned under ``~/python-hpc/_ref``, override with ``PYHPC_REF``):

* **pystencils** 2.1 (FAU; the code generator behind waLBerla), CUDA target through CuPy, run
  by ``bench/cp_pystencils.py`` in its own venv (``CP_PYSTENCILS_PY``, default
  ``~/.venvs/cp-pystencils/bin/python``). Variants ``default`` and ``tuned`` (fixed shape and a
  timed choice among five block sizes).
* **AN5D** (Matsumura et al., CGO 2020): the CUDA code that AN5D generated for the artifact
  (``AN5D-Artifact/compiled/float``, stencils ``star2d1r`` = 5-point and ``star3d1r`` =
  7-point), built with nvcc for the local GPU. The artifact's own coefficients are replaced
  by pyheat3d's (``-DAN5D_C0/-DAN5D_C1``), and ``bench/an5d_common.h`` replaces the
  artifact's driver; the generated host and kernel code is used unchanged otherwise.
  ``--quick`` builds the V100 top-1 configuration per stencil, ``--full`` every shipped one.

Setup (once; binaries are cached in ``_ref/an5d-build``)::

    cd ~/python-hpc/_ref
    git clone https://github.com/khaki3/AN5D-Artifact an5d-artifact  # pinned: fbebf5f
    git clone https://github.com/khaki3/AN5D an5d                    # pinned: 21e1f52 (reference)
    # pystencils 2.1 sdist kept in pystencils-2.1-sdist/ (sha256 below)
    uv venv ~/.venvs/cp-pystencils --python ~/miniconda3/bin/python --system-site-packages
    uv pip install --python ~/.venvs/cp-pystencils/bin/python pystencils==2.1 \\
        cupy-cuda12x==14.2.0 nvidia-cuda-nvrtc-cu12

The same run also times pyheat3d's own fp32 variants at the same sizes, so each comparison is
from one session. Every counterpart is first checked against pyheat3d's NumPy oracle on a small
grid (Dirichlet heat update, ``r = 0.9 / (2 d)``, 23 steps).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pyheat3d import traffic  # noqa: E402
from pyheat3d.backends.numpy_ref import reference_step  # noqa: E402
from pyheat3d.benchmark import _free_gpu, gpu_state, machine_info, time_variant  # noqa: E402
from pyheat3d.problem import heat_stencil  # noqa: E402

REF = Path(os.environ.get("PYHPC_REF", REPO.parent / "_ref"))
AN5D_ART = REF / "an5d-artifact"
AN5D_SRC = AN5D_ART / "compiled" / "float"
AN5D_BUILD = REF / "an5d-build"
PS_PY = Path(os.environ.get("CP_PYSTENCILS_PY", "~/.venvs/cp-pystencils/bin/python")).expanduser()
PS_SDIST = REF / "pystencils-2.1-sdist" / "pystencils-2.1.tar.gz"
PS_SDIST_SHA256 = "d6dd4c1a640435dc27712f7f8db488a25cf644908b321d17161d044161e2b831"
NVCC = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
# the artifact's flags (nvcc_compile_top5.sh) minus its P100/V100 targets and OpenMP host code
NVCC_FLAGS = ["-O3", "--use_fast_math", "-diag-suppress", "177,186,550"]
STENCIL = {2: "star2d1r", 3: "star3d1r"}
CENTER = "0.2500f"  # the artifact's centre coefficient in both star stencils
NEIGH = {2: r"0\.187[3467]f", 3: r"0\.12(4[89]|5[0-3])f"}
VERIFY = {2: 200, 3: 70}
VERIFY_STEPS = 23

PLANS: dict[str, dict[str, Any]] = {
    "quick": {"size": {2: 4096, 3: 256}, "min_time": 0.1, "repeats": 5, "an5d": "top1",
              "pyheat3d": {2: ["triton/tile", "cupy-cuda-c/naive", "cupy-cuda-c/tb4",
                               "cupy-cuda-c/tb8"],
                           3: ["triton/zmarch", "cupy-cuda-c/zmarch", "cupy-cuda-c/tb3"]},
              "provisional": True},
    "full": {"size": {2: 16384, 3: 512}, "min_time": 0.5, "repeats": 5, "an5d": "all",
             "pyheat3d": {2: ["torch-compile/inductor", "triton/tile", "warp/naive",
                              "numba/naive", "cupy-cuda-c/naive", "cupy-cuda-c/ymarch",
                              "cupy-cuda-c/tb2", "cupy-cuda-c/tb3", "cupy-cuda-c/tb4",
                              "cupy-cuda-c/tb8"],
                          3: ["torch-compile/inductor", "triton/zmarch", "warp/zmarch",
                              "numba/zmarch", "cupy-cuda-c/zmarch", "cupy-cuda-c/tb2",
                              "cupy-cuda-c/tb3", "cupy-cuda-c/tb4"]},
             "provisional": False},
}


def coeffs(dims: int) -> tuple[float, float]:
    """pyheat3d's benchmark update: FTCS heat with r = 0.9 / (2 d)."""
    st = heat_stencil(dims, 0.9 / (2 * dims))
    return st.c0, st.c1


def _git(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def _sm_arch() -> str:
    out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.split()
    return "sm_" + (out[0].replace(".", "") if out else "86")


def _nvcc_version() -> str:
    out = subprocess.run([NVCC, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"release ([\d.]+), V([\d.]+)", out)
    return m.group(2) if m else "?"


# ------------------------------------------------------------------------------------- AN5D
def an5d_configs(dims: int, which: str) -> list[str]:
    """AN5D configurations shipped in the artifact: ``top1`` = V100 rank 1, or ``all``."""
    st = STENCIL[dims]
    if which == "top1":
        for line in (AN5D_ART / "compiled" / "v100_float.csv").read_text().splitlines():
            f = line.split(",")
            if f[0] == st:
                bs = f[1] if dims == 2 else f"{f[1]}x{f[2]}"
                return [f"{st}-{bs}-{f[3]}-{f[4]}"]
        raise RuntimeError(f"{st} not in v100_float.csv")
    names = sorted({p.name.split("_")[0] for p in AN5D_SRC.glob(f"{st}-*_host.cu")})
    return sorted(names, key=lambda s: [int(x) for x in re.findall(r"\d+", s.split("-", 1)[1])])


def an5d_bt(cfg: str) -> int:
    """Temporal blocking degree bT of a configuration name ``<stencil>-<bs>-<bT>-<sl>``."""
    return int(cfg.split("-")[2])


def _f32_literal(x: float) -> str:
    return f"{float(np.float32(x)):.9g}f"


def an5d_build(cfg: str, dims: int, arch: str) -> Path:
    """Patch coefficients into a copy of the generated code and compile it (cached)."""
    c0, c1 = coeffs(dims)
    tag = f"{cfg}_{arch}_c0={_f32_literal(c0)}_c1={_f32_literal(c1)}"
    out = AN5D_BUILD / tag
    exe = out / "an5d"
    harness = REPO / "bench" / "an5d_common.h"
    if exe.exists() and exe.stat().st_mtime > harness.stat().st_mtime:
        return exe
    out.mkdir(parents=True, exist_ok=True)
    kern = (AN5D_SRC / f"{cfg}_kernel.cu").read_text()
    n_center = kern.count(CENTER)
    kern, n_neigh = re.subn(NEIGH[dims], "AN5D_C1", kern.replace(CENTER, "AN5D_C0"))
    if n_center == 0 or n_neigh != 2 * dims * n_center or re.search(r"\b0\.\d+f", kern):
        raise RuntimeError(f"{cfg}: unexpected coefficient pattern ({n_center}, {n_neigh})")
    (out / f"{cfg}_kernel.cu").write_text(kern)
    for suffix in ("_host.cu", "_kernel.hu"):
        shutil.copy(AN5D_SRC / f"{cfg}{suffix}", out / f"{cfg}{suffix}")
    shutil.copy(harness, out / "common.h")
    cmd = [NVCC, *NVCC_FLAGS, f"-gencode=arch=compute_{arch[3:]},code={arch}",
           f"-DAN5D_C0={_f32_literal(c0)}", f"-DAN5D_C1={_f32_literal(c1)}", "-I", str(out),
           str(out / f"{cfg}_host.cu"), str(out / f"{cfg}_kernel.cu"), "-o", str(exe)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    (out / "build.log").write_text(" ".join(cmd) + "\n" + res.stdout + res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"nvcc failed for {cfg}, see {out / 'build.log'}")
    return exe


def _parse(stdout: str) -> dict[str, Any]:
    d: dict[str, Any] = {"run_ms": []}
    for line in stdout.splitlines():
        k, _, v = line.partition(" ")
        if k == "run_ms":
            d["run_ms"].append(float(v))
        elif k in ("load_ms", "warmup_ms"):
            d[k] = float(v)
        elif k == "steps":
            d["steps"] = int(v)
    return d


def an5d_verify(exe: Path, dims: int, tmp: Path) -> float:
    n = VERIFY[dims]
    u0 = np.random.default_rng(7).random((n,) * dims, dtype=np.float32)
    fin, fout = tmp / "in.f32", tmp / "out.f32"
    u0.tofile(fin)
    subprocess.run([str(exe), "-s", str(n - 2), "-t", str(VERIFY_STEPS), "-n", "1",
                    "-i", str(fin), "-o", str(fout)], check=True, capture_output=True)
    got = np.fromfile(fout, np.float32).reshape(u0.shape)
    return float(np.abs(got - _oracle(u0, dims)).max())


def an5d_time(exe: Path, dims: int, n: int, plan: dict) -> dict[str, Any]:
    t0 = time.perf_counter()
    res = subprocess.run([str(exe), "-s", str(n - 2), "-t", "1", "-n", str(plan["repeats"]),
                          "-w", "50", "-m", str(1e3 * plan["min_time"])],
                         check=True, capture_output=True, text=True)
    d = _parse(res.stdout)
    d["wall_s"] = time.perf_counter() - t0
    return d


# ------------------------------------------------------------------------------- pystencils
def ps_job(job: dict[str, Any]) -> dict[str, Any]:
    res = subprocess.run([str(PS_PY), "-I", str(REPO / "bench" / "cp_pystencils.py"),
                          json.dumps(job)], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip().splitlines()[-1] if res.stderr else "failed")
    return json.loads(res.stdout.strip().splitlines()[-1])


def ps_verify(variant: str, dims: int, tmp: Path) -> float:
    n = VERIFY[dims]
    u0 = np.random.default_rng(7).random((n,) * dims, dtype=np.float32)
    fin, fout = tmp / "ps_in.npy", tmp / "ps_out.npy"
    np.save(fin, u0)
    c0, c1 = coeffs(dims)
    ps_job({"dims": dims, "n": n, "c0": c0, "c1": c1, "variant": variant,
            "verify": {"infile": str(fin), "outfile": str(fout), "steps": VERIFY_STEPS}})
    return float(np.abs(np.load(fout) - _oracle(u0, dims)).max())


# ---------------------------------------------------------------------------------- helpers
def _oracle(u0: np.ndarray, dims: int) -> np.ndarray:
    c0, c1 = coeffs(dims)
    u = u0.copy()
    for _ in range(VERIFY_STEPS):
        u = reference_step(u, c0, c1, "dirichlet")
    return u


def record(code: str, variant: str, dims: int, n: int, k: int, seconds: list[float],
           steps: int, **extra: Any) -> dict[str, Any]:
    """One result row; rates in GLUPS, bandwidth from the minimal traffic model."""
    cells = (n - 2) ** dims
    glups = [cells * steps / s / 1e9 for s in seconds]
    med = float(np.median(glups))
    bpu = traffic.minimal_bytes(4, k)
    eff = med * bpu
    rec = {"code": code, "variant": variant, "dims": dims, "size": [n] * dims,
           "dtype": "float32", "bc": "dirichlet", "steps_fused": k, "metric": "GLUPS",
           "median": med, "min": float(min(glups)), "max": float(max(glups)),
           "repeats": len(seconds), "steps_per_repeat": steps, "seconds": seconds,
           "bytes_per_unit": bpu, "effective_GBs": eff,
           "pct_copy_bw": 100.0 * eff / traffic.COPY_BW_GBS,
           "pct_spec_bw": 100.0 * eff / traffic.SPEC_BW_GBS,
           "single_step_equiv_GBs": med * traffic.minimal_bytes(4, 1),
           "x_single_step_roofline": med / (traffic.COPY_BW_GBS / traffic.minimal_bytes(4, 1)),
           "gpu_state": gpu_state()}
    rec.update(extra)
    return rec


def _print(rec: dict[str, Any]) -> None:
    err = rec.get("verify_max_abs_err")
    print(f"{rec['dims']}D {rec['size'][0]:>6} {rec['code']:10s} {rec['variant']:24s} "
          f"k={rec['steps_fused']:<2d} {rec['median']:7.1f} GLUPS "
          f"({rec['min']:.1f}-{rec['max']:.1f}) {rec['pct_copy_bw']:5.1f}% copy "
          f"x{rec['x_single_step_roofline']:.2f} roof"
          + (f"  verify {err:.1e}" if err is not None else ""), flush=True)


def run(mode: str, out: Path) -> dict[str, Any]:
    plan = PLANS[mode]
    arch = _sm_arch()
    t_start = time.perf_counter()
    result: dict[str, Any] = {
        "machine": machine_info() | {"nvcc": _nvcc_version(), "sm": arch},
        "provisional": plan["provisional"], "mode": mode,
        "ceiling": {"copy_bandwidth_GBs": traffic.COPY_BW_GBS,
                    "spec_bandwidth_GBs": traffic.SPEC_BW_GBS},
        "traffic_model": "minimal: 8 B / steps_fused per fp32 lattice-site update",
        "update": {str(d): {"c0": coeffs(d)[0], "c1": coeffs(d)[1], "r": 0.9 / (2 * d)}
                   for d in (2, 3)},
        "verification": {"sizes": VERIFY, "steps": VERIFY_STEPS, "oracle":
                         "pyheat3d.backends.numpy_ref.reference_step (float32)"},
        "counterparts": {
            "an5d": {"artifact": "https://github.com/khaki3/AN5D-Artifact",
                     "artifact_commit": _git(AN5D_ART),
                     "generator": "https://github.com/khaki3/AN5D",
                     "generator_commit": _git(REF / "an5d"),
                     "nvcc_flags": NVCC_FLAGS + [f"-gencode=arch=compute_{arch[3:]},code={arch}"],
                     "notes": "pre-generated CUDA from the artifact (generator not rebuilt: it "
                              "needs LLVM/Clang 3.4 and GCC 4.8); coefficients replaced by "
                              "pyheat3d's; timing = AN5D's own (device-synchronised host "
                              "clock around the kernel launches, copies excluded)"},
            "pystencils": {"source": "https://i10git.cs.fau.de/pycodegen/pystencils",
                           "version": None, "sdist": str(PS_SDIST),
                           "sdist_sha256": PS_SDIST_SHA256,
                           "python": str(PS_PY)},
        },
        "gpu_state_start": gpu_state(),
        "runs": [],
    }
    tmp = Path(tempfile.mkdtemp(prefix="pyheat3d-cp-"))
    try:
        # build all AN5D binaries first (<= 4 parallel nvcc jobs)
        cfgs = {d: an5d_configs(d, plan["an5d"]) for d in (2, 3)}
        with cf.ThreadPoolExecutor(max_workers=4) as ex:
            futs = {(d, c): ex.submit(an5d_build, c, d, arch) for d in (2, 3) for c in cfgs[d]}
        exes: dict[tuple[int, str], Path] = {}
        for key, fut in futs.items():
            try:
                exes[key] = fut.result()
            except Exception as e:
                print(f"AN5D build failed {key}: {e}")
        print(f"built {len(exes)} AN5D binaries ({time.perf_counter() - t_start:.0f} s)")
        for dims in (2, 3):
            n = plan["size"][dims]
            c0, c1 = coeffs(dims)
            # pyheat3d's own variants, same session, same size
            rng = np.random.default_rng(12345)
            u0 = rng.random((n,) * dims, dtype=np.float32)
            for key in plan["pyheat3d"][dims]:
                try:
                    r = time_variant(key, u0, plan["min_time"], plan["repeats"])
                except Exception as e:
                    print(f"pyheat3d {key} failed: {type(e).__name__}: {e}")
                    _free_gpu()
                    continue
                rec = record("pyheat3d", key, dims, n, r["steps_per_pass"], r["seconds"],
                             r["steps_per_repeat"], backend=r["backend"], notes=r["notes"])
                result["runs"].append(rec)
                _print(rec)
                _free_gpu()
            del u0
            # pystencils
            for variant in ("default", "tuned"):
                try:
                    err = ps_verify(variant, dims, tmp)
                    d = ps_job({"dims": dims, "n": n, "c0": c0, "c1": c1, "variant": variant,
                                "min_time": plan["min_time"], "repeats": plan["repeats"]})
                except Exception as e:
                    print(f"pystencils {variant} {dims}D failed: {e}")
                    continue
                result["counterparts"]["pystencils"]["version"] = d["pystencils"]
                note = ("runtime-sized fields, pystencils default indexing/block size"
                        if variant == "default" else
                        f"shape fixed at codegen, block {tuple(d['block'])} picked by timing")
                rec = record("pystencils", variant, dims, n, 1, d["seconds"], d["steps"],
                             backend="pystencils", verify_max_abs_err=err,
                             block=d.get("block"), warmup_s=d["warmup_s"],
                             finite=d["finite"], notes=note + f"; cupy {d['cupy']}")
                result["runs"].append(rec)
                _print(rec)
            # AN5D
            for cfg in cfgs[dims]:
                exe = exes.get((dims, cfg))
                if exe is None:
                    continue
                try:
                    err = an5d_verify(exe, dims, tmp)
                    d = an5d_time(exe, dims, n, plan)
                except Exception as e:
                    print(f"AN5D {cfg} failed: {e}")
                    continue
                bt = an5d_bt(cfg)
                rec = record("AN5D", cfg, dims, n, bt, [ms / 1e3 for ms in d["run_ms"]],
                             d["steps"], backend="an5d", verify_max_abs_err=err,
                             notes=f"AN5D-generated, bT={bt} fused steps per kernel launch "
                                   "(remainder steps with smaller bT)")
                result["runs"].append(rec)
                _print(rec)
            out.write_text(json.dumps(result, indent=1))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    result["gpu_state_end"] = gpu_state()
    result["wall_time_s"] = time.perf_counter() - t_start
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out} ({len(result['runs'])} runs, {result['wall_time_s']:.0f} s)")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true", help="small sizes, top-1 AN5D configs")
    g.add_argument("--full", action="store_true", help="16384^2 / 512^3, all AN5D configs")
    ap.add_argument("--out", default="counterparts.json", help="file name inside results/")
    a = ap.parse_args()
    run("full" if a.full else "quick", REPO / "results" / a.out)


if __name__ == "__main__":
    main()
