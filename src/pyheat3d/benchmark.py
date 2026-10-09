"""Benchmark driver behind ``bench/run.py`` and ``pyheat3d bench``.

Timing rules: JIT/compile warm-up before timing, device synchronisation around every timed
region, each timed repeat runs >= ``min_time`` seconds, median of ``repeats`` repeats reported
together with min/max. Throughput is in lattice-site updates per second (MLUPS); bandwidth is
derived with the minimal traffic model of :mod:`pyheat3d.traffic`.
"""

from __future__ import annotations

import datetime as _dt
import gc
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import traffic
from .problem import heat_stencil
from .registry import VARIANTS, availability, cuda_available, make_stepper

PLANS: dict[str, dict[str, Any]] = {
    "quick": {
        "sizes": {2: [1024, 4096], 3: [64, 256]},
        "cpu_max": {2: 1024, 3: 64},
        "min_time": 0.05,
        "repeats": 5,
        "mem_limit_GB": 0.9,
        "provisional": True,
        "out": "quick.json",
    },
    "dev": {  # provisional numbers on a shared GPU: every size that fits in < 1 GB
        "sizes": {2: [512, 1024, 2048, 4096, 8192], 3: [32, 64, 128, 256, 320]},
        "cpu_max": {2: 2048, 3: 128},
        "min_time": 0.1,
        "repeats": 5,
        "mem_limit_GB": 0.9,
        "provisional": True,
        "out": "dev-provisional.json",
    },
    "full": {
        "sizes": {2: [1024, 2048, 4096, 8192, 16384], 3: [64, 128, 256, 384, 512]},
        "cpu_max": {2: 4096, 3: 256},
        "min_time": 0.5,
        "repeats": 5,
        "mem_limit_GB": 6.5,
        "provisional": False,
        "out": "rtx3070-laptop.json",
    },
}


def _versions() -> dict[str, str]:
    out: dict[str, str] = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ("torch", "triton", "warp", "cupy", "numba"):
        try:
            if mod == "cupy":
                from .backends import cupy_backend  # noqa: F401  (preloads NVRTC)
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = "unavailable"
    try:
        import numba_cuda  # type: ignore

        out["numba_cuda"] = getattr(numba_cuda, "__version__", "?")
    except Exception:
        pass
    return out


def _nvml() -> Any:
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def gpu_state() -> dict[str, Any]:
    """Current SM/memory clock, temperature and power (for the throttling audit)."""
    nv = _nvml()
    if nv is None:
        return {}
    try:
        h = nv.nvmlDeviceGetHandleByIndex(0)
        return {
            "sm_clock_MHz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
            "mem_clock_MHz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
            "temp_C": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
            "power_W": round(nv.nvmlDeviceGetPowerUsage(h) / 1000.0, 1),
        }
    except Exception:
        return {}


def machine_info() -> dict[str, Any]:
    info: dict[str, Any] = {"gpu": None, "driver": None, "cpu": platform.processor() or None}
    nv = _nvml()
    if nv is not None:
        try:
            h = nv.nvmlDeviceGetHandleByIndex(0)
            name = nv.nvmlDeviceGetName(h)
            info["gpu"] = name.decode() if isinstance(name, bytes) else name
            drv = nv.nvmlSystemGetDriverVersion()
            info["driver"] = drv.decode() if isinstance(drv, bytes) else drv
        except Exception:
            pass
    info.update(_versions())
    info["date"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    return info


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        from .backends import cupy_backend

        cupy_backend.cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _mem_estimate_GB(key: str, shape: tuple[int, ...]) -> float:
    var = VARIANTS[key]
    cells = math.prod(shape)
    bufs = 4 if var.backend.startswith("torch") else 2
    return bufs * cells * var.bytes_per_value / 1e9


def time_variant(key: str, u0: np.ndarray, min_time: float, repeats: int,
                 r: float | None = None) -> dict[str, Any]:
    """Time one variant on one initial state (Dirichlet heat mode). Returns a result record."""
    var = VARIANTS[key]
    dims = u0.ndim
    stencil = heat_stencil(dims, r if r is not None else 0.9 / (2 * dims))
    st = make_stepper(key, u0, stencil)
    k = var.steps_per_pass
    try:
        t0 = time.perf_counter()
        st.step(2 * k)  # JIT / autotune / compile warm-up
        st.synchronize()
        compile_s = time.perf_counter() - t0
        # calibrate the number of steps per timed repeat
        n = k
        while True:
            t0 = time.perf_counter()
            st.step(n)
            st.synchronize()
            dt = time.perf_counter() - t0
            if dt >= 0.2 * min_time or n >= 1 << 22:
                break
            n *= 2
        per_step = dt / n
        n = max(k, int(math.ceil(min_time / per_step / k)) * k)
        times = []
        for _ in range(repeats):
            st.synchronize()
            t0 = time.perf_counter()
            st.step(n)
            st.synchronize()
            times.append(time.perf_counter() - t0)
        final = st.get()
        finite = bool(np.isfinite(final).all())
        st_tile = dict(getattr(st, "tile", {}))
    finally:
        st.close()
        del st
    ncells = math.prod(s - 2 for s in u0.shape)
    mlups = [ncells * n / t / 1e6 for t in times]
    med = float(np.median(mlups))
    bpu = traffic.minimal_bytes(var.bytes_per_value, k)
    flops = traffic.flops_per_update(dims)
    eff = med * bpu / 1e3
    rec: dict[str, Any] = {
        "backend": var.backend,
        "variant": var.name,
        "key": key,
        "rung": var.rung,
        "dims": dims,
        "size": list(u0.shape),
        "bc": "dirichlet",
        "dtype": var.storage if var.storage != "native" else str(u0.dtype),
        "metric": "MLUPS",
        "value": med,
        "repeats": repeats,
        "median": med,
        "min": float(min(mlups)),
        "max": float(max(mlups)),
        "steps_per_repeat": n,
        "seconds": [float(t) for t in times],
        "steps_per_pass": k,
        "bytes_per_unit": bpu,
        "effective_GBs": eff,
        "pct_copy_bw": 100.0 * eff / traffic.COPY_BW_GBS,
        "pct_spec_bw": 100.0 * eff / traffic.SPEC_BW_GBS,
        "roofline_MLUPS": traffic.roofline_mlups(traffic.COPY_BW_GBS, bpu),
        "flops_per_unit": flops,
        "GFLOPs": med * flops / 1e3,
        "warmup_s": compile_s,
        "finite": finite,
        "gpu_state": gpu_state(),
        "notes": var.description,
    }
    if k > 1:
        # throughput expressed as the bandwidth a single-step kernel would need (NOT a DRAM rate)
        rec["single_step_equiv_GBs"] = med * traffic.minimal_bytes(var.bytes_per_value, 1) / 1e3
        if var.backend == "cupy-cuda-c":
            cfg = st_tile
            if dims == 3:
                tx, ty, zch = 32, cfg["TYT"] * cfg["R"], cfg["ZCH"]
            else:
                tx, ty, zch = 32 * cfg["R"], 1, cfg["ZCH"]
            hb = traffic.tb_model_bytes(dims, k, var.bytes_per_value, tx, ty, zch)
            rec["halo_model_bytes_per_unit"] = hb
            rec["halo_model_GBs"] = med * hb / 1e3
            rec["tile"] = cfg
        rec["notes"] += (f"; temporal blocking: per-update traffic is 1/{k} of single-step, "
                         "so its roofline is k times higher")
    return rec


def run(mode: str = "quick", out_dir: str | Path = "results", variants: list[str] | None = None,
        dims_list: tuple[int, ...] = (2, 3), out_name: str | None = None,
        verbose: bool = True) -> dict[str, Any]:
    """Run a benchmark plan and write the results JSON. Returns the result dictionary."""
    plan = PLANS[mode]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (out_name or plan["out"])
    keys = [k for k in VARIANTS if variants is None or any(v in k for v in variants)]
    ok_keys = []
    for k in keys:
        ok, why = availability(k)
        if ok:
            ok_keys.append(k)
        elif verbose:
            print(f"skip {k}: {why}")
    measured_bw = None
    if cuda_available():
        try:
            from .roofline import copy_bandwidth

            measured_bw = copy_bandwidth(2**30 if mode == "full" else 2**27)
        except Exception as e:  # pragma: no cover
            print("copy-bandwidth measurement failed:", e)
    result: dict[str, Any] = {
        "machine": machine_info(),
        "provisional": plan["provisional"],
        "mode": mode,
        "ceiling": {
            "copy_bandwidth_GBs": traffic.COPY_BW_GBS,
            "spec_bandwidth_GBs": traffic.SPEC_BW_GBS,
            "measured_copy_bandwidth_GBs": measured_bw,
            "fp32_tflops": None,
        },
        "traffic_model": "minimal: 2*bytes_per_value/steps_per_pass per lattice-site update",
        "gpu_state_start": gpu_state(),
        "runs": [],
    }
    t_start = time.perf_counter()
    rng = np.random.default_rng(12345)
    for dims in dims_list:
        for n in plan["sizes"][dims]:
            shape = (n,) * dims
            u0 = rng.random(shape, dtype=np.float32)
            for key in ok_keys:
                var = VARIANTS[key]
                if dims not in var.dims:
                    continue
                if not var.gpu and n > plan["cpu_max"][dims]:
                    continue
                if _mem_estimate_GB(key, shape) > plan["mem_limit_GB"]:
                    continue
                try:
                    rec = time_variant(key, u0, plan["min_time"], plan["repeats"])
                except Exception as e:
                    print(f"FAILED {key} {shape}: {type(e).__name__}: {e}")
                    _free_gpu()
                    continue
                result["runs"].append(rec)
                if verbose:
                    extra = ""
                    if "single_step_equiv_GBs" in rec:
                        extra = f"  (single-step equiv {rec['single_step_equiv_GBs']:.0f} GB/s)"
                    print(f"{dims}D {n:>6} {key:28s} {rec['median'] / 1e3:8.2f} GLUPS "
                          f"{rec['effective_GBs']:7.1f} GB/s {rec['pct_copy_bw']:5.1f}% copy"
                          + extra, flush=True)
                _free_gpu()
            del u0
            out_path.write_text(json.dumps(result, indent=1))
    result["gpu_state_end"] = gpu_state()
    result["wall_time_s"] = time.perf_counter() - t_start
    out_path.write_text(json.dumps(result, indent=1))
    if verbose:
        print(f"wrote {out_path} ({len(result['runs'])} runs, {result['wall_time_s']:.0f} s)")
    return result
