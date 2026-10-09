"""Measure the hardware ceilings of the current machine and write results/roofline.json.

    python bench/roofline.py            # 1 GiB copy (needs ~2 GiB GPU memory)
    python bench/roofline.py --small    # 128 MiB copy, for development on a shared GPU
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyheat3d import traffic
from pyheat3d.benchmark import gpu_state, machine_info
from pyheat3d.roofline import copy_bandwidth, fma_tflops


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--small", action="store_true", help="128 MiB copy instead of 1 GiB")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    a = ap.parse_args()
    nbytes = 2**27 if a.small else 2**30
    bw = copy_bandwidth(nbytes)
    tf = fma_tflops()
    res = {
        "machine": machine_info(),
        "provisional": bool(a.small),
        "copy_bytes": nbytes,
        "measured_copy_bandwidth_GBs": bw,
        "reference_copy_bandwidth_GBs": traffic.COPY_BW_GBS,
        "spec_bandwidth_GBs": traffic.SPEC_BW_GBS,
        "fp32_fma_tflops": tf,
        "gpu_state": gpu_state(),
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "roofline.json").write_text(json.dumps(res, indent=1))
    print(f"copy bandwidth {bw:.1f} GB/s ({100 * bw / traffic.SPEC_BW_GBS:.1f}% of spec), "
          f"FP32 FMA {tf:.2f} TFLOP/s -> {out / 'roofline.json'}")


if __name__ == "__main__":
    main()
