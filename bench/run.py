"""The only benchmark entry point.

    python bench/run.py --quick   # < 2 min smoke run (small sizes) -> results/quick.json
    python bench/run.py --full    # publishable run on an idle GPU -> results/rtx3070-laptop.json
    python bench/run.py --dev     # < 1 GB GPU memory sweep -> results/dev-provisional.json

Optional filters: ``--variants cupy-cuda-c/tb triton`` (substring match), ``--dims 3``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyheat3d.benchmark import run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="small sizes, short repeats (default)")
    g.add_argument("--full", action="store_true", help="full size sweep, publishable timing")
    g.add_argument("--dev", action="store_true",
                   help="all sizes below 1 GB GPU memory (provisional numbers on a shared GPU)")
    ap.add_argument("--variants", nargs="*", default=None, help="substring filter on keys")
    ap.add_argument("--dims", type=int, nargs="*", default=[2, 3])
    ap.add_argument("--out", default=None, help="output file name inside results/")
    a = ap.parse_args()
    results = Path(__file__).resolve().parents[1] / "results"
    mode = "full" if a.full else ("dev" if a.dev else "quick")
    run(mode, results, a.variants, tuple(a.dims), a.out)


if __name__ == "__main__":
    main()
