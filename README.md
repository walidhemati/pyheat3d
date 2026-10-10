# pyheat3d

**The explicit heat equation and Jacobi iteration (5-point 2D, 7-point 3D stencil) at the GPU
memory roofline, from Python: seven backends, one interface, validated against analytic
solutions.**

[![CI](https://github.com/walidhemati/pyheat3d/actions/workflows/ci.yml/badge.svg)](https://github.com/walidhemati/pyheat3d/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Paper](https://img.shields.io/badge/paper-PDF-red.svg)](paper/paper.pdf)

![A periodic 96^3 temperature field relaxing under the heat equation](docs/media/hero.gif)

## Headline result

<!-- headline:start -->
> **3D 512^3, single step:** `numba/zmarch` **41.0 GLUPS** = 328 GB/s = **91% of the 361 GB/s copy bandwidth** (73% of the 448 GB/s spec).<br>
> **3D temporal blocking (k=3):** `cupy-cuda-c/tb3` **94.1 GLUPS**, 2.09x the single-step roofline of 45.1 GLUPS while moving an estimated 251 GB/s (minimal model) to 333 GB/s (with halo over-fetch) of DRAM traffic. The roofline per step drops k-fold; the hardware is not beaten.<br>
> **2D 16384^2, single step:** `torch-compile/inductor` **45.5 GLUPS** = 364 GB/s = **101% of the 361 GB/s copy bandwidth** (81% of the 448 GB/s spec).<br>
> **2D temporal blocking (k=8):** `cupy-cuda-c/tb8` **157.8 GLUPS**, 3.50x the single-step roofline of 45.1 GLUPS while moving an estimated 158 GB/s (minimal model) to 191 GB/s (with halo over-fetch) of DRAM traffic. The roofline per step drops k-fold; the hardware is not beaten.<br>
> NVIDIA GeForce RTX 3070 Laptop GPU, 2026-10-10.
<!-- headline:end -->

The stencil is the canonical memory-bound kernel: about 1 FLOP per byte of DRAM traffic.
Throughput is reported in lattice-site updates per second (GLUPS). Bandwidth comes from the
*minimal* traffic model: one read and one write per site per DRAM pass, so 8 B per fp32
update, 4 B for fp16/bf16 storage, and 8/k B when k time steps are fused per pass. The model
cannot inflate an inefficient kernel. Percentages are relative to the measured 361 GB/s
device-to-device copy bandwidth (and the 448 GB/s spec) of an RTX 3070 Laptop GPU.

**About temporal blocking.** The `tb<k>` variants fuse k time steps into one sweep over
memory. Each update then needs k times less DRAM traffic, so the roofline *per update* is k
times higher. Their lattice-update rate can exceed the single-step roofline (45 GLUPS for fp32)
while the actual DRAM traffic stays below the copy bandwidth. For every such run we report the
update rate, the estimated DRAM traffic (minimal model and a halo-over-fetch model), and the
"single-step-equivalent" bandwidth. The last one is not a DRAM rate.

## What's inside

| backend | variants | dims | notes |
|---|---|---|---|
| `numpy` | `slicing` | 2D, 3D | CPU reference and correctness oracle, any float dtype, Poisson source |
| `torch-eager` | `slicing` | 2D, 3D | NumPy-style slicing on the GPU (fp64 capable) |
| `torch-compile` | `inductor` | 2D, 3D | the same expression fused by `torch.compile` |
| `triton` | `tile`, `zmarch` (+ `-fp16`, `-bf16`) | 2D, 3D / 3D | tiled kernels; 2.5D z-marching with a register queue |
| `warp` | `naive`, `zmarch` | 2D, 3D / 3D | NVIDIA Warp kernels with pre-recorded launches |
| `numba` | `naive`, `zmarch` | 2D, 3D / 3D | Numba-CUDA kernels |
| `cupy-cuda-c` | `naive` (+ fp16/bf16), `zmarch`, `ymarch`, `tb2`, `tb3`, `tb4`, `tb8`, `tb4-fp16` | 2D, 3D | hand-written CUDA C compiled at run time by NVRTC: 2.5D blocking, temporal blocking |

Physics and numerics: explicit FTCS for `u_t = alpha lap(u)` with the stability limit
`r = alpha dt / h^2 <= 1/(2d)` enforced; Dirichlet (fixed boundary layer) and periodic
boundaries; Jacobi iteration for Laplace and Poisson (`--mode jacobi`). Every kernel implements
`u' = c0 u + c1 S` with the same summation order, so the backends agree to round-off.

## Quick start

```bash
pip install -e .[gpu]          # or .[dev] for the CPU-only reference + tests
pyheat3d list                  # variants and whether they run here
pyheat3d run --variant cupy-cuda-c/tb4 --dims 3 --n 256 --steps 400
pyheat3d validate --quick      # convergence orders + cross-backend agreement
python bench/run.py --quick    # < 2 min benchmark smoke run
```

## Validation

![Validation](figures/validation.png)

<!-- validation:start -->
- Heat equation, decaying sine mode vs the exact PDE solution: observed spatial order 1.97 (2D) / 1.99 (3D) with Dirichlet walls, 2.00 / 2.08 periodic; temporal order 1.00 / 1.00 against the exact-in-time semi-discrete solution.
- Jacobi steady states: Laplace order 1.99 (2D) / 1.98 (3D), Poisson (manufactured) 2.00 / 2.01; a quadratic harmonic function (on which the stencil is exact) is recovered to 5.3e-15.
- Cross-backend: 76/76 comparisons of 23 variants against the NumPy reference pass; worst fp32 relative deviation 5.1e-7 (FMA contraction only). fp16/bf16 variants are checked against a reference that applies the same storage rounding.
<!-- validation:end -->

References: (a) the exact PDE solution of a decaying sine mode, (b) its semi-discrete
(exact-in-time) solution, (c) the fully discrete FTCS solution, which every backend reproduces
to round-off, (d) analytic harmonic functions and a manufactured Poisson solution, (e) a NumPy
oracle on random fields with both boundary conditions (`tests/test_backends.py` checks
every variant, including grids smaller than one tile and step counts that are not a multiple of
k).

## Performance

![Roofline](figures/roofline.png)

![Scaling with problem size](figures/scaling.png)

<!-- perf-table:start -->
| dims | variant | storage | k | GLUPS | B/update | GB/s | % copy | % spec |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 3D 512^3 | `cupy-cuda-c/tb3` | float32 | 3 | 94.10 | 2.7 | 251 | 70 | 56 |
| 3D 512^3 | `cupy-cuda-c/tb4` | float32 | 4 | 92.47 | 2 | 185 | 51 | 41 |
| 3D 512^3 | `cupy-cuda-c/tb4-fp16` | float16 | 4 | 84.81 | 1 | 85 | 23 | 19 |
| 3D 512^3 | `triton/zmarch-bf16` | bfloat16 | 1 | 78.03 | 4 | 312 | 86 | 70 |
| 3D 512^3 | `triton/zmarch-fp16` | float16 | 1 | 72.53 | 4 | 290 | 80 | 65 |
| 3D 512^3 | `cupy-cuda-c/tb2` | float32 | 2 | 66.27 | 4 | 265 | 73 | 59 |
| 3D 512^3 | `triton/tile-bf16` | bfloat16 | 1 | 57.19 | 4 | 229 | 63 | 51 |
| 3D 512^3 | `triton/tile-fp16` | float16 | 1 | 56.23 | 4 | 225 | 62 | 50 |
| 3D 512^3 | `cupy-cuda-c/naive-fp16` | float16 | 1 | 51.90 | 4 | 208 | 58 | 46 |
| 3D 512^3 | `cupy-cuda-c/naive-bf16` | bfloat16 | 1 | 51.45 | 4 | 206 | 57 | 46 |
| 3D 512^3 | `numba/zmarch` | float32 | 1 | 40.99 | 8 | 328 | 91 | 73 |
| 3D 512^3 | `warp/zmarch` | float32 | 1 | 40.19 | 8 | 322 | 89 | 72 |
| 3D 512^3 | `cupy-cuda-c/zmarch` | float32 | 1 | 40.06 | 8 | 321 | 89 | 72 |
| 3D 512^3 | `triton/zmarch` | float32 | 1 | 38.33 | 8 | 307 | 85 | 68 |
| 3D 512^3 | `cupy-cuda-c/naive` | float32 | 1 | 30.57 | 8 | 245 | 68 | 55 |
| 3D 512^3 | `torch-compile/inductor` | float32 | 1 | 26.24 | 8 | 210 | 58 | 47 |
| 3D 512^3 | `triton/tile` | float32 | 1 | 24.31 | 8 | 194 | 54 | 43 |
| 3D 512^3 | `numba/naive` | float32 | 1 | 22.42 | 8 | 179 | 50 | 40 |
| 3D 512^3 | `warp/naive` | float32 | 1 | 20.79 | 8 | 166 | 46 | 37 |
| 3D 512^3 | `torch-eager/slicing` | float32 | 1 | 4.55 | 8 | 36 | 10 | 8 |
| 2D 16384^2 | `cupy-cuda-c/tb4-fp16` | float16 | 4 | 167.87 | 1 | 168 | 47 | 37 |
| 2D 16384^2 | `cupy-cuda-c/tb8` | float32 | 8 | 157.78 | 1 | 158 | 44 | 35 |
| 2D 16384^2 | `cupy-cuda-c/tb4` | float32 | 4 | 145.45 | 2 | 291 | 81 | 65 |
| 2D 16384^2 | `cupy-cuda-c/tb3` | float32 | 3 | 124.82 | 2.7 | 333 | 92 | 74 |
| 2D 16384^2 | `cupy-cuda-c/tb2` | float32 | 2 | 84.26 | 4 | 337 | 93 | 75 |
| 2D 16384^2 | `cupy-cuda-c/naive-bf16` | bfloat16 | 1 | 73.56 | 4 | 294 | 82 | 66 |
| 2D 16384^2 | `cupy-cuda-c/naive-fp16` | float16 | 1 | 71.87 | 4 | 287 | 80 | 64 |
| 2D 16384^2 | `triton/tile-bf16` | bfloat16 | 1 | 53.18 | 4 | 213 | 59 | 47 |
| 2D 16384^2 | `triton/tile-fp16` | float16 | 1 | 52.60 | 4 | 210 | 58 | 47 |
| 2D 16384^2 | `torch-compile/inductor` | float32 | 1 | 45.54 | 8 | 364 | 101 | 81 |
| 2D 16384^2 | `triton/tile` | float32 | 1 | 45.07 | 8 | 361 | 100 | 80 |
| 2D 16384^2 | `cupy-cuda-c/naive` | float32 | 1 | 43.76 | 8 | 350 | 97 | 78 |
| 2D 16384^2 | `cupy-cuda-c/ymarch` | float32 | 1 | 42.49 | 8 | 340 | 94 | 76 |
| 2D 16384^2 | `numba/naive` | float32 | 1 | 34.46 | 8 | 276 | 76 | 62 |
| 2D 16384^2 | `warp/naive` | float32 | 1 | 32.20 | 8 | 258 | 71 | 58 |
| 2D 16384^2 | `torch-eager/slicing` | float32 | 1 | 6.51 | 8 | 52 | 14 | 12 |

Source: `results/rtx3070-laptop.json`. GB/s uses the minimal traffic model (2 x bytes per value / k per update), so it is a lower bound on real DRAM traffic.
<!-- perf-table:end -->

<!-- counterparts:start -->
Same-GPU head-to-head with pystencils and AN5D (`bench/counterparts.py`, [figure](figures/counterparts.png), **provisional quick run on a shared GPU**): 2D 4096^2: pyheat3d triton/tile 36.7, pystencils tuned 37.6, pyheat3d cupy-cuda-c/tb8 126.7, AN5D 256-10-256 185.1 GLUPS; 3D 256^3: pyheat3d triton/zmarch 36.6, pystencils tuned 29.1, pyheat3d cupy-cuda-c/tb3 64.6, AN5D 32x32-5-128 59.0 GLUPS.
<!-- counterparts:end -->

## How it works

1. **NumPy / torch eager.** Slicing expressions; each operator is a full pass over memory.
2. **`torch.compile`.** The same Python expression becomes one fused kernel: one read and one
   write per site.
3. **Triton, Warp, Numba, CUDA C (naive).** One kernel, neighbours reused through L1/L2.
4. **2.5D blocking (`zmarch`).** A block owns an xy tile and marches in z. The z neighbours
   live in a register queue and the plane sits in shared memory (Micikevicius 2009).
5. **Temporal blocking (`tb<k>`).** A wavefront in (z, t): stage s of the pipeline advances
   the plane behind stage s-1 by one more time level, entirely in registers and shared memory or
   warp shuffles. Halos are recomputed redundantly (overlapped tiling) and only the centre of
   each tile is written, once per k steps. The 2D kernel is warp-synchronous, with no shared
   memory and no barriers.
6. **fp16/bf16 storage.** 2 B per value with fp32 arithmetic: half the traffic, but visibly
   less accurate on long diffusive runs (see the paper).

## Reproduce the paper

```bash
python bench/roofline.py                         # measure copy bandwidth + FP32 FMA peak
python bench/run.py --full                       # ~15 min on an idle GPU -> results/rtx3070-laptop.json
python bench/counterparts.py --full              # pystencils + AN5D head-to-head -> results/counterparts.json
                                                 # (needs ~/python-hpc/_ref sources, the cp-pystencils venv and nvcc;
                                                 #  setup in the bench/counterparts.py docstring)
pyheat3d validate --out results/validation.json  # convergence studies (GPU recommended)
python figures/make_figures.py                   # figures, paper/results.tex, README blocks
python figures/make_hero.py                      # docs/media/hero.gif
cd paper && tectonic paper.tex                   # paper/paper.pdf
```

## Citation

```bibtex
@software{hemati2026pyheat3d,
  author  = {Hemati, Walid},
  title   = {pyheat3d: explicit heat-equation and Jacobi stencils at the GPU memory roofline, from Python},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/walidhemati/pyheat3d}
}
```

See [CITATION.cff](CITATION.cff). Released under the [MIT license](LICENSE).

---

Part of the **Python-HPC suite**:
[pylbm](https://github.com/walidhemati/pylbm) ·
[pyheat3d](https://github.com/walidhemati/pyheat3d) ·
[pygravity](https://github.com/walidhemati/pygravity) ·
[pymd](https://github.com/walidhemati/pymd) ·
[pyising](https://github.com/walidhemati/pyising) ·
[pyspectral](https://github.com/walidhemati/pyspectral)
