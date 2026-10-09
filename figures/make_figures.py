"""Build every figure, ``paper/results.tex`` and the README result blocks from results/*.json.

    python figures/make_figures.py

Benchmark source, in order of preference: results/rtx3070-laptop.json (``--full``),
results/dev-provisional.json (``--dev``), results/quick.json (``--quick``). Validation numbers
come from results/validation.json (``pyheat3d validate --out results/validation.json``).
Nothing in the paper or README results blocks is typed by hand.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
PAPER = ROOT / "paper"

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
COL, FULL = 3.4, 7.0
COLORS = {"numpy": "#7f7f7f", "torch-eager": "#9467bd", "torch-compile": "#8c564b",
          "triton": "#1f77b4", "warp": "#2ca02c", "numba": "#ff7f0e", "cupy-cuda-c": "#d62728"}
BACKEND_ORDER = list(COLORS)
LOWP = ("float16", "bfloat16")


# ------------------------------------------------------------------------------------ loading
def load_bench() -> tuple[dict, str]:
    for name in ("rtx3070-laptop.json", "dev-provisional.json", "quick.json"):
        p = RES / name
        if p.exists():
            return json.loads(p.read_text()), name
    raise SystemExit("no benchmark JSON in results/; run bench/run.py first")


def load_validation() -> dict | None:
    p = RES / "validation.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_roofline() -> dict | None:
    p = RES / "roofline.json"
    return json.loads(p.read_text()) if p.exists() else None


def cells(rec: dict) -> int:
    return math.prod(rec["size"])


def largest(runs: list[dict], dims: int) -> list[dict]:
    """Records at the largest size benchmarked for ``dims`` on the GPU."""
    gpu = [r for r in runs if r["dims"] == dims and r["backend"] != "numpy"]
    if not gpu:
        return []
    n = max(cells(r) for r in gpu)
    return [r for r in runs if r["dims"] == dims and cells(r) == n]


def is_tb(r: dict) -> bool:
    return r.get("steps_per_pass", 1) > 1


def is_low(r: dict) -> bool:
    return r["dtype"] in LOWP


def is_tb_fp32(r: dict) -> bool:
    return is_tb(r) and not is_low(r)


def best(recs: list[dict], pred=lambda r: True) -> dict | None:
    c = [r for r in recs if pred(r)]
    return max(c, key=lambda r: r["median"]) if c else None


def size_label(rec: dict, tex: bool = False) -> str:
    s = rec["size"]
    if len(set(s)) == 1:
        return f"${s[0]}^{len(s)}$" if tex else f"{s[0]}^{len(s)}"
    return "x".join(map(str, s))


# ------------------------------------------------------------------------------------ figures
def _save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


def fig_roofline_bars(bench: dict) -> None:
    ceil = bench["ceiling"]["copy_bandwidth_GBs"]
    spec = bench["ceiling"]["spec_bandwidth_GBs"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 4.6))
    for ax, dims in zip(axes, (2, 3)):
        recs = largest(bench["runs"], dims)
        if not recs:
            ax.set_visible(False)
            continue
        recs = sorted(recs, key=lambda r: (BACKEND_ORDER.index(r["backend"]), r["rung"],
                                           r["variant"]))
        y = np.arange(len(recs))[::-1]
        g = [r["median"] / 1e3 for r in recs]
        for yi, r, gi in zip(y, recs, g):
            hatch = "////" if is_low(r) else ("xx" if is_tb(r) else None)
            ax.barh(yi, gi, color=COLORS[r["backend"]], alpha=0.85 if not is_tb(r) else 0.6,
                    hatch=hatch, edgecolor="white" if hatch else "none", linewidth=0)
            ax.text(gi * 1.02, yi, f"{r['pct_copy_bw']:.0f}% BW", va="center", fontsize=6.5)
        ax.set_yticks(y, [f"{r['backend']}/{r['variant']}" for r in recs], fontsize=7)
        roof = ceil / 8.0
        ax.axvline(roof, ls="--", color="k", lw=1)
        ax.axvline(spec / 8.0, ls=":", color="k", lw=0.8)
        ax.axvline(ceil / 4.0, ls="-.", color="0.4", lw=0.8)
        top = max(max(g), ceil / 4.0) * 1.25
        ax.set_xlim(0, top)
        ax.text(roof, len(recs) - 0.3, " fp32 roof\n (copy BW / 8 B)", fontsize=6.5, va="top")
        ax.text(ceil / 4.0, len(recs) - 0.3, " 2-byte roof", fontsize=6.5, va="top",
                color="0.3")
        ax.set_xlabel("GLUPS (10$^9$ lattice-site updates / s)")
        ax.set_title(f"{dims}D, {size_label(recs[0], tex=True)} grid", fontsize=9)
    fig.text(0.01, -0.02, "Dashed: single-step fp32 roofline (361 GB/s copy / 8 B); dotted: same "
             "for 448 GB/s spec; dash-dot: 2-byte storage roofline. Hatched '//': fp16/bf16 "
             "storage; 'xx': temporal blocking (k steps per DRAM pass, roofline k times higher). "
             "Labels: share of copy bandwidth used under the minimal traffic model.",
             fontsize=6.5, wrap=True)
    fig.tight_layout()
    _save(fig, "roofline")


def fig_roofline_classic(bench: dict, roof: dict | None) -> None:
    bw = bench["ceiling"]["copy_bandwidth_GBs"]
    peak = (roof or {}).get("fp32_fma_tflops") or 9.4
    peak_label = "FP32 FMA (measured)" if roof else "SGEMM 9.4 TFLOP/s"
    fig, ax = plt.subplots(figsize=(COL, 2.8))
    ai = np.logspace(-1.3, 2.3, 200)
    ax.loglog(ai, np.minimum(bw * ai, peak * 1e3), "k--", lw=1)
    ax.text(0.07, bw * 0.07 * 1.4, f"{bw:.0f} GB/s", rotation=33, fontsize=7)
    ax.text(30, peak * 1e3 * 1.12, f"{peak_label}: {peak:.1f} TFLOP/s", fontsize=6.5, ha="center")
    for dims, mk in ((2, "s"), (3, "o")):
        for r in largest(bench["runs"], dims):
            x = r["flops_per_unit"] / r["bytes_per_unit"]
            ax.loglog(x, r["GFLOPs"], mk, ms=4, color=COLORS[r["backend"]],
                      mfc="none" if is_tb(r) else COLORS[r["backend"]], mew=1)
    ax.set_xlabel("FLOP / byte (minimal traffic model)")
    ax.set_ylabel("GFLOP/s")
    ax.set_ylim(1, peak * 3e3)
    from matplotlib.lines import Line2D

    h = [Line2D([], [], marker="o", ls="", color="k", ms=4, label="3D"),
         Line2D([], [], marker="s", ls="", color="k", ms=4, label="2D"),
         Line2D([], [], marker="o", ls="", color="k", mfc="none", ms=4,
                label="temporal blocking")]
    ax.legend(handles=h, fontsize=6.5, frameon=False, loc="upper left")
    _save(fig, "roofline_classic")


def fig_scaling(bench: dict) -> None:
    ceil = bench["ceiling"]["copy_bandwidth_GBs"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 3.9), sharey=True)
    for ax, dims in zip(axes, (2, 3)):
        runs = [r for r in bench["runs"] if r["dims"] == dims]
        top = largest(bench["runs"], dims)
        if not top:
            continue
        for b in BACKEND_ORDER:
            cand = best([r for r in top if r["backend"] == b],
                        lambda r: not is_tb(r) and not is_low(r))
            if cand is None:
                cand = best([r for r in runs if r["backend"] == b],
                            lambda r: not is_tb(r) and not is_low(r))
            if cand is None:
                continue
            series = sorted([r for r in runs if r["key"] == cand["key"]], key=cells)
            ax.loglog([cells(r) for r in series], [r["median"] / 1e3 for r in series], "o-",
                      color=COLORS[b], ms=3, lw=1.2, label=cand["key"])
        for pred, ls, lab in ((is_tb_fp32, "--", "best fp32 temporal blocking"),
                              (lambda r: is_low(r) and not is_tb(r), ":",
                               "best fp16/bf16 single-step")):
            c = best(top, pred)
            if c is None:
                continue
            series = sorted([r for r in runs if r["key"] == c["key"]], key=cells)
            ax.loglog([cells(r) for r in series], [r["median"] / 1e3 for r in series], "^" + ls,
                      color=COLORS[c["backend"]], ms=3, lw=1, label=f"{c['key']} ({lab})")
        ax.axhline(ceil / 8.0, ls="--", color="k", lw=0.8)
        ax.text(ax.get_xlim()[0] * 1.3 if ax.get_xlim()[0] > 0 else 1, ceil / 8.0 * 1.08,
                "fp32 single-step roofline", fontsize=6.5)
        ax.set_xlabel("lattice sites")
        ax.set_title(f"{dims}D", fontsize=9)
        ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2),
                  ncol=2)
    axes[0].set_ylabel("GLUPS")
    fig.tight_layout()
    _save(fig, "scaling")


def fig_validation(val: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.4),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    ax = axes[0]
    for s, mk in zip(val["spatial"], "osD^"):
        h = [r["h"] for r in s["runs"]]
        e = [r["err_exact"] for r in s["runs"]]
        ax.loglog(h, e, mk + "-", ms=3, lw=1,
                  label=f"{s['dims']}D {s['bc'][:3]} ({s['order']:.2f})")
    h = np.array([r["h"] for r in val["spatial"][0]["runs"]])
    e0 = val["spatial"][0]["runs"][0]["err_exact"]
    ax.loglog(h, e0 * 0.4 * (h / h[0]) ** 2, "k:", lw=0.8, label="slope 2")
    ax.set_xlabel("$h$")
    ax.set_ylabel("rel. max error")
    ax.set_title("(a) heat, space", fontsize=8)
    ax.legend(fontsize=5.5, frameon=False)
    ax = axes[1]
    for s, mk in zip(val["temporal"], "os"):
        dt = [r["dt"] for r in s["runs"]]
        e = [r["err_semidiscrete"] for r in s["runs"]]
        ax.loglog(dt, e, mk + "-", ms=3, lw=1, label=f"{s['dims']}D ({s['order']:.2f})")
    dt = np.array([r["dt"] for r in val["temporal"][0]["runs"]])
    e0 = val["temporal"][0]["runs"][0]["err_semidiscrete"]
    ax.loglog(dt, e0 * 0.5 * dt / dt[0], "k:", lw=0.8, label="slope 1")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_title("(b) heat, time", fontsize=8)
    ax.legend(fontsize=5.5, frameon=False)
    ax = axes[2]
    for key, mk, name in (("laplace", "o", "Laplace"), ("poisson", "s", "Poisson")):
        for s in val[key]:
            h = [r["h"] for r in s["runs"]]
            e = [r["err"] for r in s["runs"]]
            ax.loglog(h, e, mk + ("-" if s["dims"] == 2 else "--"), ms=3, lw=1,
                      label=f"{name} {s['dims']}D ({s['order']:.2f})")
    ax.set_xlabel("$h$")
    ax.set_title("(c) Jacobi steady state", fontsize=8)
    ax.legend(fontsize=5.5, frameon=False)
    ax = axes[3]
    cb = val["cross_backend"]
    keys = []
    for r in cb:
        if r["variant"] not in keys and r["variant"] != "numpy/slicing":
            keys.append(r["variant"])
    worst = [max(x["rel_err"] for x in cb if x["variant"] == k) for k in keys]
    tol = [cb[[x["variant"] for x in cb].index(k)]["tolerance"] for k in keys]
    y = np.arange(len(keys))[::-1]
    for yi, k, w, t in zip(y, keys, worst, tol):
        ax.barh(yi, max(w, 1e-9), color=COLORS[k.split("/")[0]], left=0)
        ax.plot([t], [yi], "k|", ms=5)
    ax.set_xscale("log")
    ax.set_xlim(1e-8, 1e-1)
    ax.set_xticks([1e-7, 1e-4, 1e-1])
    ax.set_yticks(y, keys, fontsize=4.6)
    ax.set_xlabel("max rel. deviation")
    ax.set_title("(d) all variants", fontsize=8)
    fig.tight_layout(w_pad=0.6)
    _save(fig, "validation")


# ------------------------------------------------------------------------------- results.tex
def _fmt(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}"


def _sci(x: float) -> str:
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10**e
    return f"{m:.1f}\\times10^{{{e}}}"


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("^", r"\^{}")


def _size_tex(rec: dict) -> str:
    s = rec["size"]
    return f"{s[0]}^{len(s)}" if len(set(s)) == 1 else r"\times".join(map(str, s))


def bench_macros(bench: dict, src: str) -> dict[str, str]:
    m: dict[str, str] = {}
    mach = bench["machine"]
    m["benchSource"] = _tex_escape(src)
    m["benchDate"] = _tex_escape(str(mach.get("date", "?"))[:10])
    m["gpuName"] = _tex_escape(str(mach.get("gpu", "?")))
    m["driverVersion"] = _tex_escape(str(mach.get("driver", "?")))
    for k in ("torch", "triton", "warp", "cupy", "numba", "python", "numpy"):
        m[k + "Version"] = _tex_escape(str(mach.get(k, "?")))
    m["copyBW"] = _fmt(bench["ceiling"]["copy_bandwidth_GBs"], 0)
    m["specBW"] = _fmt(bench["ceiling"]["spec_bandwidth_GBs"], 0)
    mc = bench["ceiling"].get("measured_copy_bandwidth_GBs")
    m["measuredCopyBW"] = _fmt(mc, 0) if mc else "n/a"
    ceil = bench["ceiling"]["copy_bandwidth_GBs"]
    m["rooflineFP"] = _fmt(ceil / 8.0, 1)
    m["rooflineHalf"] = _fmt(ceil / 4.0, 1)
    m["nRuns"] = str(len(bench["runs"]))
    rows = []
    for dims, D in ((2, "TwoD"), (3, "ThreeD")):
        top = largest(bench["runs"], dims)
        if not top:
            continue
        m[f"size{D}"] = _size_tex(top[0])
        b1 = best(top, lambda r: not is_tb(r) and not is_low(r))
        bt = best(top, is_tb_fp32)
        bl = best(top, lambda r: is_low(r) and not is_tb(r))
        for tag, r in (("Single", b1), ("TB", bt), ("Low", bl)):
            if r is None:
                continue
            m[f"best{tag}{D}Variant"] = _tex_escape(r["key"])
            m[f"best{tag}{D}GLUPS"] = _fmt(r["median"] / 1e3, 1)
            m[f"best{tag}{D}BW"] = _fmt(r["effective_GBs"], 0)
            m[f"best{tag}{D}PctCopy"] = _fmt(r["pct_copy_bw"], 0)
            m[f"best{tag}{D}PctSpec"] = _fmt(r["pct_spec_bw"], 0)
            m[f"best{tag}{D}Min"] = _fmt(r["min"] / 1e3, 1)
            m[f"best{tag}{D}Max"] = _fmt(r["max"] / 1e3, 1)
        if bt is not None:
            m[f"bestTB{D}K"] = str(bt["steps_per_pass"])
            m[f"bestTB{D}Equiv"] = _fmt(bt["single_step_equiv_GBs"], 0)
            m[f"bestTB{D}OverRoof"] = _fmt(bt["median"] / 1e3 / (ceil / 8.0), 2)
            if "halo_model_GBs" in bt:
                m[f"bestTB{D}HaloBW"] = _fmt(bt["halo_model_GBs"], 0)
                m[f"bestTB{D}HaloBytes"] = _fmt(bt["halo_model_bytes_per_unit"], 2)
        npy = best([r for r in bench["runs"] if r["dims"] == dims and r["backend"] == "numpy"])
        if npy is not None:
            m[f"numpy{D}GLUPS"] = _fmt(npy["median"] / 1e3, 3)
            m[f"numpy{D}Size"] = _size_tex(npy)
            same = [r for r in bench["runs"] if r["dims"] == dims and cells(r) == cells(npy)]
            b_same = best(same, lambda r: not is_tb(r) and not is_low(r))
            if b_same is not None:
                m[f"speedupNumpy{D}"] = _fmt(b_same["median"] / npy["median"], 0)
        # one table row per variant at the largest size
        for r in sorted(top, key=lambda r: (BACKEND_ORDER.index(r["backend"]), r["rung"],
                                            r["variant"])):
            extra = (f"{r['single_step_equiv_GBs']:.0f}" if is_tb(r) else "--")
            rows.append(
                f"{dims}D & \\texttt{{{_tex_escape(r['backend'])}}} & "
                f"\\texttt{{{_tex_escape(r['variant'])}}} & {r['dtype'].replace('float', 'fp')}"
                f" & {r['steps_per_pass']} & {r['median'] / 1e3:.2f} & "
                f"{r['bytes_per_unit']:.2g} & {r['effective_GBs']:.0f} & "
                f"{r['pct_copy_bw']:.0f} & {r['pct_spec_bw']:.0f} & {extra} \\\\")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()
    m["perfTableRows"] = "\n".join(rows)
    return m


def validation_macros(val: dict) -> dict[str, str]:
    m: dict[str, str] = {}
    names = {(2, "dirichlet"): "TwoD", (3, "dirichlet"): "ThreeD",
             (2, "periodic"): "TwoDPer", (3, "periodic"): "ThreeDPer"}
    for s in val["spatial"]:
        k = names[(s["dims"], s["bc"])]
        m[f"spatialOrder{k}"] = _fmt(s["order"], 2)
        m[f"spatialErrFinest{k}"] = _sci(s["runs"][-1]["err_exact"])
        m[f"spatialNFinest{k}"] = str(s["runs"][-1]["n"])
    for s in val["temporal"]:
        k = "TwoD" if s["dims"] == 2 else "ThreeD"
        m[f"temporalOrder{k}"] = _fmt(s["order"], 2)
        m[f"temporalN{k}"] = str(s["n"])
    for key in ("laplace", "poisson"):
        for s in val[key]:
            k = "TwoD" if s["dims"] == 2 else "ThreeD"
            m[f"{key}Order{k}"] = _fmt(s["order"], 2)
            m[f"{key}ErrFinest{k}"] = _sci(s["runs"][-1]["err"])
    for q in val["laplace_quadratic"]:
        k = "TwoD" if q["dims"] == 2 else "ThreeD"
        m[f"quadErr{k}"] = _sci(q["err"])
    m["fpVariant"] = _tex_escape(val["fp64_variant"])
    cb = val["cross_backend"]
    m["nCross"] = str(len(cb))
    m["nCrossPass"] = str(sum(x["pass"] for x in cb))
    m["nVariants"] = str(len({x["variant"] for x in cb}))
    for st, k in (("float32", "FP"), ("float16", "Half"), ("bfloat16", "BF")):
        errs = [x["rel_err"] for x in cb if x["storage"] == st]
        if errs:
            m[f"maxCrossErr{k}"] = _sci(max(errs))
    m["nBitwise"] = str(sum(x["bitwise"] for x in cb if x["variant"] != "numpy/slicing"))
    fb = val["fourier_backends"]
    for dims, D in ((2, "TwoD"), (3, "ThreeD")):
        fp = [x for x in fb if x["dims"] == dims
              and not any(t in x["variant"] for t in ("fp16", "bf16"))]
        if fp:
            m[f"fourierErr{D}"] = _sci(float(np.median([x["err_exact"] for x in fp])))
            m[f"fourierDiscErr{D}"] = _sci(max(x["err_discrete"] for x in fp))
            m[f"fourierN{D}"] = str(fp[0]["n"])
            m[f"fourierSteps{D}"] = str(fp[0]["steps"])
        for tag, D2 in (("naive-fp16", "Half"), ("naive-bf16", "BF"), ("tb4-fp16", "TBHalf")):
            c = [x for x in fb if x["dims"] == dims and x["variant"] == f"cupy-cuda-c/{tag}"]
            if c:
                m[f"fourierErr{D2}{D}"] = _sci(c[0]["err_exact"])
    return m


def write_results_tex(bench: dict, src: str, val: dict | None) -> None:
    m = bench_macros(bench, src)
    if val is not None:
        m.update(validation_macros(val))
    lines = ["% GENERATED by figures/make_figures.py from results/*.json -- do not edit.",
             f"% benchmark source: {src}"]
    lines.append(r"\provisionaltrue" if bench.get("provisional", True) else r"\provisionalfalse")
    for k, v in m.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (PAPER / "results.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------- README
def _replace_block(text: str, tag: str, body: str) -> str:
    pat = re.compile(rf"(<!-- {tag}:start -->)(.*?)(<!-- {tag}:end -->)", re.S)
    return pat.sub(lambda mo: mo.group(1) + "\n" + body.strip() + "\n" + mo.group(3), text)


def update_readme(bench: dict, src: str, val: dict | None) -> None:
    p = ROOT / "README.md"
    if not p.exists():
        return
    text = p.read_text()
    ceil = bench["ceiling"]["copy_bandwidth_GBs"]
    prov = bench.get("provisional", True)
    gpu = bench["machine"].get("gpu", "?")
    head = []
    if prov:
        head.append(f"> **Provisional numbers** (`results/{src}`, measured on a shared GPU "
                    "during development; the final numbers come from `bench/run.py --full` on "
                    "an idle GPU).\n>")
    for dims in (3, 2):
        top = largest(bench["runs"], dims)
        b1 = best(top, lambda r: not is_tb(r) and not is_low(r))
        bt = best(top, is_tb_fp32)
        if b1 is None:
            continue
        head.append(f"> **{dims}D {size_label(b1)}, single step:** `{b1['key']}` "
                    f"**{b1['median'] / 1e3:.1f} GLUPS** = {b1['effective_GBs']:.0f} GB/s = "
                    f"**{b1['pct_copy_bw']:.0f}% of the {ceil:.0f} GB/s copy bandwidth** "
                    f"({b1['pct_spec_bw']:.0f}% of the 448 GB/s spec).<br>")
        if bt is not None:
            head.append(f"> **{dims}D temporal blocking (k={bt['steps_per_pass']}):** "
                        f"`{bt['key']}` **{bt['median'] / 1e3:.1f} GLUPS**, "
                        f"{bt['median'] / 1e3 / (ceil / 8):.2f}x the single-step roofline of "
                        f"{ceil / 8:.1f} GLUPS while moving an estimated "
                        f"{bt['effective_GBs']:.0f} GB/s (minimal model) to "
                        f"{bt.get('halo_model_GBs', bt['effective_GBs']):.0f} GB/s (with halo "
                        "over-fetch) of DRAM traffic. The roofline per step drops k-fold; the "
                        "hardware is not beaten.<br>")
    head.append(f"> {gpu}, {bench['machine'].get('date', '')[:10]}.")
    text = _replace_block(text, "headline", "\n".join(head))
    rows = ["| dims | variant | storage | k | GLUPS | B/update | GB/s | % copy | % spec |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for dims in (3, 2):
        for r in sorted(largest(bench["runs"], dims),
                        key=lambda r: -r["median"]):
            rows.append(f"| {dims}D {size_label(r)} | `{r['key']}` | {r['dtype']} | "
                        f"{r['steps_per_pass']} | {r['median'] / 1e3:.2f} | "
                        f"{r['bytes_per_unit']:.2g} | {r['effective_GBs']:.0f} | "
                        f"{r['pct_copy_bw']:.0f} | {r['pct_spec_bw']:.0f} |")
    note = (f"\nSource: `results/{src}`" + (" (provisional)." if prov else ".") +
            " GB/s uses the minimal traffic model (2 x bytes per value / k per update), so it "
            "is a lower bound on real DRAM traffic.")
    text = _replace_block(text, "perf-table", "\n".join(rows) + "\n" + note)
    if val is not None:
        vm = validation_macros(val)
        vt = [f"- Heat equation, decaying sine mode vs the exact PDE solution: observed spatial "
              f"order {vm.get('spatialOrderTwoD')} (2D) / {vm.get('spatialOrderThreeD')} (3D) "
              f"with Dirichlet walls, {vm.get('spatialOrderTwoDPer')} / "
              f"{vm.get('spatialOrderThreeDPer')} periodic; temporal order "
              f"{vm.get('temporalOrderTwoD')} / {vm.get('temporalOrderThreeD')} against the "
              "exact-in-time semi-discrete solution.",
              f"- Jacobi steady states: Laplace order {vm.get('laplaceOrderTwoD')} (2D) / "
              f"{vm.get('laplaceOrderThreeD')} (3D), Poisson (manufactured) "
              f"{vm.get('poissonOrderTwoD')} / {vm.get('poissonOrderThreeD')}; a quadratic "
              "harmonic function (on which the stencil is exact) is recovered to "
              f"${vm.get('quadErrThreeD', vm.get('quadErrTwoD'))}$.",
              f"- Cross-backend: {vm['nCrossPass']}/{vm['nCross']} comparisons of "
              f"{vm['nVariants']} variants against the NumPy reference pass; worst fp32 "
              f"relative deviation ${vm.get('maxCrossErrFP')}$ (FMA contraction only). fp16/bf16 "
              "variants are checked against a reference that applies the same storage rounding."]
        body = re.sub(r"\$([\d.]+)\\times10\^\{(-?\d+)\}\$", r"\1e\2", "\n".join(vt))
        text = _replace_block(text, "validation", body)
    p.write_text(text)


def main() -> None:
    bench, src = load_bench()
    val = load_validation()
    roof = load_roofline()
    FIG.mkdir(exist_ok=True)
    fig_roofline_bars(bench)
    fig_roofline_classic(bench, roof)
    fig_scaling(bench)
    if val is not None:
        fig_validation(val)
    write_results_tex(bench, src, val)
    update_readme(bench, src, val)
    print(f"figures + paper/results.tex + README blocks from results/{src}"
          + (" (PROVISIONAL)" if bench.get("provisional", True) else ""))


if __name__ == "__main__":
    main()
