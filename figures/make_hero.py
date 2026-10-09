"""Render the README hero animation: a periodic 3D temperature field relaxing under the heat
equation, shown as an isometric cube (three outer faces) next to its mid-plane slice.

    python figures/make_hero.py            # -> docs/media/hero.gif (+ hero.png, a still)

Uses the fastest available GPU variant, otherwise the NumPy reference.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402
from PIL import Image  # noqa: E402

from pyheat3d import heat_stencil, make_stepper  # noqa: E402
from pyheat3d.registry import availability  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "media"
N = 96
FRAMES = 36
C30, S30 = math.cos(math.pi / 6), math.sin(math.pi / 6)


def initial_field(n: int, seed: int = 3) -> np.ndarray:
    """Gaussian random field with a k^-3.5 power spectrum (multi-scale hot/cold blobs)."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n) * n
    kz, ky, kx = np.meshgrid(k, k, k, indexing="ij")
    kk = np.sqrt(kx**2 + ky**2 + kz**2)
    kk[0, 0, 0] = 1.0
    amp = kk ** (-3.5 / 2)
    amp[0, 0, 0] = 0.0
    phase = rng.normal(size=(n, n, n)) + 1j * rng.normal(size=(n, n, n))
    u = np.real(np.fft.ifftn(amp * phase))
    u = (u - u.mean()) / (3 * u.std())
    return u.astype(np.float32)


def pick_variant() -> str:
    for key in ("cupy-cuda-c/naive", "triton/tile", "torch-eager/slicing"):
        if availability(key)[0]:
            return key
    return "numpy/slicing"


def draw(u: np.ndarray, t: float, step: int, vmax: float, key: str) -> Image.Image:
    fig = plt.figure(figsize=(6.4, 3.2), dpi=100)
    fig.patch.set_facecolor("#0d1117")
    ax = fig.add_axes((0.0, 0.02, 0.56, 0.86))
    ax.set_facecolor("#0d1117")
    kw = {"cmap": "inferno", "vmin": -vmax, "vmax": vmax, "interpolation": "bilinear",
          "origin": "lower", "extent": (0, 1, 0, 1)}
    faces = (
        (u[-1], Affine2D.from_values(C30, -S30, -C30, -S30, 0, 1)),  # top z=1: (x, y)
        (u[:, :, -1], Affine2D.from_values(-C30, -S30, 0, 1, C30, -S30)),  # x=1: (y, z)
        (u[:, -1, :], Affine2D.from_values(C30, -S30, 0, 1, -C30, -S30)),  # y=1: (x, z)
    )
    for img, tr in faces:
        ax.imshow(img, transform=tr + ax.transData, **kw)
    # cube edges
    def P(x, y, z):
        return (C30 * x - C30 * y, -S30 * x - S30 * y + z)

    for a, b in [((0, 1, 1), (1, 1, 1)), ((1, 0, 1), (1, 1, 1)), ((1, 1, 0), (1, 1, 1)),
                 ((0, 1, 1), (0, 0, 1)), ((1, 0, 1), (0, 0, 1)), ((0, 1, 0), (0, 1, 1)),
                 ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (1, 1, 0)), ((1, 0, 0), (1, 1, 0))]:
        pa, pb = P(*a), P(*b)
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="w", lw=0.6, alpha=0.7)
    ax.set_xlim(-C30 - 0.05, C30 + 0.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax2 = fig.add_axes((0.6, 0.12, 0.36, 0.72))
    ax2.imshow(u[u.shape[0] // 2], cmap="inferno", vmin=-vmax, vmax=vmax, origin="lower",
               interpolation="bilinear")
    ax2.set_xticks([])
    ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_color("w")
        s.set_linewidth(0.6)
        s.set_visible(True)
    ax2.set_title("mid-plane slice", color="w", fontsize=8, pad=3)
    fig.text(0.03, 0.93, f"pyheat3d  |  {N}$^3$ periodic heat equation, 7-point FTCS",
             color="w", fontsize=9, family="serif")
    fig.text(0.6, 0.04, f"step {step:5d}   t = {t:.2e}   ({key})", color="#bbbbbb",
             fontsize=6.5, family="serif")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    key = pick_variant()
    u0 = initial_field(N)
    r = 1.0 / 6.0
    h = 1.0 / N
    st = make_stepper(key, u0, heat_stencil(3, r, "periodic"))
    vmax = float(np.abs(u0).max()) * 0.8
    # geometric frame spacing: early frames show fast small-scale decay, late ones the slow modes
    targets = np.unique(np.round(np.geomspace(1, 3000, FRAMES - 1)).astype(int))
    frames = [draw(u0, 0.0, 0, vmax, key)]
    done = 0
    for tgt in targets:
        st.step(int(tgt - done))
        done = int(tgt)
        st.synchronize()
        frames.append(draw(st.get(), done * r * h * h, done, vmax, key))
    frames += [frames[-1]] * 6
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
           for f in frames]
    path = OUT / "hero.gif"
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=110, loop=0, optimize=True)
    frames[len(frames) // 3].save(OUT / "hero.png")
    print(f"{path} ({path.stat().st_size / 1e6:.2f} MB, {len(frames)} frames, {key})")


if __name__ == "__main__":
    main()
