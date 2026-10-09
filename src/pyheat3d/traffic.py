"""Memory-traffic and FLOP models used to turn lattice updates/s into bandwidth and roofline %.

Minimal traffic model (used for every reported "effective bandwidth"): a sweep that advances
the field by ``k`` time steps must read every value once and write every value once, so

    bytes per lattice-site update = 2 * bytes_per_value / k

i.e. 8 B for fp32 single-step kernels, 4 B for fp16/bf16 storage, 8/k B for temporal blocking
with ``k`` fused steps. Halo re-reads, cache misses and redundant halo computation are *not*
counted, so the effective bandwidth is a lower bound on the true DRAM traffic and can never be
inflated by an inefficient kernel. For temporally blocked kernels :func:`tb_model_bytes` adds
the halo over-fetch of overlapped tiling as a second, more realistic estimate.
"""

from __future__ import annotations

#: measured device-to-device copy bandwidth of the reference machine (suite-wide constant)
COPY_BW_GBS = 361.0
#: theoretical DRAM bandwidth of the RTX 3070 Laptop GPU (GA104, 256-bit GDDR6 @ 14 Gbps)
SPEC_BW_GBS = 448.0


def minimal_bytes(bytes_per_value: int, steps_per_pass: int = 1) -> float:
    """Minimal DRAM bytes per lattice-site update (one read + one write per sweep)."""
    return 2.0 * bytes_per_value / steps_per_pass


def flops_per_update(dims: int) -> int:
    """FLOPs per site update: (2d-1) neighbour adds + c1*S + c0*u + final add."""
    return (2 * dims - 1) + 3


def roofline_mlups(bw_gbs: float, bytes_per_update: float) -> float:
    """Upper bound on million lattice updates per second for a bandwidth ceiling."""
    return bw_gbs * 1e3 / bytes_per_update


def tb_model_bytes(dims: int, K: int, bytes_per_value: int, tx: int, ty: int, zch: int) -> float:
    """Estimated DRAM bytes per update of the overlapped-tiling temporal-blocking kernel.

    Every block reads its *extended* tile (output tile plus a K-deep halo on every tiled side,
    and K extra planes at each end of its z/y chunk) and writes its output tile once per K
    steps. Assumes no L2 reuse of halo reads between neighbouring blocks (pessimistic).
    """
    if dims == 3:
        over_xy = (tx * ty) / ((tx - 2 * K) * (ty - 2 * K))
    else:
        over_xy = tx / (tx - 2 * K)
    over_z = (zch + 2 * K) / zch
    return bytes_per_value * (over_xy * over_z + 1.0) / K
