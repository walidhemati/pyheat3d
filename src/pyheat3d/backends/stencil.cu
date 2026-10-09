// pyheat3d hand-written CUDA C reference kernels, compiled at runtime with NVRTC (CuPy).
//
// Update (all kernels, canonical summation order):
//   u'[c] = c0*u[c] + c1*( ((u[x-1]+u[x+1]) + (u[y-1]+u[y+1])) + (u[z-1]+u[z+1]) )
// Arrays are C-ordered (z, y, x) / (y, x); x is contiguous.
//
// Compile-time macros (set by the Python side):
//   STORAGE   0 = fp32, 1 = fp16, 2 = bf16 storage (arithmetic is always fp32)
//   PERIODIC  0 = Dirichlet (boundary layer fixed, never written), 1 = periodic
//   BX, BY    thread-block shape of the z-marching kernel
//   VEC       sites per thread in the naive kernels (x direction, strided by blockDim.x)
//   ZCH       z-planes (y-rows in 2D) marched per block
//   K         time steps fused per sweep in the temporal-blocking kernels
//   TX, TY    extended tile (including the K-deep halo) of the temporal-blocking kernels

#if STORAGE == 1
#include <cuda_fp16.h>
typedef __half T;
__device__ __forceinline__ float ld(const T* __restrict__ p, size_t i) { return __half2float(p[i]); }
__device__ __forceinline__ void st(T* __restrict__ p, size_t i, float v) { p[i] = __float2half_rn(v); }
#elif STORAGE == 2
#include <cuda_bf16.h>
typedef __nv_bfloat16 T;
__device__ __forceinline__ float ld(const T* __restrict__ p, size_t i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ void st(T* __restrict__ p, size_t i, float v) { p[i] = __float2bfloat16_rn(v); }
#else
typedef float T;
__device__ __forceinline__ float ld(const T* __restrict__ p, size_t i) { return __ldg(p + i); }
__device__ __forceinline__ void st(T* __restrict__ p, size_t i, float v) { p[i] = v; }
#endif

__device__ __forceinline__ int wrapi(int i, int n) { i %= n; return i < 0 ? i + n : i; }
__device__ __forceinline__ int clampi(int i, int lo, int hi) { return i < lo ? lo : (i > hi ? hi : i); }

// ---------------------------------------------------------------------------------------------
// Rung "naive": one thread per lattice site, every operand loaded through L1/L2.
// ---------------------------------------------------------------------------------------------
#ifndef VEC
#define VEC 1
#endif
// VEC consecutive-in-blockDim.x sites per thread (coalesced per instruction): more bytes in
// flight per thread, which the 2-byte storage variants need to saturate DRAM.
extern "C" __global__ void naive2d(const T* __restrict__ u, T* __restrict__ o,
                                   int ny, int nx, float c0, float c1)
{
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= ny) return;
#pragma unroll
    for (int v = 0; v < VEC; ++v) {
        const int i = (blockIdx.x * VEC + v) * blockDim.x + threadIdx.x;
        if (i >= nx) continue;
#if PERIODIC
        const int im = i == 0 ? nx - 1 : i - 1, ip = i == nx - 1 ? 0 : i + 1;
        const int jm = j == 0 ? ny - 1 : j - 1, jp = j == ny - 1 ? 0 : j + 1;
#else
        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) continue;
        const int im = i - 1, ip = i + 1, jm = j - 1, jp = j + 1;
#endif
        const size_t r = (size_t)j * nx;
        const float s = (ld(u, r + im) + ld(u, r + ip)) +
                        (ld(u, (size_t)jm * nx + i) + ld(u, (size_t)jp * nx + i));
        st(o, r + i, c0 * ld(u, r + i) + c1 * s);
    }
}

extern "C" __global__ void naive3d(const T* __restrict__ u, T* __restrict__ o,
                                   int nz, int ny, int nx, float c0, float c1)
{
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (j >= ny || k >= nz) return;
    const size_t P = (size_t)nx * ny;
#pragma unroll
    for (int v = 0; v < VEC; ++v) {
        const int i = (blockIdx.x * VEC + v) * blockDim.x + threadIdx.x;
        if (i >= nx) continue;
#if PERIODIC
        const int im = i == 0 ? nx - 1 : i - 1, ip = i == nx - 1 ? 0 : i + 1;
        const int jm = j == 0 ? ny - 1 : j - 1, jp = j == ny - 1 ? 0 : j + 1;
        const int km = k == 0 ? nz - 1 : k - 1, kp = k == nz - 1 ? 0 : k + 1;
#else
        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1) continue;
        const int im = i - 1, ip = i + 1, jm = j - 1, jp = j + 1, km = k - 1, kp = k + 1;
#endif
        const size_t b = (size_t)k * P + (size_t)j * nx;
        const float sx = ld(u, b + im) + ld(u, b + ip);
        const float sy = ld(u, (size_t)k * P + (size_t)jm * nx + i) +
                         ld(u, (size_t)k * P + (size_t)jp * nx + i);
        const float sz = ld(u, (size_t)km * P + (size_t)j * nx + i) +
                         ld(u, (size_t)kp * P + (size_t)j * nx + i);
        st(o, b + i, c0 * ld(u, b + i) + c1 * ((sx + sy) + sz));
    }
}

// ---------------------------------------------------------------------------------------------
// Rung "zmarch" (2.5D blocking, Micikevicius 2009): a (BX, BY) block owns an xy tile and marches
// ZCH planes in z. The current plane (plus a 1-wide halo) is staged in shared memory; the z
// neighbours are kept in a per-thread register queue (below, center, above), and the next
// plane is prefetched one iteration ahead. Each plane is read from DRAM ~once per sweep.
// ---------------------------------------------------------------------------------------------
#if defined(BX) && defined(BY) && defined(ZCH)
extern "C" __global__ void __launch_bounds__(BX * BY)
zmarch3d(const T* __restrict__ u, T* __restrict__ o, int nz, int ny, int nx, float c0, float c1)
{
    __shared__ float tile[BY + 2][BX + 2];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int i = blockIdx.x * BX + tx, j = blockIdx.y * BY + ty;
    int z0 = blockIdx.z * ZCH, z1 = min(z0 + ZCH, nz);
#if PERIODIC
    const int ic = wrapi(i, nx), jc = wrapi(j, ny);
    const int im = wrapi(ic - 1, nx), ip = wrapi(ic + 1, nx);
    const int jm = wrapi(jc - 1, ny), jp = wrapi(jc + 1, ny);
    const bool active = i < nx && j < ny;
    const int zb = z0 == 0 ? nz - 1 : z0 - 1;
#else
    const int ic = min(i, nx - 1), jc = min(j, ny - 1);
    const int im = max(ic - 1, 0), ip = min(ic + 1, nx - 1);
    const int jm = max(jc - 1, 0), jp = min(jc + 1, ny - 1);
    const bool active = i > 0 && i < nx - 1 && j > 0 && j < ny - 1;
    z0 = max(z0, 1);
    z1 = min(z1, nz - 1);
    const int zb = z0 - 1;
#endif
    const size_t P = (size_t)nx * ny;
    const size_t col = (size_t)jc * nx + ic;
    float below = ld(u, (size_t)zb * P + col);
    float center = ld(u, (size_t)z0 * P + col);
    int za = z0 + 1;
#if PERIODIC
    if (za >= nz) za -= nz;
#endif
    float above = (z0 < z1) ? ld(u, (size_t)za * P + col) : 0.f;
    for (int z = z0; z < z1; ++z) {
        // prefetch the plane after "above" for the next iteration
        int zn = z + 2;
#if PERIODIC
        if (zn >= nz) zn -= nz;
        const bool more = z + 1 < z1;
#else
        const bool more = z + 1 < z1;
#endif
        const float next = more ? ld(u, (size_t)zn * P + col) : 0.f;
        const size_t base = (size_t)z * P;
        __syncthreads();
        tile[ty + 1][tx + 1] = center;
        if (tx == 0) tile[ty + 1][0] = ld(u, base + (size_t)jc * nx + im);
        if (tx == BX - 1) tile[ty + 1][BX + 1] = ld(u, base + (size_t)jc * nx + ip);
        if (ty == 0) tile[0][tx + 1] = ld(u, base + (size_t)jm * nx + ic);
        if (ty == BY - 1) tile[BY + 1][tx + 1] = ld(u, base + (size_t)jp * nx + ic);
        __syncthreads();
        if (active) {
            const float s = ((tile[ty + 1][tx] + tile[ty + 1][tx + 2]) +
                             (tile[ty][tx + 1] + tile[ty + 2][tx + 1])) + (below + above);
            st(o, base + (size_t)j * nx + i, c0 * center + c1 * s);
        }
        below = center;
        center = above;
        above = next;
    }
}
#endif

// ---------------------------------------------------------------------------------------------
// Rung "tb" (temporal blocking): K time steps per DRAM pass.
//
// A block of 32 x TYT threads covers an extended xy tile of 32 x (TYT*R) columns: lane tx owns
// column x, and thread (tx, ty) owns R consecutive y rows (register blocking in y). The block
// marches along z through planes z0-K .. z1+K-1. Stage s (1..K) computes time level t+s of
// plane (iz - s) from the level-(t+s-1) register queue q[s-1] (z neighbours). x neighbours
// come from warp shuffles, y neighbours from the thread's own registers, and only the first and
// last row of each thread go through shared memory. This is a wavefront in (z, t): level t is
// read from DRAM once and level t+K written once per sweep. Halo columns are recomputed
// redundantly (overlapped tiling): the valid region shrinks by one column per stage and only
// the central (32-2K) x (TYT*R-2K) columns are stored.
// Per output site and sweep the DRAM traffic is ~ (32*TY)/((32-2K)(TY-2K)) * (ZCH+2K)/ZCH
// reads plus one write, i.e. roughly 1/K of the single-step traffic (see traffic.py).
// ---------------------------------------------------------------------------------------------
#if defined(K) && defined(TYT) && defined(R) && defined(ZCH)
#define TYR (TYT * R)
extern "C" __global__ void __launch_bounds__(32 * TYT)
tb3d(const T* __restrict__ u, T* __restrict__ o, int nz, int ny, int nx, float c0, float c1)
{
    __shared__ float edge[2][K][2][TYT][32];  // [parity][stage][first/last row][ty][tx]
    const unsigned FULL = 0xffffffffu;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int gx = blockIdx.x * (32 - 2 * K) - K + tx;
    const int gy0 = blockIdx.y * (TYR - 2 * K) - K + ty * R;
    const int z0 = blockIdx.z * ZCH;
    const int z1 = min(z0 + ZCH, nz);
#if PERIODIC
    const int lx = wrapi(gx, nx);
#else
    const int lx = clampi(gx, 0, nx - 1);
#endif
    const size_t P = (size_t)nx * ny;
    size_t col[R];
    bool fixed[R], writer[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int gy = gy0 + r;
#if PERIODIC
        col[r] = (size_t)wrapi(gy, ny) * nx + lx;
        fixed[r] = false;
#else
        col[r] = (size_t)clampi(gy, 0, ny - 1) * nx + lx;
        fixed[r] = gx <= 0 || gx >= nx - 1 || gy <= 0 || gy >= ny - 1;
#endif
        const int row = ty * R + r;
        writer[r] = tx >= K && tx < 32 - K && row >= K && row < TYR - K && gx < nx && gy < ny
                    && !fixed[r];
    }
    float q[K][3][R];
#pragma unroll
    for (int s = 0; s < K; ++s)
#pragma unroll
        for (int r = 0; r < R; ++r) { q[s][0][r] = 0.f; q[s][1][r] = 0.f; q[s][2][r] = 0.f; }

    float pref[R];
    {
#if PERIODIC
        const size_t zoff = (size_t)wrapi(z0 - K, nz) * P;
#else
        const size_t zoff = (size_t)clampi(z0 - K, 0, nz - 1) * P;
#endif
#pragma unroll
        for (int r = 0; r < R; ++r) pref[r] = ld(u, zoff + col[r]);
    }
    for (int iz = z0 - K; iz < z1 + K; ++iz) {
        const int par = iz & 1;
#pragma unroll
        for (int r = 0; r < R; ++r) { q[0][0][r] = q[0][1][r]; q[0][1][r] = q[0][2][r]; q[0][2][r] = pref[r]; }
        if (iz + 1 < z1 + K) {
#if PERIODIC
            const size_t zoff = (size_t)wrapi(iz + 1, nz) * P;
#else
            const size_t zoff = (size_t)clampi(iz + 1, 0, nz - 1) * P;
#endif
#pragma unroll
            for (int r = 0; r < R; ++r) pref[r] = ld(u, zoff + col[r]);
        }
        float res[R];
#pragma unroll
        for (int s = 1; s <= K; ++s) {
            const int c = iz - s;
            edge[par][s - 1][0][ty][tx] = q[s - 1][1][0];
            edge[par][s - 1][1][ty][tx] = q[s - 1][1][R - 1];
            __syncthreads();
            const float up = ty > 0 ? edge[par][s - 1][1][ty - 1][tx] : q[s - 1][1][0];
            const float dn = ty < TYT - 1 ? edge[par][s - 1][0][ty + 1][tx] : q[s - 1][1][R - 1];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float ctr = q[s - 1][1][r];
                const float xm = __shfl_up_sync(FULL, ctr, 1);
                const float xp = __shfl_down_sync(FULL, ctr, 1);
                const float ym = r == 0 ? up : q[s - 1][1][r - 1];
                const float yp = r == R - 1 ? dn : q[s - 1][1][r + 1];
                float v = c0 * ctr + c1 * (((xm + xp) + (ym + yp)) + (q[s - 1][0][r] + q[s - 1][2][r]));
#if !PERIODIC
                if (fixed[r] || c <= 0 || c >= nz - 1) v = ctr;
#endif
                res[r] = v;
            }
            if (s < K) {
#pragma unroll
                for (int r = 0; r < R; ++r) { q[s][0][r] = q[s][1][r]; q[s][1][r] = q[s][2][r]; q[s][2][r] = res[r]; }
            }
        }
        const int c = iz - K;
#if PERIODIC
        const bool zok = c >= z0 && c < z1;
#else
        const bool zok = c >= z0 && c < z1 && c > 0 && c < nz - 1;
#endif
        if (zok) {
#pragma unroll
            for (int r = 0; r < R; ++r)
                if (writer[r]) st(o, (size_t)c * P + (size_t)(gy0 + r) * nx + gx, res[r]);
        }
    }
}
#endif

// 2D temporal blocking, warp-synchronous: every warp owns an x-strip of 32*R columns (lane l
// holds columns l, l+32, ..., l+32(R-1), so every load is coalesced) including a K-deep halo on
// both sides, and marches ZCH rows along y with a K-stage register queue. x neighbours come
// from warp shuffles, y neighbours from the queue: no shared memory and no __syncthreads.
// Strips overlap by 2K columns (redundant halo work 32R/(32R-2K)). With K = 1 this is the 2D
// "y-marching" register-queue kernel.
#if defined(K) && defined(R) && defined(WARPS) && defined(ZCH)
#define SW (32 * R)
extern "C" __global__ void __launch_bounds__(32 * WARPS)
tb2d(const T* __restrict__ u, T* __restrict__ o, int ny, int nx, float c0, float c1)
{
    const unsigned FULL = 0xffffffffu;
    const int lane = threadIdx.x;
    const int strip = blockIdx.x * WARPS + threadIdx.y;
    const int xs = strip * (SW - 2 * K) - K;  // first (halo) column of the strip
    const int y0 = blockIdx.y * ZCH;
    const int y1 = min(y0 + ZCH, ny);
    int lx[R];
    bool fixed[R], writer[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int cl = lane + 32 * r;
        const int gx = xs + cl;
#if PERIODIC
        lx[r] = wrapi(gx, nx);
        fixed[r] = false;
#else
        lx[r] = clampi(gx, 0, nx - 1);
        fixed[r] = gx <= 0 || gx >= nx - 1;
#endif
        writer[r] = cl >= K && cl < SW - K && gx < nx && !fixed[r];
    }
    float q[K][3][R];
#pragma unroll
    for (int s = 0; s < K; ++s)
#pragma unroll
        for (int r = 0; r < R; ++r) { q[s][0][r] = 0.f; q[s][1][r] = 0.f; q[s][2][r] = 0.f; }

    float pref[R];
    {
#if PERIODIC
        const size_t yoff = (size_t)wrapi(y0 - K, ny) * nx;
#else
        const size_t yoff = (size_t)clampi(y0 - K, 0, ny - 1) * nx;
#endif
#pragma unroll
        for (int r = 0; r < R; ++r) pref[r] = ld(u, yoff + lx[r]);
    }
    for (int iy = y0 - K; iy < y1 + K; ++iy) {
#pragma unroll
        for (int r = 0; r < R; ++r) { q[0][0][r] = q[0][1][r]; q[0][1][r] = q[0][2][r]; q[0][2][r] = pref[r]; }
        if (iy + 1 < y1 + K) {
#if PERIODIC
            const size_t yoff = (size_t)wrapi(iy + 1, ny) * nx;
#else
            const size_t yoff = (size_t)clampi(iy + 1, 0, ny - 1) * nx;
#endif
#pragma unroll
            for (int r = 0; r < R; ++r) pref[r] = ld(u, yoff + lx[r]);
        }
        float res[R];
#pragma unroll
        for (int s = 1; s <= K; ++s) {
            const int c = iy - s;
            float up[R], dn[R], first[R], last[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float v = q[s - 1][1][r];
                up[r] = __shfl_up_sync(FULL, v, 1);      // lane-1, same r
                dn[r] = __shfl_down_sync(FULL, v, 1);    // lane+1, same r
                last[r] = __shfl_sync(FULL, v, 31);      // lane 31 of r (left of lane 0, r+1)
                first[r] = __shfl_sync(FULL, v, 0);      // lane 0 of r (right of lane 31, r-1)
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float ctr = q[s - 1][1][r];
                const float xm = (lane == 0 && r > 0) ? last[r > 0 ? r - 1 : 0] : up[r];
                const float xp = (lane == 31 && r < R - 1) ? first[r < R - 1 ? r + 1 : 0] : dn[r];
                float v = c0 * ctr + c1 * ((xm + xp) + (q[s - 1][0][r] + q[s - 1][2][r]));
#if !PERIODIC
                if (fixed[r] || c <= 0 || c >= ny - 1) v = ctr;
#endif
                res[r] = v;
            }
            if (s < K) {
#pragma unroll
                for (int r = 0; r < R; ++r) { q[s][0][r] = q[s][1][r]; q[s][1][r] = q[s][2][r]; q[s][2][r] = res[r]; }
            }
        }
        const int c = iy - K;
#if PERIODIC
        const bool yok = c >= y0 && c < y1;
#else
        const bool yok = c >= y0 && c < y1 && c > 0 && c < ny - 1;
#endif
        if (yok) {
#pragma unroll
            for (int r = 0; r < R; ++r)
                if (writer[r]) st(o, (size_t)c * nx + xs + lane + 32 * r, res[r]);
        }
    }
}
#endif
