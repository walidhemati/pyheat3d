// pyheat3d hand-written CUDA C reference kernels, compiled at runtime with NVRTC (CuPy).
//
// Update (all kernels, canonical summation order):
//   u'[c] = c0*u[c] + c1*( ((u[x-1]+u[x+1]) + (u[y-1]+u[y+1])) + (u[z-1]+u[z+1]) )
// Arrays are C-ordered (z, y, x) / (y, x); x is contiguous.
//
// Compile-time macros (set by the Python side):
//   STORAGE   0 = fp32, 1 = fp16, 2 = bf16 storage (arithmetic is always fp32)
//   PERIODIC  0 = Dirichlet (boundary layer fixed, never written), 1 = periodic
//   BX, BY    thread-block shape of the naive / z-marching kernels
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
extern "C" __global__ void naive2d(const T* __restrict__ u, T* __restrict__ o,
                                   int ny, int nx, float c0, float c1)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
#if PERIODIC
    const int im = i == 0 ? nx - 1 : i - 1, ip = i == nx - 1 ? 0 : i + 1;
    const int jm = j == 0 ? ny - 1 : j - 1, jp = j == ny - 1 ? 0 : j + 1;
#else
    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) return;
    const int im = i - 1, ip = i + 1, jm = j - 1, jp = j + 1;
#endif
    const size_t r = (size_t)j * nx;
    const float s = (ld(u, r + im) + ld(u, r + ip)) + (ld(u, (size_t)jm * nx + i) + ld(u, (size_t)jp * nx + i));
    st(o, r + i, c0 * ld(u, r + i) + c1 * s);
}

extern "C" __global__ void naive3d(const T* __restrict__ u, T* __restrict__ o,
                                   int nz, int ny, int nx, float c0, float c1)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
#if PERIODIC
    const int im = i == 0 ? nx - 1 : i - 1, ip = i == nx - 1 ? 0 : i + 1;
    const int jm = j == 0 ? ny - 1 : j - 1, jp = j == ny - 1 ? 0 : j + 1;
    const int km = k == 0 ? nz - 1 : k - 1, kp = k == nz - 1 ? 0 : k + 1;
#else
    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1) return;
    const int im = i - 1, ip = i + 1, jm = j - 1, jp = j + 1, km = k - 1, kp = k + 1;
#endif
    const size_t P = (size_t)nx * ny;
    const size_t b = (size_t)k * P + (size_t)j * nx;
    const float sx = ld(u, b + im) + ld(u, b + ip);
    const float sy = ld(u, (size_t)k * P + (size_t)jm * nx + i) + ld(u, (size_t)k * P + (size_t)jp * nx + i);
    const float sz = ld(u, (size_t)km * P + (size_t)j * nx + i) + ld(u, (size_t)kp * P + (size_t)j * nx + i);
    st(o, b + i, c0 * ld(u, b + i) + c1 * ((sx + sy) + sz));
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
// A block of TX*TY threads covers an extended xy tile; thread (tx, ty) owns one column. The
// block marches along z through planes z0-K .. z1+K-1. Stage s (1..K) computes time level
// t+s of plane (iz - s) from the level-(t+s-1) register queue q[s-1] (z neighbours) and the
// shared-memory copy of that queue's center plane (x/y neighbours). This is a wavefront in
// (z, t): level t is read from DRAM once and level t+K written once per sweep. Halo columns
// are recomputed redundantly (overlapped tiling), the valid region shrinks by one column per
// stage, and only the central (TX-2K) x (TY-2K) columns are stored.
// Per sweep and output site the DRAM traffic is ~ (TX*TY)/((TX-2K)(TY-2K)) * (ZCH+2K)/ZCH
// reads plus one write, i.e. about 1/K of the single-step traffic (see traffic.py).
// ---------------------------------------------------------------------------------------------
#if defined(K) && defined(TX) && defined(TY) && defined(ZCH)
extern "C" __global__ void __launch_bounds__(TX * TY)
tb3d(const T* __restrict__ u, T* __restrict__ o, int nz, int ny, int nx, float c0, float c1)
{
    __shared__ float sm[2][K][TY][TX];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int gx = blockIdx.x * (TX - 2 * K) - K + tx;
    const int gy = blockIdx.y * (TY - 2 * K) - K + ty;
    const int z0 = blockIdx.z * ZCH;
    const int z1 = min(z0 + ZCH, nz);
#if PERIODIC
    const int lx = wrapi(gx, nx), ly = wrapi(gy, ny);
    const bool fixed_xy = false;
#else
    const int lx = clampi(gx, 0, nx - 1), ly = clampi(gy, 0, ny - 1);
    const bool fixed_xy = gx <= 0 || gx >= nx - 1 || gy <= 0 || gy >= ny - 1;
#endif
    const size_t P = (size_t)nx * ny;
    const size_t col = (size_t)ly * nx + lx;
    const int txm = max(tx - 1, 0), txp = min(tx + 1, TX - 1);
    const int tym = max(ty - 1, 0), typ = min(ty + 1, TY - 1);
    const bool writer = tx >= K && tx < TX - K && ty >= K && ty < TY - K && gx < nx && gy < ny
                        && !fixed_xy;
    float q[K][3];
#pragma unroll
    for (int s = 0; s < K; ++s) { q[s][0] = 0.f; q[s][1] = 0.f; q[s][2] = 0.f; }

#if PERIODIC
    float pref = ld(u, (size_t)wrapi(z0 - K, nz) * P + col);
#else
    float pref = ld(u, (size_t)clampi(z0 - K, 0, nz - 1) * P + col);
#endif
    for (int iz = z0 - K; iz < z1 + K; ++iz) {
        const int par = iz & 1;
        q[0][0] = q[0][1]; q[0][1] = q[0][2]; q[0][2] = pref;
        if (iz + 1 < z1 + K) {
#if PERIODIC
            pref = ld(u, (size_t)wrapi(iz + 1, nz) * P + col);
#else
            pref = ld(u, (size_t)clampi(iz + 1, 0, nz - 1) * P + col);
#endif
        }
        float r = 0.f;
#pragma unroll
        for (int s = 1; s <= K; ++s) {
            const int c = iz - s;
            const float ctr = q[s - 1][1];
            sm[par][s - 1][ty][tx] = ctr;
            __syncthreads();
            const float sxy = (sm[par][s - 1][ty][txm] + sm[par][s - 1][ty][txp]) +
                              (sm[par][s - 1][tym][tx] + sm[par][s - 1][typ][tx]);
            r = c0 * ctr + c1 * (sxy + (q[s - 1][0] + q[s - 1][2]));
#if !PERIODIC
            if (fixed_xy || c <= 0 || c >= nz - 1) r = ctr;
#endif
            if (s < K) { q[s][0] = q[s][1]; q[s][1] = q[s][2]; q[s][2] = r; }
        }
        const int c = iz - K;
#if PERIODIC
        const bool zok = c >= z0 && c < z1;
#else
        const bool zok = c >= z0 && c < z1 && c > 0 && c < nz - 1;
#endif
        if (writer && zok) st(o, (size_t)c * P + (size_t)gy * nx + gx, r);
    }
}
#endif

// 2D analogue: a 1D block of TX threads covers an x-strip (with K-deep halo) and marches ZCH
// rows along y. With K = 1 this is the 2D "y-marching" (register-queue) kernel.
#if defined(K) && defined(TX) && defined(ZCH)
extern "C" __global__ void __launch_bounds__(TX)
tb2d(const T* __restrict__ u, T* __restrict__ o, int ny, int nx, float c0, float c1)
{
    __shared__ float sm[2][K][TX];
    const int tx = threadIdx.x;
    const int gx = blockIdx.x * (TX - 2 * K) - K + tx;
    const int y0 = blockIdx.y * ZCH;
    const int y1 = min(y0 + ZCH, ny);
#if PERIODIC
    const int lx = wrapi(gx, nx);
    const bool fixed_x = false;
#else
    const int lx = clampi(gx, 0, nx - 1);
    const bool fixed_x = gx <= 0 || gx >= nx - 1;
#endif
    const int txm = max(tx - 1, 0), txp = min(tx + 1, TX - 1);
    const bool writer = tx >= K && tx < TX - K && gx < nx && !fixed_x;
    float q[K][3];
#pragma unroll
    for (int s = 0; s < K; ++s) { q[s][0] = 0.f; q[s][1] = 0.f; q[s][2] = 0.f; }

#if PERIODIC
    float pref = ld(u, (size_t)wrapi(y0 - K, ny) * nx + lx);
#else
    float pref = ld(u, (size_t)clampi(y0 - K, 0, ny - 1) * nx + lx);
#endif
    for (int iy = y0 - K; iy < y1 + K; ++iy) {
        const int par = iy & 1;
        q[0][0] = q[0][1]; q[0][1] = q[0][2]; q[0][2] = pref;
        if (iy + 1 < y1 + K) {
#if PERIODIC
            pref = ld(u, (size_t)wrapi(iy + 1, ny) * nx + lx);
#else
            pref = ld(u, (size_t)clampi(iy + 1, 0, ny - 1) * nx + lx);
#endif
        }
        float r = 0.f;
#pragma unroll
        for (int s = 1; s <= K; ++s) {
            const int c = iy - s;
            const float ctr = q[s - 1][1];
            sm[par][s - 1][tx] = ctr;
            __syncthreads();
            const float sx = sm[par][s - 1][txm] + sm[par][s - 1][txp];
            r = c0 * ctr + c1 * (sx + (q[s - 1][0] + q[s - 1][2]));
#if !PERIODIC
            if (fixed_x || c <= 0 || c >= ny - 1) r = ctr;
#endif
            if (s < K) { q[s][0] = q[s][1]; q[s][1] = q[s][2]; q[s][2] = r; }
        }
        const int c = iy - K;
#if PERIODIC
        const bool yok = c >= y0 && c < y1;
#else
        const bool yok = c >= y0 && c < y1 && c > 0 && c < ny - 1;
#endif
        if (writer && yok) st(o, (size_t)c * nx + gx, r);
    }
}
#endif
