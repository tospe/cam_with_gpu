// Calibration microbenchmarks (docs/calibration_plan.md).
//   calib lat_l2   [trace]   M1: L2-hit chase latency
//   calib lat_dram [trace|hw] [MiB] [stride_B] [seed]  M2: cold-L2 DRAM chase (default 256 MiB, 256 B, seed 1)
//   calib bw       [trace]   M3: cold-L2 DRAM read bandwidth
//   calib lat_hist           per-hop DRAM latency distribution, single thread, cold L2 (hardware only)
//   calib lat_lanes [trace|hw] warps/block lanes/warp  same chains, different lanes per warp (investigation)
//   calib lat_conc [trace|hw] [threads/block]  regression: 114 x threads concurrent chases (default 32), 1 GiB, cold L2
//   calib clock              M4: sustained SM clock (hardware only)
//   calib fma_lat [trace|hw] [I1 I2]    dependent FFMA chain, 1 thread: cycles per FMA (default slope 4096 vs 16384)
//   calib fma_indep [trace|hw] K W      K independent FFMA chains per thread, W warps per SM (114 blocks):
//                                       cycles per loop iteration (slope 1024 vs 4096) and FFMA/cycle/SM
// Without "trace": warm-up + repeated reps, prints median/p95 (hardware).
// With "trace": each measured kernel once, same cache preparation (for NVBit).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

typedef unsigned long long u64;

__device__ __forceinline__ u64 gtimer() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned smid() { unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }

// Single-thread dependent chase. out: cycles, ns, final index, smid.
__global__ void chase(const unsigned* __restrict__ next, unsigned start, int hops, u64* out) {
  unsigned p = start;
  u64 c0 = clock64(), g0 = gtimer();
  for (int i = 0; i < hops; ++i) p = __ldcg(next + p);
  u64 c1 = clock64(), g1 = gtimer();
  out[0] = c1 - c0; out[1] = g1 - g0; out[2] = p; out[3] = smid();
}

// One independent chase per thread; kernel time gives latency under concurrency.
__global__ void chase_many(const unsigned* __restrict__ next, const unsigned* __restrict__ starts, int hops,
                           unsigned* sink) {
  unsigned t = blockIdx.x * blockDim.x + threadIdx.x, p = starts[t];
  for (int i = 0; i < hops; ++i) p = __ldcg(next + p);
  sink[t] = p;
}

// Only the first `lanes` lanes of each warp chase; isolates the warp-max-of-lanes effect.
__global__ void chase_lanes(const unsigned* __restrict__ next, const unsigned* __restrict__ starts, int hops,
                            int lanes, unsigned* sink) {
  int lane = threadIdx.x & 31, w = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (lane >= lanes) return;
  unsigned t = w * lanes + lane, p = starts[t];
  for (int i = 0; i < hops; ++i) p = __ldcg(next + p);
  sink[t] = p;
}

// Records each hop's latency (clock64 delta). `zero` is 0 at run time but opaque to the compiler: the load
// address depends on c0 and the loaded value is consumed before c1, so the load stays inside the window.
__global__ void chase_hist(const unsigned* __restrict__ next, unsigned start, int hops, unsigned zero, unsigned* lat) {
  unsigned p = start;
  for (int i = 0; i < hops; ++i) {
    u64 c0 = clock64();
    p = __ldcg(next + (p + ((unsigned)c0 & zero)));
    asm volatile("xor.b32 %0, %0, %1;" : "+r"(p) : "r"(zero));
    u64 c1 = clock64();
    lat[i] = (unsigned)(c1 - c0) + (p & zero);
  }
}

// Every block reads the whole list so each L2 partition has seen every line.
__global__ void warm(const unsigned* __restrict__ a, size_t n, unsigned* sink) {
  unsigned acc = 0;
  for (size_t i = threadIdx.x * 32; i < n; i += (size_t)blockDim.x * 32) acc += __ldcg(a + i);
  if (acc == 0xdeadbeef) sink[blockIdx.x] = acc;
}

// Clean flush: stream-read a 256 MiB buffer so L2 ends up holding clean lines (dirty lines, e.g. from a memset,
// are written back during the flush, not during the measured kernel).
__global__ void flush_read(const float4* __restrict__ a, size_t n4, float* sink) {
  float acc = 0.f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
    float4 v = __ldcg(a + i); acc += v.x + v.y + v.z + v.w;
  }
  if (acc == 1234.5f) sink[blockIdx.x] = acc;
}
static int g_flush_mode = 0;  // 0 = memset (dirty L2, original v1 procedure), 1 = clean read flush
static void flush_l2(unsigned char* scratch, float* sink) {
  CK(cudaMemset(scratch, 0, 256u << 20));
  if (g_flush_mode == 1) { flush_read<<<912, 256>>>((const float4*)scratch, (256u << 20) / 16, sink); }
  CK(cudaDeviceSynchronize());
}

__global__ void read_bw(const float4* __restrict__ a, size_t n4, float* sink) {
  float acc = 0.f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
    float4 v = __ldcg(a + i); acc += v.x + v.y + v.z + v.w;
  }
  if (acc == 1234.5f) sink[blockIdx.x] = acc;
}

__global__ void spin(int iters, u64* out) {
  float x = threadIdx.x;
  u64 c0 = clock64(), g0 = gtimer();
  for (int i = 0; i < iters; ++i) x = fmaf(x, 1.000001f, 0.5f);
  u64 c1 = clock64(), g1 = gtimer();
  if (threadIdx.x == 0) { out[2 * blockIdx.x] = c1 - c0; out[2 * blockIdx.x + 1] = g1 - g0; }
  if (x == 0.f) out[0] = 0;
}

// Random single-cycle permutation (Sattolo), nodes spaced stride_el elements apart.
static std::vector<unsigned> make_list(size_t nodes, size_t stride_el, unsigned seed) {
  std::vector<unsigned> perm(nodes);
  for (size_t i = 0; i < nodes; ++i) perm[i] = i;
  std::mt19937_64 rng(seed);
  for (size_t i = nodes - 1; i > 0; --i) std::swap(perm[i], perm[std::uniform_int_distribution<size_t>(0, i - 1)(rng)]);
  std::vector<unsigned> next(nodes * stride_el, 0);
  for (size_t i = 0; i < nodes; ++i) next[perm[i] * stride_el] = (unsigned)(perm[(i + 1) % nodes] * stride_el);
  return next;
}

static void stats(const char* name, std::vector<double> v) {
  std::sort(v.begin(), v.end());
  printf("%s,n=%zu,median=%.3f,p95=%.3f,min=%.3f,max=%.3f\n", name, v.size(), v[v.size() / 2],
         v[(size_t)(0.95 * (v.size() - 1))], v.front(), v.back());
}

static int n_sms() { int s; CK(cudaDeviceGetAttribute(&s, cudaDevAttrMultiProcessorCount, 0)); return s; }

// Latency: chase at H1 and H2 hops; slope = cycles/hop. For DRAM, flush L2 before each kernel
// and start each kernel at a fresh node so the chase is cold.
static void latency(bool dram, bool trace, size_t dram_mib = 256, size_t dram_stride_b = 256, unsigned seed = 1) {
  const size_t bytes = dram ? (dram_mib << 20) : (1u << 20);
  const size_t stride_el = dram ? dram_stride_b / 4 : 32;  // DRAM: default 256 B; L2: 128 B
  const size_t nodes = bytes / 4 / stride_el;
  const int H1 = 256, H2 = 1024;
  std::vector<unsigned> h = make_list(nodes, stride_el, seed);
  unsigned *next, *sink; u64* out; unsigned char* scratch = nullptr;
  CK(cudaMalloc(&next, h.size() * 4)); CK(cudaMalloc(&out, 4 * sizeof(u64))); CK(cudaMalloc(&sink, 4096 * 4));
  CK(cudaMemcpy(next, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  if (dram && !trace) CK(cudaMalloc(&scratch, 256u << 20));
  unsigned start = 0;
  auto run = [&](int hops, u64 r[4]) {
    if (scratch) flush_l2(scratch, (float*)sink);
    if (!dram) warm<<<n_sms(), 256>>>(next, h.size(), sink);
    chase<<<1, 1>>>(next, start, hops, out);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    CK(cudaMemcpy(r, out, 4 * sizeof(u64), cudaMemcpyDeviceToHost));
    if (dram) start = (unsigned)r[2];  // continue the cycle: never revisit recent nodes
  };
  u64 r1[4], r2[4];
  const char* tag = dram ? "lat_dram" : "lat_l2";
  if (trace) { run(H1, r1); run(H2, r2); printf("%s,trace,done\n", tag); return; }
  for (int i = 0; i < 5; ++i) run(H1, r1);
  std::vector<double> cyc, ns; std::vector<int> sm;
  for (int i = 0; i < 30; ++i) {
    run(H1, r1); run(H2, r2);
    cyc.push_back(double(r2[0] - r1[0]) / (H2 - H1));
    ns.push_back(double(r2[1] - r1[1]) / (H2 - H1));
    sm.push_back((int)r1[3]); sm.push_back((int)r2[3]);
  }
  printf("%s,bytes=%zu,stride_B=%zu,seed=%u,H1=%d,H2=%d\n", tag, bytes, stride_el * 4, seed, H1, H2);
  stats((std::string(tag) + "_cycles_per_hop").c_str(), cyc);
  stats((std::string(tag) + "_ns_per_hop").c_str(), ns);
  std::sort(sm.begin(), sm.end()); sm.erase(std::unique(sm.begin(), sm.end()), sm.end());
  printf("%s_smids,", tag); for (int s : sm) printf("%d ", s); printf("\n");
}

// Chains start evenly spaced along the single random cycle, so they never overlap.
static void lat_conc(bool trace, int threads = 32, int warps_lanes = 0, int lanes = 0) {
  const size_t bytes = 1024u << 20, stride_el = 64, nodes = bytes / 4 / stride_el;
  const int blocks = n_sms(), T = warps_lanes ? blocks * warps_lanes * lanes : blocks * threads, H1 = 128, H2 = 512;
  std::vector<unsigned> h = make_list(nodes, stride_el, 1);
  std::vector<unsigned> starts(T);
  unsigned p = 0;  // walk the cycle once on the host to place starts
  for (size_t i = 0, t = 0; t < (size_t)T; ++i) {
    if (i % (nodes / T) == 0) starts[t++] = p;
    p = h[p];
  }
  unsigned *next, *st, *sink; unsigned char* scratch = nullptr;
  CK(cudaMalloc(&next, h.size() * 4)); CK(cudaMalloc(&st, T * 4)); CK(cudaMalloc(&sink, T * 4));
  CK(cudaMemcpy(next, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(st, starts.data(), T * 4, cudaMemcpyHostToDevice));
  if (!trace) CK(cudaMalloc(&scratch, 256u << 20));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto run = [&](int hops) {
    if (scratch) flush_l2(scratch, (float*)sink);
    CK(cudaEventRecord(e0));
    if (warps_lanes) chase_lanes<<<blocks, 32 * warps_lanes>>>(next, st, hops, lanes, sink);
    else chase_many<<<blocks, threads>>>(next, st, hops, sink);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return (double)ms;
  };
  // Trace mode has no L2 flush between kernels, so the H2 kernel must not revisit the H1 kernel's lines:
  // continue every chain from where H1 stopped (spacing nodes/T >= H1 + H2 keeps chains disjoint).
  if (trace) {
    if (nodes / T < (size_t)(H1 + H2)) { fprintf(stderr, "chains too dense for continuation\n"); exit(1); }
    run(H1); CK(cudaMemcpy(st, sink, T * 4, cudaMemcpyDeviceToDevice)); run(H2);
    printf("lat_conc,trace,continued,done\n"); return;
  }
  for (int i = 0; i < 5; ++i) { run(H1); run(H2); }
  std::vector<double> cyc, el, lps;
  for (int i = 0; i < 30; ++i) {
    double a = run(H1), b = run(H2);
    cyc.push_back((b - a) * 1e-3 * 1755e6 / (H2 - H1));
    el.push_back(b * 1e3);                                   // H2 kernel elapsed, us
    lps.push_back((double)T * (H2 - H1) / ((b - a) * 1e-3));  // completed loads/s (slope)
  }
  printf("lat_conc,chains=%d,warps_per_block=%d,lanes_per_warp=%d,bytes=%zu,H1=%d,H2=%d,cycles_at_1755MHz_from_events\n", T, warps_lanes ? warps_lanes : threads / 32, warps_lanes ? lanes : threads, bytes, H1, H2);
  stats("lat_conc_cycles_per_hop", cyc);
  stats("lat_conc_elapsed_H2_us", el);
  stats("lat_conc_loads_per_s", lps);
}

static void bandwidth(bool trace) {
  const size_t B1 = 128u << 20, B2 = 256u << 20;
  float4 *a1, *a2; float* sink; unsigned char* scratch = nullptr;
  CK(cudaMalloc(&a1, B1)); CK(cudaMalloc(&a2, B2)); CK(cudaMalloc(&sink, 1 << 20));
  if (!trace) CK(cudaMalloc(&scratch, 256u << 20));
  const int grid = n_sms() * 8;
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto run = [&](float4* a, size_t bytes) {
    if (scratch) flush_l2(scratch, sink);
    CK(cudaEventRecord(e0));
    read_bw<<<grid, 256>>>(a, bytes / 16, sink);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return (double)ms;
  };
  if (trace) { run(a1, B1); run(a2, B2); printf("bw,trace,done\n"); return; }
  for (int i = 0; i < 5; ++i) { run(a1, B1); run(a2, B2); }
  std::vector<double> t1, t2, bw_slope, bw_256;
  for (int i = 0; i < 20; ++i) {
    double x1 = run(a1, B1), x2 = run(a2, B2);
    t1.push_back(x1 * 1e3); t2.push_back(x2 * 1e3);
    bw_slope.push_back((B2 - B1) / ((x2 - x1) * 1e-3) / 1e9);
    bw_256.push_back(B2 / (x2 * 1e-3) / 1e9);
  }
  printf("bw,grid=%d,block=256,B1=%zu,B2=%zu\n", grid, B1, B2);
  stats("bw_t128MiB_us", t1); stats("bw_t256MiB_us", t2);
  stats("bw_slope_GBps", bw_slope); stats("bw_256MiB_GBps", bw_256);
}

// Same busy loop shape as pipeline.cu busy(): dependent fmaf chain.
__global__ void fma_chain(int iters, float seed, float* out, u64* cyc) {
  float x = seed;
  u64 c0 = clock64();
  for (int i = 0; i < iters; ++i) x = fmaf(x, 1.0001f, 0.25f);
  u64 c1 = clock64();
  out[0] = x; cyc[0] = c1 - c0;
}

static void fma_lat(bool trace, int I1 = 4096, int I2 = 16384) {
  float* out; u64* cyc; CK(cudaMalloc(&out, 4)); CK(cudaMalloc(&cyc, 8));
  auto run = [&](int it) { fma_chain<<<1, 1>>>(it, 1.f, out, cyc); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
                           u64 c; CK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost)); return (double)c; };
  if (trace) { run(I1); run(I2); printf("fma_lat,trace,done\n"); return; }
  for (int i = 0; i < 5; ++i) run(I1);
  std::vector<double> v;
  for (int i = 0; i < 30; ++i) { double a = run(I1), b = run(I2); v.push_back((b - a) / (I2 - I1)); }
  printf("fma_lat,I1=%d,I2=%d\n", I1, I2);
  stats("fma_lat_cycles_per_iter", v);
}

// K independent dependent-chains per thread; each loop iteration issues K FFMAs.
template <int K>
__global__ void fma_indep_k(int iters, float* out) {
  float x[K];
#pragma unroll
  for (int k = 0; k < K; ++k) x[k] = threadIdx.x + k;
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < K; ++k) x[k] = fmaf(x[k], 1.0001f, 0.25f);
  }
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) s += x[k];
  if (s == -1.f) out[blockIdx.x] = s;
}

static void fma_indep(bool trace, int K, int W) {
  const int I1 = 1024, I2 = 4096, blocks = n_sms(), threads = 32 * W;
  float* out; CK(cudaMalloc(&out, blocks * 4));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto run = [&](int it) {
    CK(cudaEventRecord(e0));
    if (K == 2) fma_indep_k<2><<<blocks, threads>>>(it, out); else fma_indep_k<8><<<blocks, threads>>>(it, out);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return (double)ms;
  };
  if (K != 2 && K != 8) { fprintf(stderr, "K must be 2 or 8\n"); exit(2); }
  if (trace) { run(I1); run(I2); printf("fma_indep,trace,K=%d,W=%d,done\n", K, W); return; }
  for (int i = 0; i < 5; ++i) { run(I1); run(I2); }
  std::vector<double> cyc, thr;
  for (int i = 0; i < 30; ++i) {
    double a = run(I1), b = run(I2), c = (b - a) * 1e-3 * 1755e6 / (I2 - I1);
    cyc.push_back(c); thr.push_back(32.0 * W * K / c);
  }
  printf("fma_indep,K=%d,W=%d,blocks=%d,threads=%d,I1=%d,I2=%d,cycles_at_1755MHz_from_events\n", K, W, blocks, threads, I1, I2);
  stats("fma_indep_cycles_per_iter", cyc);
  stats("fma_indep_ffma_per_cycle_per_sm", thr);
}

static void lat_hist() {
  const size_t nodes = (256u << 20) / 256, stride_el = 64; const int hops = 4096;
  std::vector<unsigned> h = make_list(nodes, stride_el, 1);
  unsigned *next, *lat; unsigned char* scratch;
  CK(cudaMalloc(&next, h.size() * 4)); CK(cudaMalloc(&lat, hops * 4)); CK(cudaMalloc(&scratch, 256u << 20));
  CK(cudaMemcpy(next, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  std::vector<double> all; std::vector<unsigned> hl(hops); unsigned start = 0;
  for (int r = 0; r < 10; ++r) {
    flush_l2(scratch, (float*)lat);
    chase_hist<<<1, 1>>>(next, start, hops, 0u, lat); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    CK(cudaMemcpy(hl.data(), lat, hops * 4, cudaMemcpyDeviceToHost));
    for (unsigned v : hl) all.push_back(v);
    start = h[start] ;  // different start each rep (next node)
    for (int k = 0; k < hops * 3; ++k) start = h[start];
  }
  std::sort(all.begin(), all.end());
  auto q = [&](double f) { return all[(size_t)(f * (all.size() - 1))]; };
  double mean = 0; for (double v : all) mean += v; mean /= all.size();
  printf("lat_hist,samples=%zu,mean=%.1f,p1=%.0f,p10=%.0f,p50=%.0f,p90=%.0f,p99=%.0f,max=%.0f\n", all.size(), mean,
         q(0.01), q(0.1), q(0.5), q(0.9), q(0.99), all.back());
  // expected max over N lanes, from the empirical distribution (independent draws approximated by quantile 1-1/(N+1))
  for (int n : {1, 4, 8, 32}) printf("lat_hist,approx_E_max_of_%d=%.0f\n", n, q(1.0 - 1.0 / (n + 1)));
}

static void clock_rate() {
  u64* out; const int maxb = n_sms() * 4;
  CK(cudaMalloc(&out, 2 * maxb * sizeof(u64)));
  for (int full = 0; full < 2; ++full) {
    int blocks = full ? maxb : 1, threads = full ? 256 : 1;
    spin<<<blocks, threads>>>(1 << 20, out); CK(cudaDeviceSynchronize());  // warm-up
    spin<<<blocks, threads>>>(full ? (1 << 26) : (1 << 28), out); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    std::vector<u64> h(2 * blocks); CK(cudaMemcpy(h.data(), out, h.size() * sizeof(u64), cudaMemcpyDeviceToHost));
    std::vector<double> mhz;
    for (int b = 0; b < blocks; ++b) mhz.push_back(1e3 * h[2 * b] / (double)h[2 * b + 1]);
    stats(full ? "clock_full_load_MHz" : "clock_light_load_MHz", mhz);
  }
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: calib lat_l2|lat_dram|bw|clock [trace]\n"); return 2; }
  bool trace = argc > 2 && !strcmp(argv[2], "trace");
  if (getenv("CLEAN_FLUSH")) g_flush_mode = atoi(getenv("CLEAN_FLUSH"));
  fprintf(stderr, "flush_mode=%d\n", g_flush_mode);
  if (!strcmp(argv[1], "lat_l2")) latency(false, trace);
  else if (!strcmp(argv[1], "lat_dram"))
    latency(true, trace, argc > 3 ? (size_t)atoi(argv[3]) : 256, argc > 4 ? (size_t)atoi(argv[4]) : 256,
            argc > 5 ? (unsigned)atoi(argv[5]) : 1);
  else if (!strcmp(argv[1], "bw")) bandwidth(trace);
  else if (!strcmp(argv[1], "lat_conc")) lat_conc(trace, argc > 3 ? atoi(argv[3]) : 32);
  else if (!strcmp(argv[1], "lat_lanes")) lat_conc(trace, 0, atoi(argv[3]), atoi(argv[4]));
  else if (!strcmp(argv[1], "clock")) clock_rate();
  else if (!strcmp(argv[1], "fma_lat")) fma_lat(trace, argc > 4 ? atoi(argv[3]) : 4096, argc > 4 ? atoi(argv[4]) : 16384);
  else if (!strcmp(argv[1], "fma_indep")) fma_indep(trace, atoi(argv[3]), atoi(argv[4]));
  else if (!strcmp(argv[1], "lat_hist")) lat_hist();
  else { fprintf(stderr, "unknown mode\n"); return 2; }
  return 0;
}
