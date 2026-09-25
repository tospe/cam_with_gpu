// Calibration microbenchmarks (docs/calibration_plan.md).
//   calib lat_l2   [trace]   M1: L2-hit chase latency
//   calib lat_dram [trace|hw] [MiB]  M2: cold-L2 DRAM chase latency (default 256 MiB)
//   calib bw       [trace]   M3: cold-L2 DRAM read bandwidth
//   calib clock              M4: sustained SM clock (hardware only)
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

// Every block reads the whole list so each L2 partition has seen every line.
__global__ void warm(const unsigned* __restrict__ a, size_t n, unsigned* sink) {
  unsigned acc = 0;
  for (size_t i = threadIdx.x * 32; i < n; i += (size_t)blockDim.x * 32) acc += __ldcg(a + i);
  if (acc == 0xdeadbeef) sink[blockIdx.x] = acc;
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
static void latency(bool dram, bool trace, size_t dram_mib = 256) {
  const size_t bytes = dram ? (dram_mib << 20) : (1u << 20);
  const size_t stride_el = dram ? 64 : 32;  // 256 B or 128 B between nodes
  const size_t nodes = bytes / 4 / stride_el;
  const int H1 = 256, H2 = 1024;
  std::vector<unsigned> h = make_list(nodes, stride_el, 1);
  unsigned *next, *sink; u64* out; unsigned char* scratch = nullptr;
  CK(cudaMalloc(&next, h.size() * 4)); CK(cudaMalloc(&out, 4 * sizeof(u64))); CK(cudaMalloc(&sink, 4096 * 4));
  CK(cudaMemcpy(next, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  if (dram && !trace) CK(cudaMalloc(&scratch, 256u << 20));
  unsigned start = 0;
  auto run = [&](int hops, u64 r[4]) {
    if (scratch) CK(cudaMemset(scratch, 0, 256u << 20));
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
  printf("%s,bytes=%zu,stride_B=%zu,H1=%d,H2=%d\n", tag, bytes, stride_el * 4, H1, H2);
  stats((std::string(tag) + "_cycles_per_hop").c_str(), cyc);
  stats((std::string(tag) + "_ns_per_hop").c_str(), ns);
  std::sort(sm.begin(), sm.end()); sm.erase(std::unique(sm.begin(), sm.end()), sm.end());
  printf("%s_smids,", tag); for (int s : sm) printf("%d ", s); printf("\n");
}

static void bandwidth(bool trace) {
  const size_t B1 = 128u << 20, B2 = 256u << 20;
  float4 *a1, *a2; float* sink; unsigned char* scratch = nullptr;
  CK(cudaMalloc(&a1, B1)); CK(cudaMalloc(&a2, B2)); CK(cudaMalloc(&sink, 1 << 20));
  if (!trace) CK(cudaMalloc(&scratch, 256u << 20));
  const int grid = n_sms() * 8;
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto run = [&](float4* a, size_t bytes) {
    if (scratch) CK(cudaMemset(scratch, 0, 256u << 20));
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
  if (!strcmp(argv[1], "lat_l2")) latency(false, trace);
  else if (!strcmp(argv[1], "lat_dram")) latency(true, trace, argc > 3 ? (size_t)atoi(argv[3]) : 256);
  else if (!strcmp(argv[1], "bw")) bandwidth(trace);
  else if (!strcmp(argv[1], "clock")) clock_rate();
  else { fprintf(stderr, "unknown mode\n"); return 2; }
  return 0;
}
