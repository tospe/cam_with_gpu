// Hardware-testable stand-in for the CAM scheduling patterns (user part 3). Ordinary TMA transfers replace CAM
// requests; sizes follow the first CAM experiment (H = 64, D = 128, k = 2048):
//   query  16,640 B (64x128 FP16 + 64 FP32 weights): built in shared memory, sent with a TMA bulk STORE
//   result 16,384 B (2048 x {u32 id, u32 score})  : fetched with a TMA bulk LOAD, completion on an mbarrier
// Schedules (identical useful work per query: 128 threads prep, W-iteration FMA chain per thread, 4-warp consume):
//   A  prep -> submit -> wait -> work -> consume
//   C  prep -> submit -> work -> wait -> consume
//   E  4 producer warps (prep + submit into ring slot) / 4 consumer warps (wait, work, consume), S slots
// Consume: each consumer warp sums a quarter of the result words and atomically adds it to out[q].
//
// Usage: sched <A|C|E> <hw|trace|xferlat> <streams> <W> [slots]
//   Three kernels per run on disjoint queries: Q = 1, 32, 64 queries per stream.
//   first-result latency = T(1); steady interval = (T(64) - T(32)) / 32; total = T(64).
//   hw: 5 warm-up + 30 reps, clean cold L2 before every kernel; trace: each kernel once; xferlat: in-kernel
//   submit->result-visible latency for schedule A, W = 0 (used to choose W).
//   launch: kernel with Q = 0 queries (launch + barrier init only), hw 30 reps or trace once: timing-boundary baseline.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int QUERY_B = 16640, RESULT_B = 16384, SLOT_B = QUERY_B + RESULT_B;  // 33,024 B (16-byte multiple)
constexpr int QUERY_W = QUERY_B / 4, RESULT_W = RESULT_B / 4;
constexpr int QPER = 97;        // query slots per stream: Q=1 at [0], Q=32 at [1,33), Q=64 at [33,97)
constexpr int MAX_SLOTS = 4;
typedef unsigned long long u64;

__host__ __device__ inline unsigned hash32(unsigned x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
__host__ __device__ inline unsigned query_word(unsigned g, unsigned i) { return hash32(g * 7919u + i * 31u + 1u); }
__host__ __device__ inline unsigned result_word(unsigned g, unsigned i) { return hash32(g * 104729u + i + 7u); }

__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u64* b, unsigned n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sa(b)), "r"(n)); }
__device__ __forceinline__ void mbar_arrive(u64* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(sa(b)) : "memory"); }
__device__ __forceinline__ void mbar_expect(u64* b, unsigned tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_wait(u64* b, unsigned parity) {
  asm volatile("{\n .reg .pred p;\n W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}"
               ::"r"(sa(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_load(void* dst, const void* src, unsigned bytes, u64* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(b)) : "memory");
}
// Send the query: shared -> global bulk store; returns once the shared buffer has been read (safe to reuse).
__device__ __forceinline__ void tma_store_and_release(void* dst_global, const void* src_smem, unsigned bytes) {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst_global), "r"(sa(src_smem)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }

__device__ __forceinline__ float work_chain(float x, int w) {
  for (int i = 0; i < w; ++i) x = fmaf(x, 1.0001f, 0.25f);
  return x;
}
__host__ __device__ inline float work_seed(unsigned g, int t) { return (float)((g * 131u + (unsigned)t) % 1000u) * 0.001f; }

// 128 threads (index t) build query g in shared memory.
__device__ __forceinline__ void build_query(unsigned* q, unsigned g, int t) {
  for (int i = t; i < QUERY_W; i += 128) q[i] = query_word(g, i);
}
// Consumer warp cw (0..3) sums its quarter of the result and adds it to out[qi].
__device__ __forceinline__ void consume(const unsigned* r, int cw, int lane, unsigned* out_q) {
  unsigned s = 0;
  for (int i = cw * (RESULT_W / 4) + lane; i < (cw + 1) * (RESULT_W / 4); i += 32) s += r[i];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (lane == 0) atomicAdd(out_q, s);
}

// Schedules A (wait_first = 1) and C (wait_first = 0): 128 threads, one slot.
__global__ void sched_ac(unsigned* qbuf, const unsigned* rbuf, unsigned* out, float* workout, int qbase, int Q, int W,
                         int wait_first, u64* xfer) {
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned* qs = (unsigned*)smem; unsigned* rs = (unsigned*)(smem + QUERY_B); u64* full = (u64*)(smem + SLOT_B);
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  if (t == 0) { mbar_init(full, 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  __syncthreads();
  float acc = 0.f;
  for (int q = 0; q < Q; ++q) {
    const unsigned g = blockIdx.x * QPER + qbase + q;
    build_query(qs, g, t);
    __syncthreads();
    u64 c0 = 0;
    if (t == 0) {
      c0 = clock64();
      tma_store_and_release(qbuf + (size_t)g * QUERY_W, qs, QUERY_B);
      mbar_expect(full, RESULT_B);
      tma_load(rs, rbuf + (size_t)g * RESULT_W, RESULT_B, full);
    }
    float x = work_seed(g, t);
    if (!wait_first) x = work_chain(x, W);
    mbar_wait(full, q & 1);
    if (t == 0 && xfer) xfer[q] = clock64() - c0;
    if (wait_first) x = work_chain(x, W);
    acc += x;
    consume(rs, warp, lane, out + g);
    __syncthreads();  // all reads of rs/qs done before the next query overwrites them
  }
  workout[blockIdx.x * 128 + t] += acc;
}

// Schedule E: warps 0-3 produce, warps 4-7 consume; S ring slots, each {query, result}.
__global__ void sched_e(unsigned* qbuf, const unsigned* rbuf, unsigned* out, float* workout, int qbase, int Q, int W,
                        int S) {
  extern __shared__ __align__(128) unsigned char smem[];
  u64* full = (u64*)(smem + MAX_SLOTS * SLOT_B); u64* empty = full + MAX_SLOTS;
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  if (t == 0) {
    for (int s = 0; s < S; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], 4); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp < 4) {  // producers, p = t
    for (int q = 0; q < Q; ++q) {
      const int s = q % S; const unsigned ph = (q / S) & 1;
      const unsigned g = blockIdx.x * QPER + qbase + q;
      unsigned* qs = (unsigned*)(smem + s * SLOT_B); unsigned* rs = (unsigned*)(smem + s * SLOT_B + QUERY_B);
      named_bar(1, 128);  // previous send from this group has released its query buffer
      build_query(qs, g, t);
      named_bar(1, 128);
      if (t == 0) {
        tma_store_and_release(qbuf + (size_t)g * QUERY_W, qs, QUERY_B);
        mbar_wait(&empty[s], ph ^ 1);  // result slot free (first use passes)
        mbar_expect(&full[s], RESULT_B);
        tma_load(rs, rbuf + (size_t)g * RESULT_W, RESULT_B, &full[s]);
      }
    }
  } else {  // consumers, c = t - 128
    const int c = t - 128, cw = warp - 4;
    float acc = 0.f;
    for (int q = 0; q < Q; ++q) {
      const int s = q % S; const unsigned ph = (q / S) & 1;
      const unsigned g = blockIdx.x * QPER + qbase + q;
      const unsigned* rs = (const unsigned*)(smem + s * SLOT_B + QUERY_B);
      mbar_wait(&full[s], ph);
      acc += work_chain(work_seed(g, c), W);
      consume(rs, cw, lane, out + g);
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[s]);
    }
    workout[blockIdx.x * 128 + c] += acc;
  }
}

__global__ void flush_read(const float4* __restrict__ a, size_t n4, float* sink) {
  float acc = 0.f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
    float4 v = __ldcg(a + i); acc += v.x + v.y + v.z + v.w;
  }
  if (acc == 1234.5f) sink[0] = acc;
}

static void stats(const std::string& name, std::vector<double> v) {
  std::sort(v.begin(), v.end());
  printf("%s,n=%zu,median=%.3f,p95=%.3f,min=%.3f,max=%.3f\n", name.c_str(), v.size(), v[v.size() / 2],
         v[(size_t)(0.95 * (v.size() - 1))], v.front(), v.back());
}

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: sched <A|C|E> <hw|trace|xferlat> <streams> <W> [slots]\n"); return 2; }
  const char sched = argv[1][0];
  const std::string mode = argv[2];
  const int streams = atoi(argv[3]), W = atoi(argv[4]), S = argc > 5 ? atoi(argv[5]) : 1;
  if ((sched != 'A' && sched != 'C' && sched != 'E') || S < 1 || S > MAX_SLOTS) { fprintf(stderr, "bad args\n"); return 2; }
  const size_t nq = (size_t)streams * QPER;
  std::vector<unsigned> hr(nq * RESULT_W);
  for (size_t g = 0; g < nq; ++g)
    for (int i = 0; i < RESULT_W; ++i) hr[g * RESULT_W + i] = result_word(g, i);
  unsigned *qbuf, *rbuf, *out; float* workout; u64* xfer;
  CK(cudaMalloc(&qbuf, nq * QUERY_B)); CK(cudaMalloc(&rbuf, nq * RESULT_B)); CK(cudaMalloc(&out, nq * 4));
  CK(cudaMalloc(&workout, streams * 128 * 4)); CK(cudaMalloc(&xfer, 64 * 8));
  CK(cudaMemcpy(rbuf, hr.data(), nq * RESULT_B, cudaMemcpyHostToDevice));
  const bool is_e = sched == 'E';
  const size_t smem = is_e ? MAX_SLOTS * SLOT_B + 2 * MAX_SLOTS * 8 : SLOT_B + 8;
  CK(cudaFuncSetAttribute(sched_ac, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(SLOT_B + 8)));
  CK(cudaFuncSetAttribute(sched_e, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(MAX_SLOTS * SLOT_B + 2 * MAX_SLOTS * 8)));
  unsigned char* scratch = nullptr;
  if (mode == "hw") CK(cudaMalloc(&scratch, 256u << 20));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  const int QB[3] = {0, 1, 33}, QN[3] = {1, 32, 64};

  auto launch = [&](int k, u64* xf) {
    if (scratch) {  // clean cold L2: dirty lines are written back here, not in the measured kernel
      CK(cudaMemset(scratch, 0, 256u << 20));
      flush_read<<<912, 256>>>((const float4*)scratch, (256u << 20) / 16, (float*)out);
      CK(cudaDeviceSynchronize());
    }
    CK(cudaEventRecord(e0));
    if (is_e) sched_e<<<streams, 256, smem>>>(qbuf, rbuf, out, workout, QB[k], QN[k], W, S);
    else sched_ac<<<streams, 128, smem>>>(qbuf, rbuf, out, workout, QB[k], QN[k], W, sched == 'A', xf);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return (double)ms * 1e3;
  };
  auto reset = [&]() { CK(cudaMemset(out, 0, nq * 4)); CK(cudaMemset(workout, 0, streams * 128 * 4)); CK(cudaMemset(qbuf, 0, nq * QUERY_B)); };

  // Host reference for all three kernels together.
  auto check = [&]() {
    std::vector<unsigned> ho(nq), hq(nq * QUERY_W); std::vector<float> hw(streams * 128);
    CK(cudaMemcpy(ho.data(), out, nq * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hq.data(), qbuf, nq * QUERY_B, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hw.data(), workout, streams * 128 * 4, cudaMemcpyDeviceToHost));
    int bad_r = 0, bad_q = 0, bad_w = 0;
    for (int b = 0; b < streams; ++b) {
      std::vector<float> wref(128, 0.f);
      for (int k = 0; k < 3; ++k) {
        std::vector<float> acc(128, 0.f);
        for (int q = 0; q < QN[k]; ++q) {
          const unsigned g = b * QPER + QB[k] + q;
          unsigned s = 0;
          for (int i = 0; i < RESULT_W; ++i) s += result_word(g, i);
          bad_r += ho[g] != s;
          for (int i = 0; i < QUERY_W; ++i) bad_q += hq[(size_t)g * QUERY_W + i] != query_word(g, i);
          for (int t = 0; t < 128; ++t) { float x = work_seed(g, t); for (int i = 0; i < W; ++i) x = fmaf(x, 1.0001f, 0.25f); acc[t] += x; }
        }
        for (int t = 0; t < 128; ++t) wref[t] += acc[t];
      }
      for (int t = 0; t < 128; ++t) bad_w += hw[b * 128 + t] != wref[t];
    }
    return std::vector<int>{bad_r, bad_q, bad_w};
  };

  printf("sched=%c,mode=%s,streams=%d,W=%d,slots=%d,query_B=%d,result_B=%d,threads=%d\n", sched, mode.c_str(), streams, W,
         is_e ? S : 1, QUERY_B, RESULT_B, is_e ? 256 : 128);
  if (mode == "trace") {
    reset(); for (int k = 0; k < 3; ++k) launch(k, nullptr);
    auto c = check(); printf("trace,mismatch_results=%d,mismatch_queries=%d,mismatch_work=%d\n", c[0], c[1], c[2]);
    return (c[0] || c[1] || c[2]) ? 1 : 0;
  }
  if (mode == "launch") {
    auto run0 = [&]() {
      CK(cudaEventRecord(e0));
      if (is_e) sched_e<<<streams, 256, smem>>>(qbuf, rbuf, out, workout, 0, 0, W, S);
      else sched_ac<<<streams, 128, smem>>>(qbuf, rbuf, out, workout, 0, 0, W, sched == 'A', nullptr);
      CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return (double)ms * 1e3;
    };
    if (getenv("SCHED_TRACE_ONCE")) { run0(); printf("launch,trace,done\n"); return 0; }
    for (int r = 0; r < 5; ++r) run0();
    std::vector<double> v; for (int r = 0; r < 30; ++r) v.push_back(run0());
    stats("launch_q0_us", v); return 0;
  }
  if (mode == "xferlat") {  // schedule A, W = 0: submit (store+load issue) -> result visible, thread 0, 64 queries
    std::vector<double> v;
    for (int r = 0; r < 10; ++r) {
      reset(); launch(2, xfer);
      std::vector<u64> h(64); CK(cudaMemcpy(h.data(), xfer, 64 * 8, cudaMemcpyDeviceToHost));
      for (u64 x : h) v.push_back((double)x);
    }
    stats("xferlat_cycles", v);
    std::vector<double> first, rest;  // per-query position: first query of the kernel vs the others
    for (size_t i = 0; i < v.size(); ++i) (i % 64 == 0 ? first : rest).push_back(v[i]);
    stats("xferlat_first_query_cycles", first); stats("xferlat_later_queries_cycles", rest);
    return 0;
  }
  for (int r = 0; r < 5; ++r) { reset(); for (int k = 0; k < 3; ++k) launch(k, nullptr); }
  std::vector<double> t1, t32, t64, interval; int bad = 0;
  for (int r = 0; r < 30; ++r) {
    reset();
    double a = launch(0, nullptr), b = launch(1, nullptr), c = launch(2, nullptr);
    t1.push_back(a); t32.push_back(b); t64.push_back(c); interval.push_back((c - b) / 32.0);
    auto m = check(); bad += m[0] + m[1] + m[2];
  }
  printf("hw,total_mismatches=%d\n", bad);
  stats("first_result_us", t1); stats("t32_us", t32); stats("total_t64_us", t64); stats("steady_interval_us", interval);
  return bad ? 1 : 0;
}
