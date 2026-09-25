// Validation kernels for guide §5 (required before the CAM port):
//   pipeline cpasync [hw|hwcold|trace] [work]   async copy (cp.async) + consumer wait + cross-thread visibility
//   (hwcold: before every launch, memset a 256 MiB buffer then stream-read it, leaving L2 cold for the input and
//    holding only clean lines; compare with sim -gpgpu_perf_sim_memcpy 0)
//   pipeline tma     [hw|trace] [slots] [work]  warp-specialized TMA bulk ring: full/empty mbarriers with
//                                                repeated phases, buffer reuse, producer/consumer progress
// Every tile holds distinct data and the host recomputes the exact checksum, so early buffer reuse, a stale
// barrier phase, or a consumer running ahead of its data changes the result.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int TILE_F = 1024;           // floats per tile (4 KiB)
constexpr int TILE_B = TILE_F * 4;
constexpr int CONSUMERS = 4;           // consumer warps in the TMA kernel
constexpr int MAX_SLOTS = 8;

__global__ void flush_read(const float4* __restrict__ a, size_t n4, float* sink) {
  float acc = 0.f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
    float4 v = __ldcg(a + i); acc += v.x + v.y + v.z + v.w;
  }
  if (acc == 1234.5f) sink[0] = acc;
}

__device__ __forceinline__ unsigned smem_addr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

// Independent arithmetic that does not touch the tile; `work` iterations.
__device__ __forceinline__ float busy(float x, int work) {
  for (int i = 0; i < work; ++i) x = fmaf(x, 1.0001f, 0.25f);
  return x;
}

// ---------------------------------------------------------------- cp.async
// Each of 256 threads copies 4 floats (16 B) of a tile, then works, then waits and syncs, then reads the
// element copied by a *different* thread (visibility across the block), then syncs before the buffer is reused.
__global__ void cpasync_kernel(const float* __restrict__ in, int tiles, int work, float* out) {
  __shared__ __align__(16) float buf[TILE_F];
  const int tid = threadIdx.x;
  const float* base = in + (size_t)blockIdx.x * tiles * TILE_F;
  float acc = 0.f, side = tid;
  for (int t = 0; t < tiles; ++t) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_addr(buf + 4 * tid)),
                 "l"(base + (size_t)t * TILE_F + 4 * tid));
    asm volatile("cp.async.commit_group;");
    side = busy(side, work);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    const int o = 4 * ((tid + 37) % 256);  // another thread's chunk
    acc += buf[o] + buf[o + 1] + buf[o + 2] + buf[o + 3];
    __syncthreads();  // all reads done before the next copy overwrites buf
  }
  out[blockIdx.x * blockDim.x + tid] = acc + (side == -1.f ? 1.f : 0.f);
}

// ---------------------------------------------------------------- TMA ring
__device__ __forceinline__ void mbar_init(unsigned long long* b, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(b)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(unsigned long long* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_addr(b)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_tx(unsigned long long* b, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_addr(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long* b, unsigned parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}" ::"r"(smem_addr(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_load(void* dst, const void* src, unsigned bytes, unsigned long long* b) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smem_addr(dst)),
      "l"(src), "r"(bytes), "r"(smem_addr(b)) : "memory");
}

// Warp 0 = producer (lane 0 issues), warps 1..CONSUMERS = consumers. Slot s holds tile t when t % slots == s;
// its full/empty barriers complete once per use, so phase parity for tile t is (t / slots) & 1.
__global__ void tma_ring_kernel(const float* __restrict__ in, int tiles, int slots, int work, float* out) {
  extern __shared__ __align__(128) unsigned char smem[];
  float* ring = (float*)smem;
  unsigned long long* full = (unsigned long long*)(smem + MAX_SLOTS * TILE_B);
  unsigned long long* empty = full + MAX_SLOTS;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const float* base = in + (size_t)blockIdx.x * tiles * TILE_F;
  if (threadIdx.x == 0) {
    for (int s = 0; s < slots; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], CONSUMERS); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 0) {
    if (lane == 0) {
      for (int t = 0; t < tiles; ++t) {
        const int s = t % slots; const unsigned ph = (t / slots) & 1;
        mbar_wait(&empty[s], ph ^ 1);  // first use of each slot passes immediately
        mbar_arrive_tx(&full[s], TILE_B);
        tma_load(ring + s * TILE_F, base + (size_t)t * TILE_F, TILE_B, &full[s]);
      }
    }
  } else {
    const int c = warp - 1;  // consumer c sums its quarter of every tile
    float acc = 0.f, side = threadIdx.x;
    for (int t = 0; t < tiles; ++t) {
      const int s = t % slots; const unsigned ph = (t / slots) & 1;
      mbar_wait(&full[s], ph);
      const float* tile = ring + s * TILE_F + c * (TILE_F / CONSUMERS);
      for (int i = lane; i < TILE_F / CONSUMERS; i += 32) acc += tile[i];
      side = busy(side, work);
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[s]);
    }
    out[blockIdx.x * (CONSUMERS * 32) + c * 32 + lane] = acc + (side == -1.f ? 1.f : 0.f);
  }
}

// ---------------------------------------------------------------- host
static float value(size_t i) { return (float)((i * 2654435761u) % 1021) * 0.5f; }  // exact in float sums below

static int n_sms() { int s; CK(cudaDeviceGetAttribute(&s, cudaDevAttrMultiProcessorCount, 0)); return s; }

static void report(const char* name, std::vector<double> v) {
  std::sort(v.begin(), v.end());
  printf("%s,n=%zu,median=%.3f,p95=%.3f,min=%.3f,max=%.3f\n", name, v.size(), v[v.size() / 2],
         v[(size_t)(0.95 * (v.size() - 1))], v.front(), v.back());
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: pipeline cpasync|tma [hw|trace] [slots] [work]\n"); return 2; }
  const bool tma = !strcmp(argv[1], "tma");
  const bool trace = argc > 2 && !strcmp(argv[2], "trace");
  const bool cold = argc > 2 && !strcmp(argv[2], "hwcold");
  const int slots = tma ? (argc > 3 ? atoi(argv[3]) : 2) : 0;
  const int work = argc > (tma ? 4 : 3) ? atoi(argv[tma ? 4 : 3]) : 0;
  if (tma && (slots < 1 || slots > MAX_SLOTS)) { fprintf(stderr, "slots must be 1..%d\n", MAX_SLOTS); return 2; }
  const int blocks = n_sms(), tiles = 64;
  const size_t nf = (size_t)blocks * tiles * TILE_F;
  std::vector<float> h(nf);
  for (size_t i = 0; i < nf; ++i) h[i] = value(i);
  const int threads = tma ? 32 * (1 + CONSUMERS) : 256;
  const int outs = tma ? blocks * CONSUMERS * 32 : blocks * 256;
  float *in, *out; CK(cudaMalloc(&in, nf * 4)); CK(cudaMalloc(&out, outs * 4));
  CK(cudaMemcpy(in, h.data(), nf * 4, cudaMemcpyHostToDevice));
  const size_t smem = tma ? MAX_SLOTS * TILE_B + 2 * MAX_SLOTS * 8 : 0;
  if (tma) CK(cudaFuncSetAttribute(tma_ring_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

  unsigned char* scratch = nullptr;
  if (cold) CK(cudaMalloc(&scratch, 256u << 20));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto launch = [&]() {
    if (scratch) {
      CK(cudaMemset(scratch, 0, 256u << 20));
      flush_read<<<912, 256>>>((const float4*)scratch, (256u << 20) / 16, out);
      CK(cudaDeviceSynchronize());
    }
    CK(cudaEventRecord(e0));
    if (tma) tma_ring_kernel<<<blocks, threads, smem>>>(in, tiles, slots, work, out);
    else cpasync_kernel<<<blocks, threads>>>(in, tiles, work, out);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return (double)ms * 1e3;
  };

  // Expected checksums (double on host; the per-thread float sums are of values with exact representations
  // and small magnitude, compared with a relative tolerance).
  std::vector<double> expect(outs, 0.0);
  for (int b = 0; b < blocks; ++b)
    for (int t = 0; t < tiles; ++t) {
      const size_t tb = ((size_t)b * tiles + t) * TILE_F;
      if (tma) {
        for (int c = 0; c < CONSUMERS; ++c)
          for (int l = 0; l < 32; ++l)
            for (int i = l; i < TILE_F / CONSUMERS; i += 32) expect[b * CONSUMERS * 32 + c * 32 + l] += h[tb + c * (TILE_F / CONSUMERS) + i];
      } else {
        for (int tid = 0; tid < 256; ++tid) {
          const int o = 4 * ((tid + 37) % 256);
          expect[b * 256 + tid] += h[tb + o] + h[tb + o + 1] + h[tb + o + 2] + h[tb + o + 3];
        }
      }
    }
  auto check = [&]() {
    std::vector<float> r(outs); CK(cudaMemcpy(r.data(), out, outs * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < outs; ++i) bad += fabs(r[i] - expect[i]) > 1e-5 * fabs(expect[i]) + 1e-3;
    return bad;
  };

  const char* name = tma ? "tma" : "cpasync";
  if (trace) { launch(); printf("%s,trace,slots=%d,work=%d,mismatches=%d\n", name, slots, work, check()); return 0; }
  for (int i = 0; i < 5; ++i) launch();
  std::vector<double> us; int bad = 0;
  for (int i = 0; i < 30; ++i) { CK(cudaMemset(out, 0, outs * 4)); us.push_back(launch()); bad += check(); }
  printf("%s,%s,blocks=%d,threads=%d,tiles=%d,tile_B=%d,slots=%d,work=%d,total_mismatches=%d\n", name, cold ? "cold" : "warm",
         blocks, threads, tiles, TILE_B, slots, work, bad);
  report((std::string(name) + "_kernel_us").c_str(), us);
  return bad ? 1 : 0;
}
