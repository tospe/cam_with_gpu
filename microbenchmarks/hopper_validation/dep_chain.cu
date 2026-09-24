// Validation test 1 (guide §5): dependent arithmetic and memory accesses.
//   vecadd : independent streaming loads/stores (bandwidth-shaped)
//   chase  : one thread, each load address depends on the previous load (latency-shaped)
// Prints correctness and cudaEvent-timed kernel durations for hardware comparison.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void vecadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void chase(const unsigned* next, int hops, unsigned* out) {
  unsigned p = 0;
  for (int h = 0; h < hops; ++h) p = next[p];
  *out = p;
}

int main(int argc, char** argv) {
  const int n = argc > 1 ? atoi(argv[1]) : (1 << 16);
  const int hops = argc > 2 ? atoi(argv[2]) : 256;
  const int stride = 1024 + 32;  // elements; ~4 KiB + one line, defeats simple line reuse
  const int nodes = 4096;

  std::vector<float> ha(n), hb(n), hc(n);
  for (int i = 0; i < n; ++i) { ha[i] = i * 0.5f; hb[i] = 1.0f - i; }
  std::vector<unsigned> hnext((size_t)nodes * stride, 0);
  for (int k = 0; k < nodes; ++k) hnext[(size_t)k * stride] = (unsigned)(((k + 1) % nodes) * stride);

  float *a, *b, *c; unsigned *next, *out;
  CK(cudaMalloc(&a, n * sizeof(float))); CK(cudaMalloc(&b, n * sizeof(float)));
  CK(cudaMalloc(&c, n * sizeof(float)));
  CK(cudaMalloc(&next, hnext.size() * sizeof(unsigned))); CK(cudaMalloc(&out, sizeof(unsigned)));
  CK(cudaMemcpy(a, ha.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(b, hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(next, hnext.data(), hnext.size() * sizeof(unsigned), cudaMemcpyHostToDevice));

  cudaEvent_t e0, e1, e2; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2));
  CK(cudaEventRecord(e0));
  vecadd<<<(n + 255) / 256, 256>>>(a, b, c, n);
  CK(cudaEventRecord(e1));
  chase<<<1, 1>>>(next, hops, out);
  CK(cudaEventRecord(e2));
  CK(cudaEventSynchronize(e2)); CK(cudaGetLastError());

  CK(cudaMemcpy(hc.data(), c, n * sizeof(float), cudaMemcpyDeviceToHost));
  unsigned hout; CK(cudaMemcpy(&hout, out, sizeof(unsigned), cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < n; ++i) bad += hc[i] != ha[i] + hb[i];
  unsigned expect = (unsigned)((hops % nodes) * stride);
  float t_add, t_chase; CK(cudaEventElapsedTime(&t_add, e0, e1)); CK(cudaEventElapsedTime(&t_chase, e1, e2));
  printf("vecadd n=%d mismatches=%d time_ms=%.4f\n", n, bad, t_add);
  printf("chase hops=%d result=%u expect=%u %s time_ms=%.4f\n", hops, hout, expect,
         hout == expect ? "OK" : "FAIL", t_chase);
  return (bad || hout != expect) ? 1 : 0;
}
