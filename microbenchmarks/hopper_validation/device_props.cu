// Dump device properties used to derive the simulator config.
#include <cstdio>
#include <cuda_runtime.h>
int main() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  int clk, memclk, l2, bus, smem_sm, smem_blk_optin, regs_sm, maxthr_sm, maxblk_sm, asyncEng, clusterLaunch;
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
  cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, 0);
  cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0);
  cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);
  cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
  cudaDeviceGetAttribute(&smem_blk_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  cudaDeviceGetAttribute(&regs_sm, cudaDevAttrMaxRegistersPerMultiprocessor, 0);
  cudaDeviceGetAttribute(&maxthr_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
  cudaDeviceGetAttribute(&maxblk_sm, cudaDevAttrMaxBlocksPerMultiprocessor, 0);
  cudaDeviceGetAttribute(&asyncEng, cudaDevAttrAsyncEngineCount, 0);
  cudaDeviceGetAttribute(&clusterLaunch, cudaDevAttrClusterLaunch, 0);
  printf("name: %s\ncc: %d.%d\nsms: %d\nsm_clock_max_khz: %d\nmem_clock_max_khz: %d\n", p.name, p.major, p.minor, p.multiProcessorCount, clk, memclk);
  printf("mem_bus_width_bits: %d\nglobal_mem_bytes: %zu\nl2_bytes: %d\npersisting_l2_max_bytes: %d\n", bus, p.totalGlobalMem, l2, p.persistingL2CacheMaxSize);
  printf("smem_per_sm_bytes: %d\nsmem_per_block_optin_bytes: %d\nreserved_smem_per_block: %zu\nregs_per_sm: %d\nmax_threads_per_sm: %d\nmax_blocks_per_sm: %d\nwarp_size: %d\n", smem_sm, smem_blk_optin, p.reservedSharedMemPerBlock, regs_sm, maxthr_sm, maxblk_sm, p.warpSize);
  printf("async_engines: %d\ncluster_launch: %d\necc: %d\npci: %04x:%02x:%02x\n", asyncEng, clusterLaunch, p.ECCEnabled, p.pciDomainID, p.pciBusID, p.pciDeviceID);
  return 0;
}
