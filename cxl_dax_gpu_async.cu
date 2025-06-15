// cxl_dax_gpu_demo.cu  (compile with:  nvcc -O3 cxl_dax_gpu_demo.cu -o dax_demo)
// Requires: CUDA >= 12, driver that still allows cudaHostRegister on remap_pfn_range,
//            GPU & CXL endpoint on same PCIe/CXL root complex, IOMMU/ATS on.

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void write_pattern(uint32_t *buf, size_t n, uint32_t seed)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = seed + i;
}

__global__ void check_pattern(uint32_t *buf, size_t n, uint32_t seed, int *err)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && buf[i] != seed + i) atomicAdd(err, 1);
}

int main()
{
    const char *dax = "/dev/dax0.0";             // your CXL-mem device node
    const size_t BYTES = 64ULL << 20;            // 64 MiB slice
    const size_t WORDS = BYTES / sizeof(uint32_t);

    /* ───── 1) Map CXL memory into the CPU address space ───── */
    int fd = open(dax, O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/dax"); return 1; }

    void *h_dax = mmap(nullptr, BYTES,
                       PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_SYNC,   // MAP_SYNC keeps pmem writes durable
                       fd, 0);
    if (h_dax == MAP_FAILED) { perror("mmap"); return 1; }

    /* ───── 2) Pin + register for the GPU ───── */
    cudaHostRegister(h_dax, BYTES,
                     cudaHostRegisterMapped | cudaHostRegisterPortable);
    uint32_t *d_dax = nullptr;
    cudaHostGetDevicePointer(&d_dax, h_dax, 0);

    /* ───── 3) Set up a stream so everything is async ───── */
    cudaStream_t s;  cudaStreamCreate(&s);

    /* ───── 4) Kernel writes a pattern directly into CXL mem ───── */
    dim3 blk(256), grid((WORDS + blk.x - 1) / blk.x);
    write_pattern<<<grid, blk, 0, s>>>(d_dax, WORDS, 0xdeadbeef);

    /* Optional: copy the block to a temp device buffer to show D→D bandwidth */
    uint32_t *d_tmp;             cudaMallocAsync(&d_tmp, BYTES, s);
    cudaMemcpyAsync(d_tmp, d_dax, BYTES,
                    cudaMemcpyDeviceToDevice, s);

    /* Verify on-GPU, still in the same stream */
    int *d_err;   cudaMallocAsync(&d_err, sizeof(int), s);
    cudaMemsetAsync(d_err, 0, sizeof(int), s);
    check_pattern<<<grid, blk, 0, s>>>(d_tmp, WORDS, 0xdeadbeef, d_err);

    int h_err = 0;
    cudaMemcpyAsync(&h_err, d_err, sizeof(int),
                    cudaMemcpyDeviceToHost, s);

    /* ───── 5) Wait, report, and clean up ───── */
    cudaStreamSynchronize(s);
    std::cout << (h_err ? "Mismatch!\n" : "Pattern verified.\n");

    cudaFreeAsync(d_tmp, s);
    cudaFreeAsync(d_err, s);
    cudaStreamDestroy(s);

    cudaHostUnregister(h_dax);
    munmap(h_dax, BYTES);
    close(fd);
    return h_err;
}

