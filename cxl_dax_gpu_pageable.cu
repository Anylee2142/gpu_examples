// cxl_dax_gpu_demo_pageable.cu
// Build: nvcc -O3 -std=c++17 cxl_dax_gpu_demo_pageable.cu -o dax_demo_pageable

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>

#define CUDA(call)                                                         \
    do {                                                                   \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess) {                                           \
            std::cerr << "CUDA error " << cudaGetErrorString(_e)           \
                      << " at " << __FILE__ << ':' << __LINE__ << '\n';    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

/* ---------------- kernels ---------------- */
__global__ void write_pattern(uint32_t* buf, size_t n, uint32_t seed)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = seed + i;
}

__global__ void check_pattern(const uint32_t* buf, size_t n,
                              uint32_t seed, int* err)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && buf[i] != seed + i) atomicAdd(err, 1);
}

/* ---------------- host program ---------------- */
int main()
{
    constexpr const char* DAX = "/dev/dax0.0";
    constexpr size_t BYTES  = 64ULL << 20;              // 64 MiB
    constexpr size_t WORDS  = BYTES / sizeof(uint32_t);
    constexpr uint32_t SEED = 0xDEADBEEF;

    /* 1. mmap the CXL slice (pageable) */
    int fd = open(DAX, O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/dax"); return EXIT_FAILURE; }

    void* h_dax = mmap(nullptr, BYTES,
                       PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_SYNC, fd, 0);
    if (h_dax == MAP_FAILED) { perror("mmap"); return EXIT_FAILURE; }

    /* 2. Allocate device buffer */
    uint32_t* d_buf = nullptr;
    CUDA(cudaMalloc(&d_buf, BYTES));

    /* 3. GPU writes pattern into *device* buffer */
    dim3 blk(256), grid((WORDS + blk.x - 1) / blk.x);
    write_pattern<<<grid, blk>>>(d_buf, WORDS, SEED);
    CUDA(cudaDeviceSynchronize());

    /* 4. Copy device → CXL (pageable host) */
    CUDA(cudaMemcpy(h_dax, d_buf, BYTES, cudaMemcpyDeviceToHost));

    /* 5. Copy back CXL → device to verify round-trip */
    CUDA(cudaMemcpy(d_buf, h_dax, BYTES, cudaMemcpyHostToDevice));
    int* d_err = nullptr;
    CUDA(cudaMalloc(&d_err, sizeof(int)));
    CUDA(cudaMemset(d_err, 0, sizeof(int)));

    check_pattern<<<grid, blk>>>(d_buf, WORDS, SEED, d_err);
    CUDA(cudaDeviceSynchronize());

    int h_err = 0;
    CUDA(cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << (h_err ? "Mismatch!\n" : "Pattern verified.\n");

    /* 6. Cleanup */
    CUDA(cudaFree(d_buf));
    CUDA(cudaFree(d_err));
    munmap(h_dax, BYTES);
    close(fd);
    CUDA(cudaDeviceReset());
    return h_err ? EXIT_FAILURE : EXIT_SUCCESS;
}
