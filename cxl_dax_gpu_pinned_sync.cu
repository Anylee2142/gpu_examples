#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>


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

    // 1. Map CXL memory into CPU space
    int fd = open(dax, O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/dax"); return 1; }

    void *h_dax = mmap(nullptr, BYTES,
                       PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_SYNC,
                       fd, 0);
    if (h_dax == MAP_FAILED) { perror("mmap"); return 1; }

    // 2. Pin and register for GPU
    cudaHostRegister(h_dax, BYTES,
                     cudaHostRegisterMapped | cudaHostRegisterPortable);
    uint32_t *d_dax = nullptr;
    cudaHostGetDevicePointer(&d_dax, h_dax, 0);

    // 3. Kernel writes pattern to CXL memory (no stream)
    dim3 blk(256), grid((WORDS + blk.x - 1) / blk.x);
    write_pattern<<<grid, blk>>>(d_dax, WORDS, 0xdeadbeef);
    cudaDeviceSynchronize();  // ensure kernel completion

    // 4. Copy to a temporary device buffer
    uint32_t *d_tmp = nullptr;
    cudaMalloc(&d_tmp, BYTES);
    cudaMemcpy(d_tmp, d_dax, BYTES, cudaMemcpyDeviceToDevice);

    // 5. Verify pattern on GPU
    int *d_err = nullptr;
    cudaMalloc(&d_err, sizeof(int));
    cudaMemset(d_err, 0, sizeof(int));
    check_pattern<<<grid, blk>>>(d_tmp, WORDS, 0xdeadbeef, d_err);
    cudaDeviceSynchronize();

    // 6. Copy result back to host
    int h_err = 0;
    cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << (h_err ? "Mismatch!\n" : "Pattern verified.\n");

    // 7. Cleanup
    cudaFree(d_tmp);
    cudaFree(d_err);
    cudaHostUnregister(h_dax);
    munmap(h_dax, BYTES);
    close(fd);
    return h_err;
}

