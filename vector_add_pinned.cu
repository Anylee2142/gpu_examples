#include <cuda_runtime.h>
#include <iostream>

__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    // 1. Page-locked host allocations (pinning).
    float *h_a, *h_b, *h_c;
    cudaHostAlloc(&h_a, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_b, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_c, bytes, cudaHostAllocDefault);

    for (int i = 0; i < N; ++i) { h_a[i] = i; h_b[i] = 2*i; }

    // 2. Device buffers.
    float *d_a, *d_b, *d_c;
    cudaMallocAsync(&d_a, bytes, 0);   // 0 = default stream
    cudaMallocAsync(&d_b, bytes, 0);
    cudaMallocAsync(&d_c, bytes, 0);

    // 3. Create a non-default stream so we can overlap.
    cudaStream_t s;
    cudaStreamCreate(&s);

    // 4. Async copies – these return immediately.
    cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, s);

    // 5. Kernel launch in the same stream (runs after copies).
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    vecAdd<<<blocks, threads, 0, s>>>(d_a, d_b, d_c, N);

    // 6. Copy result back, still in stream `s`.
    cudaMemcpyAsync(h_c, d_c, bytes, cudaMemcpyDeviceToHost, s);

    // 7. Wait for all stream work to finish.
    cudaStreamSynchronize(s);

    std::cout << "c[123] = " << h_c[123] << " (should be 3×123)\n";

    // 8. Cleanup.
    cudaFreeAsync(d_a, 0); cudaFreeAsync(d_b, 0); cudaFreeAsync(d_c, 0);
    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);
    cudaStreamDestroy(s);
}

