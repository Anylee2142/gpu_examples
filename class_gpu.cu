// vec_add_pageable.cu
#include <cuda_runtime.h>
#include <iostream>

__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1 << 20;              // 1 M elements
    const size_t bytes = N * sizeof(float);

    // 1. Allocate host (CPU) memory – **pageable** by default.
    float *h_a = new float[N], *h_b = new float[N], *h_c = new float[N];
    for (int i = 0; i < N; ++i) { h_a[i] = i; h_b[i] = 2*i; }

    // 2. Allocate device (GPU) memory.
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 3. Copy host → device (blocking call).
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 4. Launch kernel.
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    vecAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);

    // 5. Copy result back (blocking).
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // 6. Verify a couple values.
    std::cout << "c[123] = " << h_c[123] << " (should be 3×123)\n";

    // 7. Cleanup.
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
}

