#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <thread>

__global__ void spin_kernel(float *a, float *b, float *c, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float value = a[idx];
    for (int i = 0; i < iterations; ++i) {
        value = value * 1.000001f + b[idx];
    }
    c[idx] = value;
}

void check(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    int device_id = 0;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--device") == 0 || strcmp(argv[i], "-d") == 0) && (i + 1) < argc) {
            device_id = atoi(argv[++i]);
        } else {
            fprintf(stderr, "Usage: %s [--device <id>]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }

    int device_count = 0;
    check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_id < 0 || device_id >= device_count) {
        fprintf(stderr, "Requested device %d is out of range (0-%d)\n", device_id, device_count - 1);
        return EXIT_FAILURE;
    }

    check(cudaSetDevice(device_id), "cudaSetDevice");

    const int elements = 1 << 24; // ~16 million elements
    const int iterations = 1 << 10;
    const size_t bytes = sizeof(float) * elements;

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < elements; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
        h_c[i] = 0.0f;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    check(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    check(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    check(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

    check(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "cudaMemcpy h_a");
    check(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "cudaMemcpy h_b");

    dim3 block(256);
    dim3 grid((elements + block.x - 1) / block.x);

    while (true) {
        spin_kernel<<<grid, block>>>(d_a, d_b, d_c, elements, iterations);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
            break;
        }
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
