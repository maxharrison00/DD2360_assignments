#include <stdio.h>
#include <stdlib.h>
#include <cstdio>

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                    \
    }                                                    \
} while (0)

/* Our over-simplified CUDA kernel */
__global__ void add(int *a, int *b, int *c) {
     c[0] = a[0] + b[0];
}

int main() {

    int a=11, b=22, c=0; //@@ 3. Initialize host memory

    int *d_a, *d_b, *d_c; //@@ 1. Allocate in host memory
    int size = sizeof(int);
    cudaMalloc((void **)&d_a, size); //@@ 2. Allocate in device memory
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice); //@@ 4. Copy from host memory to device memory
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    printf("result before GPU computation is %d\n",c);

    dim3 grid(1,1,1); //@@ 5. Initialize thread block and thread grid
    dim3 tpb(32,1,1);
    add<<<grid,tpb>>>(d_a, d_b, d_c); //@@ 6. Invoke the CUDA kernel
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost); //@@ 7. Copy results from GPU to CPU
    printf("result after GPU computation is %d\n",c); //@@ 8. Compare the results with the CPU reference result

    // Cleanup
    cudaFree(d_a); //@@ 10. Free device memory
    cudaFree(d_b);
    cudaFree(d_c);
    return 0; //@@ 9. Free host memory
}
