#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#define N 64

#define CHECK(call) do {                                 \
  cudaError_t err = (call);                            \
  if (err != cudaSuccess) {                            \
    std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                 cudaGetErrorString(err), __FILE__, __LINE__); \
    std::exit(1);                                    \
  }                                                    \
} while (0)

// Threads per Block
const int TBP = 32;

void vecAdd(float *a, float *b, float *c) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

__global__ void add(float *a, float *b, float *c) {
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  c[i] = a[i] + b[i];
}

int main() {
  //@@ 1. Allocate in host memory
  int size = N * sizeof(float);
  float* h_a = (float *) malloc(N * size);
  float* h_b = (float *) malloc(N * size);
  float* h_c = (float *) malloc(N * size);
  float* result = (float *) malloc(N * size);

  //@@ 2. Allocate in device memory
  float *d_a, *d_b, *d_c;  
  cudaMalloc((void **)&d_a, size); 
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  //@@ 3. Initialize host memory
  for (int i = 0; i < N; i++) {
    // use anything here to initialise values to the vectors
    h_a[i] = ((float) i) / (N-1);
    h_b[i] = ((float) i) / (N-1);
  }

  //@@ 4. Copy from host memory to device memory
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice); 
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  //@@ 5. Initialize thread block and thread grid
  int blocks_per_grid = (N + TBP - 1) / TBP; // ceiling division
  int thread_block = TBP;

  //@@ 6. Invoke the CUDA kernel
  add<<<blocks_per_grid, thread_block>>>(d_a, d_b, d_c);

  //@@ 7. Copy results from GPU to CPU
  cudaMemcpy(result, d_c, size, cudaMemcpyDeviceToHost);

  //@@ 8. Compare the results with the CPU reference result
  vecAdd(h_a, h_b, h_c);
  float error = 0;
  for (int i = 0; i < N; i++) {
    error += h_c[i]-result[i];
  }
  printf("Total error between CPU and GPU versions is %f\n", error); 

  //@@ 9. Free host  memory
  free(h_a);
  free(h_b);
  free(h_c);
  free(result);

  //@@ 10. Free device memory 
  cudaFree(d_a); 
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
