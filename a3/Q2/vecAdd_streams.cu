#include <stdio.h>
#include <stdlib.h>

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

void vecAdd(float *a, float *b, float *c, int len) {
  for (int i = 0; i < len; i++) {
    c[i] = a[i] + b[i];
  }
}

__global__ void add(float *a, float *b, float *c, int len, int offset) {
  int i = offset + (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < len) {
    c[i] = a[i] + b[i];
  }
}

int main(int argc, char **argv) {
  
  if (argc < 2 || argc > 3) {
    return 0;
  }
  int len = atoi(argv[1]);
  int nStreams = 4;
  int streamSize = (len + nStreams - 1) / nStreams;
  int streamBytes = streamSize * sizeof(float);

  //@@ 1. Allocate in host memory
  int size = len * sizeof(float);
  float* h_a;
  float* h_b;
  float* h_c;
  float* result;

  cudaHostAlloc(&h_a, size, cudaHostAllocDefault);
  cudaHostAlloc(&h_b, size, cudaHostAllocDefault);
  cudaHostAlloc(&h_c, size, cudaHostAllocDefault);
  cudaHostAlloc(&result, size, cudaHostAllocDefault);



  //@@ 2. Allocate in device memory
  float *d_a, *d_b, *d_c;  
  cudaMalloc((void **)&d_a, size); 
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  //@@ 3. Initialize host memory
  for (int i = 0; i < len; i++) {
    h_a[i] = ((float) i) / (len-1);
    h_b[i] = ((float) i) / (len-1);
  }

  cudaStream_t stream[nStreams];
  for (int i = 0; i < nStreams; i++) {
    cudaStreamCreate(&stream[i]);
  }

  //@@ 5. Initialize thread block and thread grid
  int blocks_per_grid = (streamSize + TBP - 1) / TBP;
  int thread_block = TBP;
  int offset;

  for(int i = 0; i < nStreams; i++) {
    offset = i * streamSize;
    if (i == nStreams - 1) {
      streamSize = len - offset;
      streamBytes = streamSize * sizeof(float);
    }
    cudaMemcpyAsync(&d_a[offset], &h_a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync(&d_b[offset], &h_b[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
    add<<<blocks_per_grid, TBP, 0, stream[i]>>>(d_a, d_b, d_c, len, offset);
    cudaMemcpyAsync(&result[offset], &d_c[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
  }

  cudaDeviceSynchronize();

  //@@ 8. Compare the results with the CPU reference result
  vecAdd(h_a, h_b, h_c, len);
  float error = 0;
  for (int i = 0; i < len; i++) {
    error += h_c[i]-result[i];
  }
  printf("Total error between CPU and GPU versions is %f\n", error); 

  //@@ 9. Free host  memory
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);
  cudaFreeHost(result);

  //@@ 10. Free device memory 
  cudaFree(d_a); 
  cudaFree(d_b);
  cudaFree(d_c);

  for (int i = 0; i < nStreams; i++) {
    cudaStreamDestroy(stream[i]);
  }

  return 0;
}
