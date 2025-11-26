#include <cstdlib>
#include <random>
#include <stdio.h>
#include <sys/time.h>

// Threads per block
const int TBP = 256;

// Phase 1: Each block reduces its portion and writes to separate output
__global__ void reduction_kernel_phase1(float *input, float *output,
                                        int inputLength) {
  __shared__ float partial_sums[TBP];

  int s_idx = threadIdx.x;                       // Local index tid
  int i = blockIdx.x * blockDim.x + threadIdx.x; // Global index idx

  // Each thread processes multiple elements
  float sum = 0.0f;
  for (int j = i; j < inputLength; j += gridDim.x * blockDim.x) {
    sum += input[j];
  }
  partial_sums[s_idx] = sum;
  __syncthreads();

  // Reduction in shared memory using stride pattern
  // Really a binary tree of operations, with each operation corresponding to a
  // level of the tree
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (s_idx < s) {
      partial_sums[s_idx] += partial_sums[s_idx + s];
    }
    __syncthreads();
  }

  // Write block result to global memory (no atomicAdd)
  if (s_idx == 0) {
    output[blockIdx.x] = partial_sums[0];
  }
}

// Phase 2: Reduce the block results (can be done on CPU or GPU)
void reduce_on_cpu(float *blockResults, int numBlocks, float *finalResult) {
  *finalResult = 0.0f;
  for (int i = 0; i < numBlocks; i++) {
    *finalResult += blockResults[i];
  }
}

int main(int argc, char **argv) {
  int inputLength;

  // Read in inputLength from args
  inputLength = atoi(argv[1]);
  printf("The input length is %d\n", inputLength);

  // Calculate number of blocks and allocate output for each block
  int numBlocks = min((inputLength + TBP - 1) / TBP, 1024);

  // Host memory allocation
  float *hostInput = (float *)malloc(inputLength * sizeof(float));
  float hostResult = 0;
  float *hostBlockResults = (float *)malloc(numBlocks * sizeof(float));

  // Device memory allocation
  float *deviceInput;
  float *deviceBlockResults;
  cudaMalloc((void **)&deviceInput, inputLength * sizeof(float));
  cudaMalloc((void **)&deviceBlockResults, numBlocks * sizeof(float));

  // Initialise input array with random values between 0 and 1
  for (int i = 0; i < inputLength; i++) {
    hostInput[i] = rand() / (RAND_MAX + 1.0);
  }

  // CPU reference computation with timer
  struct timeval cpu_start, cpu_end, gpu_start, gpu_end;
  gettimeofday(&cpu_start, NULL);
  for (int i = 0; i < inputLength; i++) {
    hostResult += hostInput[i];
  }
  gettimeofday(&cpu_end, NULL);
  double cpu_time = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0 +
                    (cpu_end.tv_usec - cpu_start.tv_usec) / 1000.0;

  // Copy data from CPU to GPU
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(float),
             cudaMemcpyHostToDevice);

  // Launch kernel with timer
  dim3 grid_dim(numBlocks);
  dim3 block_dim(TBP);

  gettimeofday(&gpu_start, NULL);

  // Phase 1: Reduce within blocks
  reduction_kernel_phase1<<<grid_dim, block_dim>>>(
      deviceInput, deviceBlockResults, inputLength);
  cudaDeviceSynchronize();

  // Copy block results back to host
  cudaMemcpy(hostBlockResults, deviceBlockResults, numBlocks * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Phase 2: Final reduction on CPU
  float deviceResult;
  reduce_on_cpu(hostBlockResults, numBlocks, &deviceResult);

  gettimeofday(&gpu_end, NULL);
  double gpu_time = (gpu_end.tv_sec - gpu_start.tv_sec) * 1000.0 +
                    (gpu_end.tv_usec - gpu_start.tv_usec) / 1000.0;

  // Compare results
  printf("CPU result: %f\n", hostResult);
  printf("GPU result: %f\n", deviceResult);
  printf("Difference: %f\n", fabs(hostResult - deviceResult));
  printf("CPU time: %f ms\n", cpu_time);
  printf("GPU time: %f ms\n", gpu_time);
  printf("Number of blocks: %d\n", numBlocks);
  printf("Threads per block: %d\n", TBP);

  if (gpu_time < cpu_time) {
    printf("GPU version is %fx faster than CPU\n", cpu_time / gpu_time);
  } else {
    printf("CPU version is %fx faster than GPU\n", gpu_time / cpu_time);
  }

  // Free memory
  free(hostInput);
  free(hostBlockResults);
  cudaFree(deviceInput);
  cudaFree(deviceBlockResults);

  return 0;
}
