#include <cstdlib>
#include <random>
#include <stdio.h>
#include <sys/time.h>

// Threads per Block
const int TBP = 32;

__global__ void reduction_kernel(float *input, float *output, int inputLength) {
  __shared__ float partial_sums[TBP];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory
  if (i < inputLength) {
    partial_sums[tid] = input[i];
  } else {
    partial_sums[tid] = 0.0f;
  }
  __syncthreads();

  // Reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      partial_sums[tid] += partial_sums[tid + s];
    }
    __syncthreads();
  }

  // Write block result to global memory
  if (tid == 0) {
    atomicAdd(output, partial_sums[0]);
  }
}

int main(int argc, char **argv) {
  int inputLength;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);
  printf("The input length is %d\n", inputLength);

  /*@ add other needed data allocation on CPU and GPU here */
  float *hostInput = (float *)malloc(inputLength * sizeof(float));
  float hostResult = 0;

  float *deviceInput;
  float *deviceOutput;
  cudaMalloc((void **)&deviceInput, inputLength * sizeof(float));
  cudaMalloc((void **)&deviceOutput, sizeof(float));

  //@@ Insert code below to initialize the input array with random values on CPU
  for (int i = 0; i < inputLength; i++) {
    hostInput[i] = rand() / (RAND_MAX + 1.0);
  }

  //@@ Insert code below to create reference result in CPU and add a timer
  struct timeval cpu_start, cpu_end, gpu_start, gpu_end;

  gettimeofday(&cpu_start, NULL);
  for (int i = 0; i < inputLength; i++) {
    hostResult += hostInput[i];
  }
  gettimeofday(&cpu_end, NULL);

  double cpu_time = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0 +
                    (cpu_end.tv_usec - cpu_start.tv_usec) / 1000.0;

  //@@ Insert code to copy data from CPU to the GPU
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(float),
             cudaMemcpyHostToDevice);

  //@@ Initialize device output to zero
  float zero = 0.0f;
  cudaMemcpy(deviceOutput, &zero, sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 grid_dim((inputLength + TBP - 1) / TBP); // ceiling division
  dim3 block_dim(TBP);

  //@@ Launch the GPU Kernel here and add a timer
  gettimeofday(&gpu_start, NULL);
  reduction_kernel<<<grid_dim, block_dim>>>(deviceInput, deviceOutput,
                                            inputLength);
  cudaDeviceSynchronize();
  gettimeofday(&gpu_end, NULL);

  double gpu_time = (gpu_end.tv_sec - gpu_start.tv_sec) * 1000.0 +
                    (gpu_end.tv_usec - gpu_start.tv_usec) / 1000.0;

  //@@ Copy the GPU memory back to the CPU here
  float deviceResult;
  cudaMemcpy(&deviceResult, deviceOutput, sizeof(float),
             cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  printf("CPU result: %f\n", hostResult);
  printf("GPU result: %f\n", deviceResult);
  printf("Difference: %f\n", fabs(hostResult - deviceResult));
  printf("CPU time: %f ms\n", cpu_time);
  printf("GPU time: %f ms\n", gpu_time);
  printf("CPU version took %f seconds longer than the GPU version\n",
         (cpu_time - gpu_time) / 1000.0);

  //@@ Free memory here
  free(hostInput);
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  return 0;
}
