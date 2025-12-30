#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>

#define gpuCheck(stmt)                                               \
  do {                                                               \
      cudaError_t err = stmt;                                        \
      if (err != cudaSuccess) {                                      \
          printf("ERROR. Failed to run stmt %s\n", #stmt);           \
          break;                                                     \
      }                                                              \
  } while (0)

struct timeval t_start, t_end;
void cputimer_start(){
  gettimeofday(&t_start, 0);
}

void cputimer_stop(const char* info){
  gettimeofday(&t_end, 0);
  double time = (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
  printf("Timing - %s. \t\tElasped %.0f microseconds \n", info, time);
}

#define MASK_WIDTH 5
#define MASK_RADIUS 2
#define TILE_WIDTH 256

// --- CPU REFERENCE IMPLEMENTATION ---
void convolution_1D_cpu(float *N, float *M, float *P, int Width, int Mask_Width) {
    int radius = Mask_Width / 2;
    for (int i = 0; i < Width; i++) {
        float Pvalue = 0.0f;
        int start_point = i - radius;
        for (int j = 0; j < Mask_Width; j++) {
            int idx = start_point + j;
            if (idx >= 0 && idx < Width) {
                Pvalue += N[idx] * M[j];
            }
        }
        P[i] = Pvalue;
    }
}

// --- GPU KERNELS ---
__global__ void convolution_1D_basic(float *N, float *M, float *P, int Width)
{
    //@@ INSERT CODE HERE
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Width) return;

    float Pvalue = 0;
    int start_point = i - MASK_RADIUS;

    for (int j = 0; j < MASK_WIDTH; j++) {
        int idx = start_point + j;
        if (idx >= 0 && idx < Width) {
            Pvalue += N[idx] * M[j];
        }
    }
    P[i] = Pvalue;
}

__global__ void convolution_1D_tiled(float *N, float *M, float *P, int Width)
{
    //@@ INSERT CODE HERE
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    //@@ INSERT CODE HERE
    __shared__ float N_ds[TILE_WIDTH + MASK_WIDTH - 1];

    //@@ INSERT CODE HERE
    int s_idx = threadIdx.x + MASK_RADIUS; 
    
    // Load center element
    if (i < Width) 
        N_ds[s_idx] = N[i];
    else 
        N_ds[s_idx] = 0.0f;

    // Load Left Halo 
    if (threadIdx.x < MASK_RADIUS) {
        int left_idx = i - MASK_RADIUS; 
        if (left_idx >= 0 && left_idx < Width)
            N_ds[threadIdx.x] = N[left_idx];
        else
            N_ds[threadIdx.x] = 0.0f;
    }

    // Load Right Halo 
    if (threadIdx.x >= blockDim.x - MASK_RADIUS) {
        int halo_dest_idx = threadIdx.x + 2 * MASK_RADIUS; 
        
        if (halo_dest_idx < (TILE_WIDTH + MASK_WIDTH - 1)) {
             int global_r = (blockIdx.x * blockDim.x) + halo_dest_idx - MASK_RADIUS;
             if (global_r < Width)
                 N_ds[halo_dest_idx] = N[global_r];
             else
                 N_ds[halo_dest_idx] = 0.0f;
        }
    }
    
    __syncthreads(); 

    // Compute Convolution
    if (i < Width) {
        float Pvalue = 0;
        for (int j = 0; j < MASK_WIDTH; j++) {
            Pvalue += N_ds[threadIdx.x + j] * M[j];
        }
        P[i] = Pvalue;
    }
}

int main(int argc, char *argv[]) {
  
  // Read the arguments from the command line
  if (argc != 2) {
      printf("Usage: %s <N>\n", argv[0]);
      return 0;
  }
  int N = atoi(argv[1]);

  float *hostN; // The input array N of length N
  float *hostM; // The 1D mask M of length MASK_WIDTH
  float *hostP; // The output array P of length N
  float *hostP_CPU; // Buffer for CPU result

  cputimer_start();
  //@@ Allocate the host memory
  hostN = (float*)malloc(N * sizeof(float));
  hostM = (float*)malloc(MASK_WIDTH * sizeof(float));
  hostP = (float*)malloc(N * sizeof(float));
  hostP_CPU = (float*)malloc(N * sizeof(float));
  cputimer_stop("Allocated host memory");

  float *deviceN;
  float *deviceM;
  float *deviceP;

  cputimer_start();
  //@@ Allocate the device memory
  gpuCheck(cudaMalloc((void**)&deviceN, N * sizeof(float)));
  gpuCheck(cudaMalloc((void**)&deviceM, MASK_WIDTH * sizeof(float)));
  gpuCheck(cudaMalloc((void**)&deviceP, N * sizeof(float)));
  cputimer_stop("Allocated device memory");
  
  cputimer_start();
  //@@ Initialize N with random values
  for(int i=0; i<N; i++) hostN[i] = (float)(rand() % 100) / 10.0f;
  
  //@@ Initialize M with [-0.25, 0.5, 1.0, 0.5, 0.25]
  float maskVals[] = {-0.25f, 0.5f, 1.0f, 0.5f, 0.25f};
  for(int i=0; i<MASK_WIDTH; i++) hostM[i] = maskVals[i];
  
  //@@ Initialize P with 0.0
  // (Done implicitly by kernel overwrite, or memset if needed)
  
  // RUN CPU VERSION (Reference)
  if (N <  4000){
      convolution_1D_cpu(hostN, hostM, hostP_CPU, N, MASK_WIDTH);
  }
  cputimer_stop("Initialized data");

  cputimer_start();
  //@@ INSERT CODE HERE
  gpuCheck(cudaMemcpy(deviceN, hostN, N * sizeof(float), cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy(deviceM, hostM, MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice));
  cputimer_stop("Copying data to the GPU.");
  
  int dimBlock = 256;
  int dimGrid = (N + dimBlock - 1) / dimBlock;

  /* Call the basic kernel */
  cputimer_start();
  //@@  Define the execution configuration
  //@@  Run the 1D convolution kernel (basic)
  convolution_1D_basic<<<dimGrid, dimBlock>>>(deviceN, deviceM, deviceP, N);
  gpuCheck(cudaDeviceSynchronize());
  cputimer_stop("Finished 1D convolution(basic)");
  
  // Reset Device P for Tiled test
  gpuCheck(cudaMemset(deviceP, 0, N * sizeof(float)));

  /* Call the tiled kernel */
  cputimer_start();
  //@@  Define the execution configuration
  //@@  Run the 1D convolution kernel (tiled)
  convolution_1D_tiled<<<dimGrid, TILE_WIDTH>>>(deviceN, deviceM, deviceP, N);
  gpuCheck(cudaDeviceSynchronize());
  cputimer_stop("Finished 1D convolution(tiled)");
  
  cputimer_start();
  //@@ INSERT CODE HERE
  gpuCheck(cudaMemcpy(hostP, deviceP, N * sizeof(float), cudaMemcpyDeviceToHost));
  cputimer_stop("Copying output P to the CPU and print out the results");

  //@@ Validate the results from the two implementations
  if (N < 4000){
      double total_error = 0.0;
      for(int i=0; i<N; i++) {
          total_error += fabs(hostP[i] - hostP_CPU[i]);
      }
      printf("Validation (GPU vs CPU): Total Error = %f\n", total_error);
      if(total_error < 1e-3) printf("TEST PASSED\n");
      else printf("TEST FAILED\n");
  } else {
      printf("Validation skipped for large N\n");
  }

  cputimer_start();
  //@@ INSERT CODE HERE
  free(hostN); free(hostM); free(hostP); free(hostP_CPU);
  cudaFree(deviceN); cudaFree(deviceM); cudaFree(deviceP);
  cputimer_stop("Free memory resources");

  return 0;
}