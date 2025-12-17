#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

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
#define TILE_WIDTH //@@ INSERT CODE HERE


__global__ void convolution_1D_basic(float *N, float *M, float *P)
{
    //@@ INSERT CODE HERE

}

__global__ void convolution_1D_tiled(float *N, float *M, float *P)
{
  //@@ INSERT CODE HERE
  __shared__ float input_tile[TILE_WIDTH];
  
  //@@ INSERT CODE HERE
  
}

int main(int argc, char *argv[]) {
  
  // Read the arguments from the command line
  int N = atoi(argv[1]);


  float *hostN; // The input array N of length N
  float *hostM; // The 1D mask M of length MASK_WIDTH
  float *hostP; // The output array P of length N

  cputimer_start();
  //@@ Allocate the host memory
  cputimer_stop("Allocated host memory");


  float *deviceN;
  float *deviceM;
  float *deviceP;

  cputimer_start();
  //@@ Allocate the device memory
  cputimer_stop("Allocated device memory");

  
  cputimer_start();
  //@@ Initialize N with random values
  //@@ Initialize M with [-0.25, 0.5, 1.0, 0.5, 0.25]
  //@@ Initialize P with 0.0
  cputimer_stop("Allocated device memory");

  
  cputimer_start();
  //@@ INSERT CODE HERE
  cputimer_stop("Copying data to the GPU.");
  

  /* Call the basic kernel */
  cputimer_start();
  //@@  Define the execution configuration
  //@@  Run the 1D convolution kernel (basic)
  cputimer_stop("Finished 1D convolution(basic)");
  
  cputimer_start();
  //@@ INSERT CODE HERE
  cputimer_stop("Copying output P to the CPU and print out the results");

  /* Call the tiled kernel */
  cputimer_start();
  //@@  Define the execution configuration
  //@@  Run the 1D convolution kernel (tiled)
  cputimer_stop("Finished 1D convolution(tiled)");
  
  cputimer_start();
  //@@ INSERT CODE HERE
  cputimer_stop("Copying output P to the CPU and print out the results");


  //@@ Validate the results from the two implementations


  cputimer_start();
  //@@ INSERT CODE HERE
  cputimer_stop("Free memory resources");

  return 0;
}
