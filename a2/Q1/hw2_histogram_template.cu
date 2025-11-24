#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096
// Threads per Block
const int TBP = 32;

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_elements) {
    atomicAdd(&(bins[input[i]]), 1);
  }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_bins && bins[i] > 127) {
    bins[i] = 127;
  }
}

void initialise_uniform(unsigned int *hostInput, int inputLength) {
    for (int i = 0; i < inputLength; i++) {
        hostInput[i] = rand() % NUM_BINS;
    }
}

void initialise_normal(unsigned int *hostInput, int inputLength) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(NUM_BINS/2.0, NUM_BINS/6.0);

    for (int i = 0; i < inputLength; i++) {
        double value = distribution(gen);
        // Clamp values to valid bin range [0, NUM_BINS-1]
        if (value < 0) value = 0;
        if (value >= NUM_BINS) value = NUM_BINS - 1;
        hostInput[i] = static_cast<unsigned int>(value);
    }
}

int main(int argc, char **argv) {
  
    if (argc != 3) {
        printf("Usage: %s <input_length> <distribution_type>\n", argv[0]);
        return 1;
    }
  
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *resultRef;
    unsigned int *deviceInput;
    unsigned int *deviceBins;

    // Read input length
    int inputLength = atoi(argv[1]);
    
    // Read distribution type
    char* distributionType = argv[2];
    
    printf("Input length: %d\n", inputLength);
    printf("Distribution type: %s\n", distributionType);
  
    hostInput = (unsigned int*) malloc(inputLength * sizeof(unsigned int));
    hostBins = (unsigned int*) malloc(NUM_BINS * sizeof(unsigned int));
  
    //@@ Initialise hostInput based on distribution type
    if (strcmp(distributionType, "Uniform") == 0) {
        initialise_uniform(hostInput, inputLength);
    } else if (strcmp(distributionType, "Normal") == 0) {
        initialise_normal(hostInput, inputLength);
    } else {
        printf("Error: Unknown distribution type '%s'. Using Uniform as default.\n", distributionType);
        initialise_uniform(hostInput, inputLength);
    }

    //@@ Insert code below to create reference result in CPU
    resultRef = (unsigned int*) malloc(NUM_BINS * sizeof(unsigned int));
    for (int i = 0; i < NUM_BINS; i++) {
        resultRef[i] = 0;
    }
    for (int i = 0; i < inputLength; i++) {
        resultRef[hostInput[i]] += 1;
    }
    for (int i = 0; i < NUM_BINS; i++) {
        if (resultRef[i] > 127) {
            resultRef[i] = 127;
        }
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc((void **)&deviceInput, inputLength * sizeof(unsigned int));
    cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int));

    //@@ Insert code to Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //@@ Insert code to initialize GPU results
    cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

    //@@ Initialize the grid and block dimensions here
    dim3 grid_dim_hist((inputLength + TBP - 1) / TBP); // ceiling division
    dim3 block_dim_hist(TBP);

    //@@ Launch the GPU Kernel here
    histogram_kernel<<<grid_dim_hist, block_dim_hist>>>(deviceInput, deviceBins, inputLength, NUM_BINS); 
    cudaDeviceSynchronize();

    //@@ Initialize the second grid and block dimensions here
    dim3 grid_dim_conv((NUM_BINS + TBP - 1) / TBP); // ceiling division
    dim3 block_dim_conv(TBP);

    //@@ Launch the second GPU Kernel here
    convert_kernel<<<grid_dim_conv, block_dim_conv>>>(deviceBins, NUM_BINS);
    cudaDeviceSynchronize();

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //@@ Insert code below to compare the output with the reference
    int numDiscrepancies = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        if (hostBins[i] != resultRef[i]) {
            numDiscrepancies++;
        }
    }
    printf("Total number of discrepant bins between the GPU and CPU versions is %i\n", numDiscrepancies);
    if (numDiscrepancies == 0) {
        printf("PASS: GPU and CPU results match!\n");
    } else {
        printf("FAIL: GPU and CPU results have %d differences\n", numDiscrepancies);
    }

    //@@ Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceBins);

    //@@ Free the CPU memory here
    free(hostInput);
    free(hostBins);
    free(resultRef);

    return 0;
}
