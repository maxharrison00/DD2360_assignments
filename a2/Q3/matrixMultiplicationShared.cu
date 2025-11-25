#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Helper to check CUDA errors
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
     printf("Cuda failure %s:%d: '%s'\\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
     exit(EXIT_FAILURE);                                             \
 }                                                                 \
}

// base kernel
__global__ void gemm(float *a, float *b, float *c, int N, int M, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < P) {
        float value = 0.0f;
        for (int k = 0; k < M; k++) {
            value += a[row * M + k] * b[k * P + col];
        }
        c[row * P + col] = value;
    }
}

// tiled kernel with shared memory
__global__ void tiled_gemm(float *a, float *b, float *c, int N, int M, int P, int tileX, int tileY){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_mem[];
    float *As = s_mem;                 
    float *Bs = &s_mem[tileY * tileX]; 

    float sum = 0.0f;

    for (int m = 0; m < M; m += tileX) {
        if (row < N && (m + threadIdx.x) < M) {
            As[threadIdx.y * tileX + threadIdx.x] = a[row * M + (m + threadIdx.x)];
        } else {
            As[threadIdx.y * tileX + threadIdx.x] = 0.0f;
        }

        if ((m + threadIdx.y) < M && col < P) {
            Bs[threadIdx.y * tileX + threadIdx.x] = b[(m + threadIdx.y) * P + col];
        } else {
            Bs[threadIdx.y * tileX + threadIdx.x] = 0.0f;
        }

        __syncthreads();
        for (int i = 0; i < tileX; i++){
            sum += As[threadIdx.y * tileX + i] * Bs[i * tileY + threadIdx.x];
        }

        __syncthreads();
    }
    
    if (row < N && col < P) {
        c[row * P + col] = sum;
    }
}

// CPU reference for validation
void matrixMultiplicationCPU(float *a, float *b, float *c, int N, int M, int P) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            float val = 0.0f;
            for (int k = 0; k < M; k++) {
                val += a[i * M + k] * b[k * P + j];
            }
            c[i * P + j] = val;
        }
    }
}

// Helper to compare results
float checkCorrectness(float *ref, float *gpu, int N, int P) {
    float maxError = 0.0f;
    for (int i = 0; i < N * P; i++) maxError = fmax(maxError, fabs(ref[i] - gpu[i]));
    return maxError;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Usage: ./exec N M P\n");
        return 0;
    }

    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int p = atoi(argv[3]);
    
    printf("Input matrix dim: (%d x %d) * (%d x %d)\n", n, m, m, p);

    size_t bytes_A = n * m * sizeof(float);
    size_t bytes_B = m * p * sizeof(float);
    size_t bytes_C = n * p * sizeof(float);

    // Host allocation
    float *h_a = (float*)malloc(bytes_A);
    float *h_b = (float*)malloc(bytes_B);
    float *h_c_ref = (float*)malloc(bytes_C); // CPU result
    float *h_c_gpu = (float*)malloc(bytes_C); // GPU result

    // Random initialization
    for(int i=0; i<n*m; i++) h_a[i] = (float)(rand() % 100) / 100.0f;
    for(int i=0; i<m*p; i++) h_b[i] = (float)(rand() % 100) / 100.0f;

    // Device allocation
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes_A);
    cudaMalloc(&d_b, bytes_B);
    cudaMalloc(&d_c, bytes_C);

    cudaMemcpy(d_a, h_a, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_B, cudaMemcpyHostToDevice);

    // CPU REFERENCE (Optional for very large matrices)
    if (n <= 2048) {
        matrixMultiplicationCPU(h_a, h_b, h_c_ref, n, m, p);
        printf("CPU reference result: Done\n"); 
    } else {
        printf("CPU reference result: Skipped (Size too large)\n");
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float ms = 0.0f;

    // basic gemm
    dim3 block(32, 32);
    dim3 grid((p + 31) / 32, (n + 31) / 32);

    cudaEventRecord(start);
    gemm<<<grid, block>>>(d_a, d_b, d_c, n, m, p);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaMemcpy(h_c_gpu, d_c, bytes_C, cudaMemcpyDeviceToHost);
    float err = (n <= 2048) ? checkCorrectness(h_c_ref, h_c_gpu, n, p) : 0.0f;
    
    printf("\nCUDA gemm result:\n");
    printf("Error: %.2e\n", err);
    printf("timing: %.3f ms\n", ms);

    // tiled gemm with different tile sizes
    int testTiles[] = {32, 16, 8}; // The three requested tile sizes
    
    for(int i=0; i<3; i++) {
        int tSize = testTiles[i];

        dim3 dimBlock(tSize, tSize);
        dim3 dimGrid((p + tSize - 1) / tSize, (n + tSize - 1) / tSize);
        size_t sharedMem = 2 * tSize * tSize * sizeof(float);

        cudaMemset(d_c, 0, bytes_C);

        cudaEventRecord(start);
        tiled_gemm<<<dimGrid, dimBlock, sharedMem>>>(d_a, d_b, d_c, n, m, p, tSize, tSize);
        cudaEventRecord(stop);
        cudaCheckError();
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        cudaMemcpy(h_c_gpu, d_c, bytes_C, cudaMemcpyDeviceToHost);
        err = (n <= 2048) ? checkCorrectness(h_c_ref, h_c_gpu, n, p) : 0.0f;

        printf("\nCUDA tiled_gemm with tile [%d, %d] result:\n", tSize, tSize);
        printf("Error: %.2e\n", err);
        printf("timing: %.3f ms\n", ms);
    }
    
    cudaFree(d_a); 
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a); free(h_b); free(h_c_ref); free(h_c_gpu);
    return 0;
}