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

void matrixMultiplication(float *a, float *b, float *c, int N, int M, int P) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < P; j++) {
      c[i * P + j] = 0;
      for (int k = 0; k < M; k++) {
        c[i * P + j] += a[i * M + k] * b[k * P + j];
      }
    }
  }
}

__global__ void multiplication(float *a, float *b, float *c, int N, int M, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < P) {
        float value = 0;
        for (int k = 0; k < M; k++) {
            value += a[row * M + k] * b[k * P + col];
        }
        c[row * P + col] = value;
    }
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        return 0;
    }

    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int p = atoi(argv[3]);

    //@@ 1. Allocate host memory
    float *h_a = (float*) malloc(n * m * sizeof(float));
    float *h_b = (float*) malloc(m * p * sizeof(float));
    float *h_c = (float*) malloc(n * p * sizeof(float));
    float *result = (float*) malloc(n * p * sizeof(float));
    float *d_a, *d_b, *d_c;

    //@@ 2. Allocate device memory
    cudaMalloc(&d_a, n * m * sizeof(float));
    cudaMalloc(&d_b, m * p * sizeof(float));
    cudaMalloc(&d_c, n * p * sizeof(float));

    //@@ 3. Initialize host memory
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            h_a[i * m + j] = ((float)(i * m + j)) / (n * m - 1);
        }
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            h_b[i * p + j] = ((float)(i * p + j)) / (m * p - 1);
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            h_c[i * p + j] = 0.0f;
        }
    }

    //@@ 4. Copy data from host to device
    cudaMemcpy(d_a, h_a, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, m * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, n * p * sizeof(float), cudaMemcpyHostToDevice);

    //@@ 5. Initialize thread block and thread grid
    dim3 blockSize(16, 16);
    dim3 gridSize((p + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    //@@ 6. Invoke the CUDA kernel
    multiplication<<<gridSize, blockSize>>>(d_a, d_b, d_c, n, m, p);

    //@@ 7. Copy results from device to host
    cudaMemcpy(result, d_c, n * p * sizeof(float), cudaMemcpyDeviceToHost);

    //@@ 8. Compute reference solution on CPU
    matrixMultiplication(h_a, h_b, h_c, n, m, p);

    float error = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            error += h_c[i * p + j] - result[i * p + j];
        }
    }
    printf("Total error between CPU and GPU versions is %f\n", error);

    //@@ 9. Free host memory
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