#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#define gpuCheck(stmt)                                                         \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("ERROR. Failed to run stmt %s\n", #stmt);                         \
      break;                                                                   \
    }                                                                          \
  } while (0)

#define cublasCheck(stmt)                                                      \
  do {                                                                         \
    cublasStatus_t err = stmt;                                                 \
    if (err != CUBLAS_STATUS_SUCCESS) {                                        \
      printf("ERROR. Failed to run cuBLAS stmt %s\n", #stmt);                  \
      break;                                                                   \
    }                                                                          \
  } while (0)

#define cusparseCheck(stmt)                                                    \
  do {                                                                         \
    cusparseStatus_t err = stmt;                                               \
    if (err != CUSPARSE_STATUS_SUCCESS) {                                      \
      printf("ERROR. Failed to run cuSPARSE stmt %s\n", #stmt);                \
      break;                                                                   \
    }                                                                          \
  } while (0)

struct timeval t_start, t_end;
void cputimer_start() { gettimeofday(&t_start, 0); }
void cputimer_stop(const char *info) {
  gettimeofday(&t_end, 0);
  double time = (1000000.0 * (t_end.tv_sec - t_start.tv_sec) + t_end.tv_usec -
                 t_start.tv_usec);
  printf("Timing - %s. \t\tElapsed %.0f microseconds \n", info, time);
}

// Initialize the sparse matrix
void matrixInit(double *A, int *ArowPtr, int *AcolIndx, int dimX,
                double alpha) {
  double stencil[] = {1, -2, 1};
  size_t ptr = 0;
  ArowPtr[1] = ptr;
  for (int i = 1; i < (dimX - 1); ++i) {
    for (int k = 0; k < 3; ++k) {
      A[ptr] = stencil[k];
      AcolIndx[ptr++] = i + k - 1;
    }
    ArowPtr[i + 1] = ptr;
  }
  ArowPtr[dimX] = ptr;
}

int main(int argc, char **argv) {
  int device = 0;
  int dimX;
  int nsteps;
  double alpha = 0.4;
  double *temp;
  double *A;
  int *ARowPtr;
  int *AColIndx;
  int nzv;
  double *tmp;
  size_t bufferSize = 0;
  void *buffer = nullptr;
  int concurrentAccessQ;
  double zero = 0;
  double one = 1;
  double norm;
  double error;
  double tempLeft = 200.;
  double tempRight = 300.;

  cublasHandle_t cublasHandle;
  cusparseHandle_t cusparseHandle;
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;

  // SpMV timing
  cudaEvent_t spmv_start, spmv_stop;
  float spmv_time_ms = 0.0f;
  float spmv_total_time_ms = 0.0f;

  dimX = atoi(argv[1]);
  nsteps = atoi(argv[2]);

  printf("The X dimension of the grid is %d \n", dimX);
  printf("The number of time steps to perform is %d \n", nsteps);

  gpuCheck(cudaDeviceGetAttribute(&concurrentAccessQ,
                                  cudaDevAttrConcurrentManagedAccess, device));

  nzv = 3 * dimX - 6;

  // Unified Memory allocation
  gpuCheck(cudaMallocManaged(&temp, sizeof(double) * dimX));
  gpuCheck(cudaMallocManaged(&tmp, sizeof(double) * dimX));
  gpuCheck(cudaMallocManaged(&A, sizeof(double) * nzv));
  gpuCheck(cudaMallocManaged(&ARowPtr, sizeof(int) * (dimX + 1)));
  gpuCheck(cudaMallocManaged(&AColIndx, sizeof(int) * nzv));

  if (concurrentAccessQ) {
    gpuCheck(
        cudaMemPrefetchAsync(temp, sizeof(double) * dimX, cudaCpuDeviceId));
    gpuCheck(cudaMemPrefetchAsync(tmp, sizeof(double) * dimX, cudaCpuDeviceId));
    gpuCheck(cudaMemPrefetchAsync(A, sizeof(double) * nzv, cudaCpuDeviceId));
    gpuCheck(cudaMemPrefetchAsync(ARowPtr, sizeof(int) * (dimX + 1),
                                  cudaCpuDeviceId));
    gpuCheck(
        cudaMemPrefetchAsync(AColIndx, sizeof(int) * nzv, cudaCpuDeviceId));
  }

  matrixInit(A, ARowPtr, AColIndx, dimX, alpha);

  memset(temp, 0, sizeof(double) * dimX);
  temp[0] = tempLeft;
  temp[dimX - 1] = tempRight;

  if (concurrentAccessQ) {
    gpuCheck(cudaMemPrefetchAsync(temp, sizeof(double) * dimX, device));
    gpuCheck(cudaMemPrefetchAsync(tmp, sizeof(double) * dimX, device));
    gpuCheck(cudaMemPrefetchAsync(A, sizeof(double) * nzv, device));
    gpuCheck(cudaMemPrefetchAsync(ARowPtr, sizeof(int) * (dimX + 1), device));
    gpuCheck(cudaMemPrefetchAsync(AColIndx, sizeof(int) * nzv, device));
  }

  cublasCheck(cublasCreate(&cublasHandle));
  cusparseCheck(cusparseCreate(&cusparseHandle));
  cublasCheck(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));

  cusparseCheck(cusparseCreateCsr(&matA, dimX, dimX, nzv, ARowPtr, AColIndx, A,
                                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

  cusparseCheck(cusparseCreateDnVec(&vecX, dimX, temp, CUDA_R_64F));
  cusparseCheck(cusparseCreateDnVec(&vecY, dimX, tmp, CUDA_R_64F));

  cusparseCheck(cusparseSpMV_bufferSize(
      cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecX, &zero,
      vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

  gpuCheck(cudaMalloc(&buffer, bufferSize));

  gpuCheck(cudaEventCreate(&spmv_start));
  gpuCheck(cudaEventCreate(&spmv_stop));

  for (int it = 0; it < nsteps; ++it) {

    gpuCheck(cudaEventRecord(spmv_start));

    cusparseCheck(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &one, matA, vecX, &zero, vecY, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    gpuCheck(cudaEventRecord(spmv_stop));
    gpuCheck(cudaEventSynchronize(spmv_stop));
    gpuCheck(cudaEventElapsedTime(&spmv_time_ms, spmv_start, spmv_stop));
    spmv_total_time_ms += spmv_time_ms;

    cublasCheck(cublasDaxpy(cublasHandle, dimX, &alpha, tmp, 1, temp, 1));
    cublasCheck(cublasDnrm2(cublasHandle, dimX, temp, 1, &norm));

    if (norm < 1e-4)
      break;
  }

  printf("Total SpMV time: %.6f seconds\n", spmv_total_time_ms / 1000.0);

  thrust::device_ptr<double> thrustPtr(tmp);
  thrust::sequence(thrustPtr, thrustPtr + dimX, tempLeft,
                   (tempRight - tempLeft) / (dimX - 1));

  one = -1;
  cublasCheck(cublasDaxpy(cublasHandle, dimX, &one, temp, 1, tmp, 1));
  cublasCheck(cublasDnrm2(cublasHandle, dimX, tmp, 1, &norm));

  error = norm;
  cublasCheck(cublasDnrm2(cublasHandle, dimX, temp, 1, &norm));
  error /= norm;

  printf("The relative error of the approximation is %f\n", error);

  gpuCheck(cudaEventDestroy(spmv_start));
  gpuCheck(cudaEventDestroy(spmv_stop));

  cusparseDestroySpMat(matA);
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
  cusparseDestroy(cusparseHandle);
  cublasDestroy(cublasHandle);

  gpuCheck(cudaFree(temp));
  gpuCheck(cudaFree(tmp));
  gpuCheck(cudaFree(A));
  gpuCheck(cudaFree(ARowPtr));
  gpuCheck(cudaFree(AColIndx));
  gpuCheck(cudaFree(buffer));

  return 0;
}
