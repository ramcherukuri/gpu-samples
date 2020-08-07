#include "hip/hip_runtime.h"
/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a conjugate gradient solver on GPU
 * using CUBLAS and CUSPARSE
 *
 */

// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas */
#include <hipblas.h>
#include <hip/hip_runtime.h>
#include <hipsparse.h>

#include <hip/hip_cooperative_groups.h>

// Utilities and system includes
#include <helper_hip.h>  // helper function CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples

namespace cg = cooperative_groups;

const char *sSDKname = "conjugateGradientCudaGraphs";

#ifndef WITH_GRAPH
#define WITH_GRAPH 1
#endif

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz) {
  I[0] = 0, J[0] = 0, J[1] = 1;
  val[0] = (float)rand() / RAND_MAX + 10.0f;
  val[1] = (float)rand() / RAND_MAX;
  int start;

  for (int i = 1; i < N; i++) {
    if (i > 1) {
      I[i] = I[i - 1] + 3;
    } else {
      I[1] = 2;
    }

    start = (i - 1) * 3 + 2;
    J[start] = i - 1;
    J[start + 1] = i;

    if (i < N - 1) {
      J[start + 2] = i + 1;
    }

    val[start] = val[start - 1];
    val[start + 1] = (float)rand() / RAND_MAX + 10.0f;

    if (i < N - 1) {
      val[start + 2] = (float)rand() / RAND_MAX;
    }
  }

  I[N] = nz;
}

__global__ void initVectors(float *rhs, float *x, int N) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = gid; i < N; i += gridDim.x * blockDim.x) {
    rhs[i] = 1.0;
    x[i] = 0.0;
  }
}

__global__ void r1_div_x(float *r1, float *r0, float *b) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid == 0) {
    b[0] = r1[0] / r0[0];
  }
}

__global__ void a_minus(float *a, float *na) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid == 0) {
    na[0] = -(a[0]);
  }
}

int main(int argc, char **argv) {
  int N = 0, nz = 0, *I = NULL, *J = NULL;
  float *val = NULL;
  const float tol = 1e-5f;
  const int max_iter = 10000;
  float *x;
  float *rhs;
  float r1;

  int *d_col, *d_row;
  float *d_val, *d_x;
  float *d_r, *d_p, *d_Ax;
  int k;
  float alpha, beta, alpham1;

  hipStream_t stream1, streamForGraph;

  // This will pick the best possible CUDA capable device
  hipDeviceProp_t deviceProp;
  int devID = findHIPDevice(argc, (const char **)argv);

  if (devID < 0) {
    printf("exiting...\n");
    exit(EXIT_SUCCESS);
  }

  checkHIPErrors(hipGetDeviceProperties(&deviceProp, devID));

  // Statistics about the GPU device
  printf(
      "> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
      deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

  /* Generate a random tridiagonal symmetric matrix in CSR format */
  N = 1048576;
  nz = (N - 2) * 3 + 4;
  I = (int *)malloc(sizeof(int) * (N + 1));
  J = (int *)malloc(sizeof(int) * nz);
  val = (float *)malloc(sizeof(float) * nz);
  genTridiag(I, J, val, N, nz);

  x = (float *)malloc(sizeof(float) * N);
  rhs = (float *)malloc(sizeof(float) * N);

  for (int i = 0; i < N; i++) {
    rhs[i] = 1.0;
    x[i] = 0.0;
  }

  /* Get handle to the CUBLAS context */
  hipblasHandle_t cublasHandle = 0;
  hipblasStatus_t hipblasStatus_t;
  hipblasStatus_t = hipblasCreate(&cublasHandle);

  checkHIPErrors(hipblasStatus_t);

  /* Get handle to the CUSPARSE context */
  hipsparseHandle_t cusparseHandle = 0;
  hipsparseStatus_t cusparseStatus;
  cusparseStatus = hipsparseCreate(&cusparseHandle);

  checkHIPErrors(cusparseStatus);

  checkHIPErrors(hipStreamCreate(&stream1));

  checkHIPErrors(hipMalloc((void **)&d_col, nz * sizeof(int)));
  checkHIPErrors(hipMalloc((void **)&d_row, (N + 1) * sizeof(int)));
  checkHIPErrors(hipMalloc((void **)&d_val, nz * sizeof(float)));
  checkHIPErrors(hipMalloc((void **)&d_x, N * sizeof(float)));
  checkHIPErrors(hipMalloc((void **)&d_r, N * sizeof(float)));
  checkHIPErrors(hipMalloc((void **)&d_p, N * sizeof(float)));
  checkHIPErrors(hipMalloc((void **)&d_Ax, N * sizeof(float)));

  float *d_r1, *d_r0, *d_dot, *d_a, *d_na, *d_b;
  checkHIPErrors(hipMalloc((void **)&d_r1, sizeof(float)));
  checkHIPErrors(hipMalloc((void **)&d_r0, sizeof(float)));
  checkHIPErrors(hipMalloc((void **)&d_dot, sizeof(float)));
  checkHIPErrors(hipMalloc((void **)&d_a, sizeof(float)));
  checkHIPErrors(hipMalloc((void **)&d_na, sizeof(float)));
  checkHIPErrors(hipMalloc((void **)&d_b, sizeof(float)));

  hipsparseMatDescr_t descr = 0;
  checkHIPErrors(hipsparseCreateMatDescr(&descr));

  checkHIPErrors(hipsparseSetMatType(descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
  checkHIPErrors(hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO));

  int numBlocks = 0, blockSize = 0;
  checkHIPErrors(
      hipOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, initVectors));

  checkHIPErrors(hipMemcpyAsync(d_col, J, nz * sizeof(int),
                                  hipMemcpyHostToDevice, stream1));
  checkHIPErrors(hipMemcpyAsync(d_row, I, (N + 1) * sizeof(int),
                                  hipMemcpyHostToDevice, stream1));
  checkHIPErrors(hipMemcpyAsync(d_val, val, nz * sizeof(float),
                                  hipMemcpyHostToDevice, stream1));

  hipLaunchKernelGGL(initVectors, dim3(numBlocks), dim3(blockSize), 0, stream1, d_r, d_x, N);

  alpha = 1.0;
  alpham1 = -1.0;
  beta = 0.0;

  checkHIPErrors(hipsparseSetStream(cusparseHandle, stream1));
  checkHIPErrors(
      hipsparseScsrmv(cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz,
                     &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax));

  checkHIPErrors(hipblasSetStream(cublasHandle, stream1));
  checkHIPErrors(hipblasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1));

  checkHIPErrors(
      hipblasSetPointerMode(cublasHandle, HIPBLAS_POINTER_MODE_DEVICE));
  checkHIPErrors(hipblasSdot(cublasHandle, N, d_r, 1, d_r, 1, d_r1));

  k = 1;
  // First Iteration when k=1 starts
  checkHIPErrors(hipblasScopy(cublasHandle, N, d_r, 1, d_p, 1));
  checkHIPErrors(
      hipsparseScsrmv(cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz,
                     &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax));

  checkHIPErrors(hipblasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, d_dot));

  hipLaunchKernelGGL(r1_div_x, dim3(1), dim3(1), 0, stream1, d_r1, d_dot, d_a);

  checkHIPErrors(hipblasSaxpy(cublasHandle, N, d_a, d_p, 1, d_x, 1));

  hipLaunchKernelGGL(a_minus, dim3(1), dim3(1), 0, stream1, d_a, d_na);

  checkHIPErrors(hipblasSaxpy(cublasHandle, N, d_na, d_Ax, 1, d_r, 1));

  checkHIPErrors(hipMemcpyAsync(d_r0, d_r1, sizeof(float),
                                  hipMemcpyDeviceToDevice, stream1));

  checkHIPErrors(hipblasSdot(cublasHandle, N, d_r, 1, d_r, 1, d_r1));

  checkHIPErrors(hipMemcpyAsync(&r1, d_r1, sizeof(float),
                                  hipMemcpyDeviceToHost, stream1));
  checkHIPErrors(hipStreamSynchronize(stream1));
  printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
  // First Iteration when k=1 ends
  k++;

#if WITH_GRAPH
  cudaGraph_t initGraph;
  checkHIPErrors(hipStreamCreate(&streamForGraph));
  checkHIPErrors(hipblasSetStream(cublasHandle, stream1));
  checkHIPErrors(hipsparseSetStream(cusparseHandle, stream1));
  checkHIPErrors(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

  hipLaunchKernelGGL(r1_div_x, dim3(1), dim3(1), 0, stream1, d_r1, d_r0, d_b);
  hipblasSetPointerMode(cublasHandle, HIPBLAS_POINTER_MODE_DEVICE);
  checkHIPErrors(hipblasSscal(cublasHandle, N, d_b, d_p, 1));
  hipblasSetPointerMode(cublasHandle, HIPBLAS_POINTER_MODE_HOST);
  checkHIPErrors(hipblasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1));
  hipblasSetPointerMode(cublasHandle, HIPBLAS_POINTER_MODE_DEVICE);

  checkHIPErrors(
      hipsparseSetPointerMode(cusparseHandle, HIPSPARSE_POINTER_MODE_HOST));
  checkHIPErrors(
      hipsparseScsrmv(cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz,
                     &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax));

  checkHIPErrors(hipMemsetAsync(d_dot, 0, sizeof(float), stream1));
  checkHIPErrors(hipblasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, d_dot));

  hipLaunchKernelGGL(r1_div_x, dim3(1), dim3(1), 0, stream1, d_r1, d_dot, d_a);

  checkHIPErrors(hipblasSaxpy(cublasHandle, N, d_a, d_p, 1, d_x, 1));

  hipLaunchKernelGGL(a_minus, dim3(1), dim3(1), 0, stream1, d_a, d_na);

  checkHIPErrors(hipblasSaxpy(cublasHandle, N, d_na, d_Ax, 1, d_r, 1));

  checkHIPErrors(hipMemcpyAsync(d_r0, d_r1, sizeof(float),
                                  hipMemcpyDeviceToDevice, stream1));
  checkHIPErrors(hipMemsetAsync(d_r1, 0, sizeof(float), stream1));

  checkHIPErrors(hipblasSdot(cublasHandle, N, d_r, 1, d_r, 1, d_r1));

  checkHIPErrors(hipMemcpyAsync((float *)&r1, d_r1, sizeof(float),
                                  hipMemcpyDeviceToHost, stream1));

  checkHIPErrors(cudaStreamEndCapture(stream1, &initGraph));
  cudaGraphExec_t graphExec;
  checkHIPErrors(cudaGraphInstantiate(&graphExec, initGraph, NULL, NULL, 0));
#endif

  checkHIPErrors(hipblasSetStream(cublasHandle, stream1));
  checkHIPErrors(hipsparseSetStream(cusparseHandle, stream1));

  while (r1 > tol * tol && k <= max_iter) {
#if WITH_GRAPH
    checkHIPErrors(cudaGraphLaunch(graphExec, streamForGraph));
    checkHIPErrors(hipStreamSynchronize(streamForGraph));
#else
    hipLaunchKernelGGL(r1_div_x, dim3(1), dim3(1), 0, stream1, d_r1, d_r0, d_b);
    hipblasSetPointerMode(cublasHandle, HIPBLAS_POINTER_MODE_DEVICE);
    checkHIPErrors(hipblasSscal(cublasHandle, N, d_b, d_p, 1));

    hipblasSetPointerMode(cublasHandle, HIPBLAS_POINTER_MODE_HOST);
    checkHIPErrors(hipblasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1));

    checkHIPErrors(hipsparseScsrmv(
        cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha,
        descr, d_val, d_row, d_col, d_p, &beta, d_Ax));

    hipblasSetPointerMode(cublasHandle, HIPBLAS_POINTER_MODE_DEVICE);
    checkHIPErrors(hipblasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, d_dot));

    hipLaunchKernelGGL(r1_div_x, dim3(1), dim3(1), 0, stream1, d_r1, d_dot, d_a);

    checkHIPErrors(hipblasSaxpy(cublasHandle, N, d_a, d_p, 1, d_x, 1));

    hipLaunchKernelGGL(a_minus, dim3(1), dim3(1), 0, stream1, d_a, d_na);
    checkHIPErrors(hipblasSaxpy(cublasHandle, N, d_na, d_Ax, 1, d_r, 1));

    checkHIPErrors(hipMemcpyAsync(d_r0, d_r1, sizeof(float),
                                    hipMemcpyDeviceToDevice, stream1));

    checkHIPErrors(hipblasSdot(cublasHandle, N, d_r, 1, d_r, 1, d_r1));
    checkHIPErrors(hipMemcpyAsync((float *)&r1, d_r1, sizeof(float),
                                    hipMemcpyDeviceToHost, stream1));
    checkHIPErrors(hipStreamSynchronize(stream1));
#endif
    printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
    k++;
  }

#if WITH_GRAPH
  checkHIPErrors(hipMemcpyAsync(x, d_x, N * sizeof(float),
                                  hipMemcpyDeviceToHost, streamForGraph));
  checkHIPErrors(hipStreamSynchronize(streamForGraph));
#else
  checkHIPErrors(hipMemcpyAsync(x, d_x, N * sizeof(float),
                                  hipMemcpyDeviceToHost, stream1));
  checkHIPErrors(hipStreamSynchronize(stream1));
#endif

  float rsum, diff, err = 0.0;

  for (int i = 0; i < N; i++) {
    rsum = 0.0;

    for (int j = I[i]; j < I[i + 1]; j++) {
      rsum += val[j] * x[J[j]];
    }

    diff = fabs(rsum - rhs[i]);

    if (diff > err) {
      err = diff;
    }
  }

#if WITH_GRAPH
  checkHIPErrors(cudaGraphExecDestroy(graphExec));
  checkHIPErrors(cudaGraphDestroy(initGraph));
  checkHIPErrors(hipStreamDestroy(streamForGraph));
#endif
  checkHIPErrors(hipStreamDestroy(stream1));
  hipsparseDestroy(cusparseHandle);
  hipblasDestroy(cublasHandle);

  free(I);
  free(J);
  free(val);
  free(x);
  free(rhs);
  hipFree(d_col);
  hipFree(d_row);
  hipFree(d_val);
  hipFree(d_x);
  hipFree(d_r);
  hipFree(d_p);
  hipFree(d_Ax);

  printf("Test Summary:  Error amount = %f\n", err);
  exit((k <= max_iter) ? 0 : 1);
}
