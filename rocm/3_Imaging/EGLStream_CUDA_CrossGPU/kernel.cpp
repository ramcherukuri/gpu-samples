#include "hip/hip_runtime.h"
/*
 * cuda_consumer.c
 *
 * Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

//
// DESCRIPTION:   Simple CUDA consumer rendering sample app
//

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>
#include "eglstrm_common.h"

extern bool isCrossDevice;

__device__ static unsigned int numErrors = 0, errorFound = 0;
__device__ void checkProducerDataGPU(char *data, int size, char expectedVal, int frameNumber)
{
    if ((data[blockDim.x * blockIdx.x + threadIdx.x] != expectedVal) && (!errorFound)) {
        printf("Producer FOUND:%d expected: %d at %d for trial %d %d\n", data[blockDim.x * blockIdx.x + threadIdx.x], expectedVal, (blockDim.x * blockIdx.x + threadIdx.x), frameNumber, numErrors);
        numErrors++;
        errorFound = 1;
        return;
    }
}

__device__ void checkConsumerDataGPU(char *data, int size, char expectedVal, int frameNumber)
{
    if ((data[blockDim.x * blockIdx.x + threadIdx.x] != expectedVal) && (!errorFound)) {
        printf("Consumer FOUND:%d expected: %d at %d for trial %d %d\n", data[blockDim.x * blockIdx.x + threadIdx.x], expectedVal, (blockDim.x * blockIdx.x + threadIdx.x), frameNumber, numErrors);
        numErrors++;
        errorFound = 1;
        return;
    }
}

__global__ void writeDataToBuffer(char *pSrc, char newVal)
{
    pSrc[blockDim.x * blockIdx.x + threadIdx.x] = newVal;
}

__global__ void testKernelConsumer(char *pSrc, char size, char expectedVal, char newVal, int frameNumber)
{
    checkConsumerDataGPU(pSrc, size, expectedVal, frameNumber);
}


__global__ void testKernelProducer(char *pSrc, char size, char expectedVal, char newVal, int frameNumber)
{
    checkProducerDataGPU(pSrc, size, expectedVal, frameNumber);
}
__global__ void getNumErrors(int *numErr) {
        *numErr = numErrors;
}

extern "C" hipError_t cudaProducer_filter(hipStream_t pStream, char *pSrc, int width, int height, char expectedVal, char newVal, int frameNumber)
{
    // in case where consumer is on dgpu and producer is on igpu when return is called
    // the frame is not copied back to igpu. So the consumer changes is not visible to
    // producer
    if (isCrossDevice == 0) {
        hipLaunchKernelGGL(testKernelProducer, dim3((width * height)/1024), dim3(1024), 1, pStream, pSrc, width * height, expectedVal, newVal, frameNumber);
    }
    hipLaunchKernelGGL(writeDataToBuffer, dim3((width*height)/1024), dim3(1024), 1, pStream, pSrc, newVal);
    return hipSuccess;
};

extern "C" hipError_t cudaConsumer_filter(hipStream_t cStream, char *pSrc, int width, int height, char expectedVal, char newVal, int frameNumber)
{
    hipLaunchKernelGGL(testKernelConsumer, dim3((width * height)/1024), dim3(1024), 1, cStream, pSrc, width * height, expectedVal, newVal, frameNumber);
    hipLaunchKernelGGL(writeDataToBuffer, dim3((width*height)/1024), dim3(1024), 1, cStream, pSrc, newVal);
    return hipSuccess;
};

extern "C" hipError_t cudaGetValueMismatch()
{
    int numErr_h;
    int *numErr_d = NULL;
    hipError_t err = hipSuccess;
    err = hipMalloc(&numErr_d, sizeof(int));
    if (err != hipSuccess) {
        printf("Cuda Main: hipMalloc failed with %s\n",hipGetErrorString(err));
        return err;
    }
    hipLaunchKernelGGL(getNumErrors, dim3(1), dim3(1), 0, 0, numErr_d);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("Cuda Main: hipDeviceSynchronize failed with %s\n",hipGetErrorString(err));
    }
    err = hipMemcpy(&numErr_h, numErr_d, sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        printf("Cuda Main: hipMemcpy failed with %s\n",hipGetErrorString(err));
        hipFree(numErr_d);
        return err;
    }
    err = hipFree(numErr_d);
    if (err != hipSuccess) {
        printf("Cuda Main: hipFree failed with %s\n",hipGetErrorString(err));
        return err;
    }
    if (numErr_h > 0) {
        return hipErrorUnknown;
    }
    return hipSuccess;
}
