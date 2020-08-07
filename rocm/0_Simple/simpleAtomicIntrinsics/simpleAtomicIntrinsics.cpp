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

/* A simple program demonstrating trivial use of global memory atomic
 * device functions (atomic*() functions).
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <hip/hip_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes hip/hip_runtime.h and hip/hip_runtime_api.h

// CUDA helper functions
#include <helper_hip.h>         // helper functions for CUDA error check

// Includes, kernels
#include "simpleAtomicIntrinsics_kernel.cuh"

const char *sampleName = "simpleAtomicIntrinsics";

////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

extern "C" bool computeGold(int *gpuData, const int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("%s starting...\n", sampleName);

    runTest(argc, argv);

    printf("%s completed, returned %s\n",
           sampleName,
           testResult ? "OK" : "ERROR!");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv)
{
    hipStream_t stream;
    // This will pick the best possible CUDA capable device
    findHIPDevice(argc, (const char **)argv);

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    unsigned int numThreads = 256;
    unsigned int numBlocks = 64;
    unsigned int numData = 11;
    unsigned int memSize = sizeof(int) * numData;

    //allocate mem for the result on host side
    int *hOData = (int *) malloc(memSize);

    //initialize the memory
    for (unsigned int i = 0; i < numData; i++)
        hOData[i] = 0;

    //To make the AND and XOR tests generate something other than 0...
    hOData[8] = hOData[10] = 0xff;

    checkHIPErrors(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    // allocate device memory for result
    int *dOData;
    checkHIPErrors(hipMalloc((void **) &dOData, memSize));
    // copy host memory to device to initialize to zero
    checkHIPErrors(hipMemcpyAsync(dOData, hOData, memSize,
                                    hipMemcpyHostToDevice, stream));

    // execute the kernel
    hipLaunchKernelGGL(testKernel, dim3(numBlocks), dim3(numThreads), 0, stream, dOData);
    getLastHIPError("Kernel execution failed");

    //Copy result from device to host
    checkHIPErrors(hipMemcpyAsync(hOData, dOData, memSize,
                                    hipMemcpyDeviceToHost, stream));
    checkHIPErrors(hipStreamSynchronize(stream));

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    // Compute reference solution
    testResult = computeGold(hOData, numThreads * numBlocks);

    // Cleanup memory
    free(hOData);
    checkHIPErrors(hipFree(dOData));
}
