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

// System includes.
#include <stdio.h>
#include <iostream>

// STL.
#include <vector>

// CUDA runtime.
#include <hip/hip_runtime.h>

// Helper functions and utilities to work with CUDA.
#include <helper_functions.h>
#include <helper_hip.h>

// Device library includes.
#include "simpleDeviceLibrary.cuh"

using std::cout;
using std::endl;

using std::vector;

#define EPS 1e-5

typedef unsigned int uint;
typedef float(*deviceFunc)(float);

const char *sampleName = "simpleSeparateCompilation";

////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
// Static device pointers to __device__ functions.
__device__ deviceFunc dMultiplyByTwoPtr = multiplyByTwo;
__device__ deviceFunc dDivideByTwoPtr = divideByTwo;

////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////
//! Transforms vector.
//! Applies the __device__ function "f" to each element of the vector "v".
////////////////////////////////////////////////////////////////////////////////
__global__ void transformVector(float *v, deviceFunc f, uint size)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        v[tid] = (*f)(v[tid]);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, const char **argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    cout << sampleName << " starting..." << endl;

    runTest(argc, (const char **)argv);

    cout << sampleName << " completed, returned "
         << (testResult ? "OK" : "ERROR") << endl;

    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}


void runTest(int argc, const char **argv)
{
    try
    {

        // This will pick the best possible CUDA capable device.
        findHIPDevice(argc, (const char **) argv);

        // Create host vector.
        const uint kVectorSize = 1000;

        vector<float> hVector(kVectorSize);

        for (uint i = 0; i < kVectorSize; ++i)
        {
            hVector[i] = rand() / static_cast<float>(RAND_MAX);
        }

        // Create and populate device vector.
        float *dVector;
        checkHIPErrors(hipMalloc(&dVector, kVectorSize * sizeof(float)));

        checkHIPErrors(hipMemcpy(dVector,
                                   &hVector[0],
                                   kVectorSize * sizeof(float),
                                   hipMemcpyHostToDevice));

        // Kernel configuration, where a one-dimensional
        // grid and one-dimensional blocks are configured.
        const int nThreads = 1024;
        const int nBlocks = 1;

        dim3 dimGrid(nBlocks);
        dim3 dimBlock(nThreads);

        // Test library functions.
        deviceFunc hFunctionPtr;

        hipMemcpyFromSymbol(&hFunctionPtr, HIP_SYMBOL(dMultiplyByTwoPtr),
                             sizeof(deviceFunc));
        hipLaunchKernelGGL(transformVector, dim3(dimGrid), dim3(dimBlock), 0, 0, dVector, hFunctionPtr, kVectorSize);
        checkHIPErrors(hipGetLastError());

        hipMemcpyFromSymbol(&hFunctionPtr, HIP_SYMBOL(dDivideByTwoPtr),
                             sizeof(deviceFunc));
        hipLaunchKernelGGL(transformVector, dim3(dimGrid), dim3(dimBlock), 0, 0, dVector, hFunctionPtr, kVectorSize);
        checkHIPErrors(hipGetLastError());

        // Download results.
        vector<float> hResultVector(kVectorSize);

        checkHIPErrors(hipMemcpy(&hResultVector[0],
                                   dVector,
                                   kVectorSize *sizeof(float),
                                   hipMemcpyDeviceToHost));

        // Check results.
        for (int i = 0; i < kVectorSize; ++i)
        {
            if (fabs(hVector[i] - hResultVector[i]) > EPS)
            {
                cout << "Computations were incorrect..." << endl;
                testResult = false;
                break;
            }
        }

        // Free resources.
        if (dVector) checkHIPErrors(hipFree(dVector));
    }
    catch (...)
    {
        cout << "Error occured, exiting..." << endl;

        exit(EXIT_FAILURE);
    }
}

