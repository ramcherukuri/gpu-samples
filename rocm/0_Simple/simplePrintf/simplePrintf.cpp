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


// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <hip/hip_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_hip.h>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

__global__ void testKernel(int val)
{
    printf("[%d, %d]:\t\tValue is:%d\n",\
            blockIdx.y*gridDim.x+blockIdx.x,\
            threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
            val);
}

int main(int argc, char **argv)
{
    int devID;
    hipDeviceProp_t props;

    // This will pick the best possible CUDA capable device
    devID = findHIPDevice(argc, (const char **)argv);

    //Get GPU information
    checkHIPErrors(hipGetDevice(&devID));
    checkHIPErrors(hipGetDeviceProperties(&props, devID));
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           devID, props.name, props.major, props.minor);

    printf("printf() is called. Output:\n\n");

    //Kernel configuration, where a two-dimensional grid and
    //three-dimensional blocks are configured.
    dim3 dimGrid(2, 2);
    dim3 dimBlock(2, 2, 2);
    hipLaunchKernelGGL(testKernel, dim3(dimGrid), dim3(dimBlock), 0, 0, 10);
    hipDeviceSynchronize();

    return EXIT_SUCCESS;
}

