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
 * This application demonstrates how to use the CUDA API to use multiple GPUs,
 * with an emphasis on simple illustration of the techniques (not on performance).
 *
 * Note that in order to detect multiple GPUs in your system you have to disable
 * SLI in the nvidia control panel. Otherwise only one GPU is visible to the
 * application. On the other side, you can still extend your desktop to screens
 * attached to both GPUs.
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

#include "simpleMultiGPU.h"

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int MAX_GPU_COUNT = 32;
const int DATA_N        = 1048576 * 32;

////////////////////////////////////////////////////////////////////////////////
// Simple reduction kernel.
// Refer to the 'reduction' CUDA Sample describing
// reduction optimization strategies
////////////////////////////////////////////////////////////////////////////////
__global__ static void reduceKernel(float *d_Result, float *d_Input, int N)
{
    const int     tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadN = gridDim.x * blockDim.x;
    float sum = 0;

    for (int pos = tid; pos < N; pos += threadN)
        sum += d_Input[pos];

    d_Result[tid] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    //Solver config
    TGPUplan      plan[MAX_GPU_COUNT];

    //GPU reduction results
    float     h_SumGPU[MAX_GPU_COUNT];

    float sumGPU;
    double sumCPU, diff;

    int i, j, gpuBase, GPU_N;

    const int  BLOCK_N = 32;
    const int THREAD_N = 256;
    const int  ACCUM_N = BLOCK_N * THREAD_N;

    printf("Starting simpleMultiGPU\n");
    checkHIPErrors(hipGetDeviceCount(&GPU_N));

    if (GPU_N > MAX_GPU_COUNT)
    {
        GPU_N = MAX_GPU_COUNT;
    }

    printf("CUDA-capable device count: %i\n", GPU_N);

    printf("Generating input data...\n\n");

    //Subdividing input data across GPUs
    //Get data sizes for each GPU
    for (i = 0; i < GPU_N; i++)
    {
        plan[i].dataN = DATA_N / GPU_N;
    }

    //Take into account "odd" data sizes
    for (i = 0; i < DATA_N % GPU_N; i++)
    {
        plan[i].dataN++;
    }

    //Assign data ranges to GPUs
    gpuBase = 0;

    for (i = 0; i < GPU_N; i++)
    {
        plan[i].h_Sum = h_SumGPU + i;
        gpuBase += plan[i].dataN;
    }

    //Create streams for issuing GPU command asynchronously and allocate memory (GPU and System page-locked)
    for (i = 0; i < GPU_N; i++)
    {
        checkHIPErrors(hipSetDevice(i));
        checkHIPErrors(hipStreamCreate(&plan[i].stream));
        //Allocate memory
        checkHIPErrors(hipMalloc((void **)&plan[i].d_Data, plan[i].dataN * sizeof(float)));
        checkHIPErrors(hipMalloc((void **)&plan[i].d_Sum, ACCUM_N * sizeof(float)));
        checkHIPErrors(hipHostMalloc((void **)&plan[i].h_Sum_from_device, ACCUM_N * sizeof(float)));
        checkHIPErrors(hipHostMalloc((void **)&plan[i].h_Data, plan[i].dataN * sizeof(float)));

        for (j = 0; j < plan[i].dataN; j++)
        {
            plan[i].h_Data[j] = (float)rand() / (float)RAND_MAX;
        }
    }

    //Start timing and compute on GPU(s)
    printf("Computing with %d GPUs...\n", GPU_N);
    // create and start timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);

    // start the timer
    sdkStartTimer(&timer);

    //Copy data to GPU, launch the kernel and copy data back. All asynchronously
    for (i = 0; i < GPU_N; i++)
    {
        //Set device
        checkHIPErrors(hipSetDevice(i));

        //Copy input data from CPU
        checkHIPErrors(hipMemcpyAsync(plan[i].d_Data, plan[i].h_Data, plan[i].dataN * sizeof(float), hipMemcpyHostToDevice, plan[i].stream));

        //Perform GPU computations
        hipLaunchKernelGGL(reduceKernel, dim3(BLOCK_N), dim3(THREAD_N), 0, plan[i].stream, plan[i].d_Sum, plan[i].d_Data, plan[i].dataN);
        getLastHIPError("reduceKernel() execution failed.\n");

        //Read back GPU results
        checkHIPErrors(hipMemcpyAsync(plan[i].h_Sum_from_device, plan[i].d_Sum, ACCUM_N *sizeof(float), hipMemcpyDeviceToHost, plan[i].stream));
    }

    //Process GPU results
    for (i = 0; i < GPU_N; i++)
    {
        float sum;

        //Set device
        checkHIPErrors(hipSetDevice(i));

        //Wait for all operations to finish
        hipStreamSynchronize(plan[i].stream);

        //Finalize GPU reduction for current subvector
        sum = 0;

        for (j = 0; j < ACCUM_N; j++)
        {
            sum += plan[i].h_Sum_from_device[j];
        }

        *(plan[i].h_Sum) = (float)sum;

        //Shut down this GPU
        checkHIPErrors(hipHostFree(plan[i].h_Sum_from_device));
        checkHIPErrors(hipFree(plan[i].d_Sum));
        checkHIPErrors(hipFree(plan[i].d_Data));
        checkHIPErrors(hipStreamDestroy(plan[i].stream));
    }

    sumGPU = 0;

    for (i = 0; i < GPU_N; i++)
    {
        sumGPU += h_SumGPU[i];
    }

    sdkStopTimer(&timer);
    printf("  GPU Processing time: %f (ms)\n\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    // Compute on Host CPU
    printf("Computing with Host CPU...\n\n");

    sumCPU = 0;

    for (i = 0; i < GPU_N; i++)
    {
        for (j = 0; j < plan[i].dataN; j++)
        {
            sumCPU += plan[i].h_Data[j];
        }
    }

    // Compare GPU and CPU results
    printf("Comparing GPU and Host CPU results...\n");
    diff = fabs(sumCPU - sumGPU) / fabs(sumCPU);
    printf("  GPU sum: %f\n  CPU sum: %f\n", sumGPU, sumCPU);
    printf("  Relative difference: %E \n\n", diff);

    // Cleanup and shutdown
    for (i = 0; i < GPU_N; i++)
    {
        checkHIPErrors(hipSetDevice(i));
        checkHIPErrors(hipHostFree(plan[i].h_Data));
    }

    exit((diff < 1e-5) ? EXIT_SUCCESS : EXIT_FAILURE);
}
