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

#include "FDTD3dGPU.h"

#include <iostream>
#include <algorithm>
#include <helper_functions.h>
#include <helper_hip.h>

#include "FDTD3dGPUKernel.cuh"

bool getTargetDeviceGlobalMemSize(memsize_t *result, const int argc, const char **argv)
{
    int               deviceCount  = 0;
    int               targetDevice = 0;
    size_t            memsize      = 0;

    // Get the number of CUDA enabled GPU devices
    printf(" hipGetDeviceCount\n");
    checkHIPErrors(hipGetDeviceCount(&deviceCount));

    // Select target device (device 0 by default)
    targetDevice = findHIPDevice(argc, (const char **)argv);

    // Query target device for maximum memory allocation
    printf(" hipGetDeviceProperties\n");
    struct hipDeviceProp_t deviceProp;
    checkHIPErrors(hipGetDeviceProperties(&deviceProp, targetDevice));

    memsize = deviceProp.totalGlobalMem;

    // Save the result
    *result = (memsize_t)memsize;
    return true;
}

bool fdtdGPU(float *output, const float *input, const float *coeff, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps, const int argc, const char **argv)
{
    const int         outerDimx  = dimx + 2 * radius;
    const int         outerDimy  = dimy + 2 * radius;
    const int         outerDimz  = dimz + 2 * radius;
    const size_t      volumeSize = outerDimx * outerDimy * outerDimz;
    int               deviceCount  = 0;
    int               targetDevice = 0;
    float            *bufferOut    = 0;
    float            *bufferIn     = 0;
    dim3              dimBlock;
    dim3              dimGrid;

    // Ensure that the inner data starts on a 128B boundary
    const int padding = (128 / sizeof(float)) - radius;
    const size_t paddedVolumeSize = volumeSize + padding;

#ifdef GPU_PROFILING
    hipEvent_t profileStart = 0;
    hipEvent_t profileEnd   = 0;
    const int profileTimesteps = timesteps - 1;

    if (profileTimesteps < 1)
    {
        printf(" cannot profile with fewer than two timesteps (timesteps=%d), profiling is disabled.\n", timesteps);
    }

#endif

    // Check the radius is valid
    if (radius != RADIUS)
    {
        printf("radius is invalid, must be %d - see kernel for details.\n", RADIUS);
        exit(EXIT_FAILURE);
    }

    // Get the number of CUDA enabled GPU devices
    checkHIPErrors(hipGetDeviceCount(&deviceCount));

    // Select target device (device 0 by default)
    targetDevice = findHIPDevice(argc, (const char **)argv);

    checkHIPErrors(hipSetDevice(targetDevice));

    // Allocate memory buffers
    checkHIPErrors(hipMalloc((void **)&bufferOut, paddedVolumeSize * sizeof(float)));
    checkHIPErrors(hipMalloc((void **)&bufferIn, paddedVolumeSize * sizeof(float)));

    // Check for a command-line specified block size
    int userBlockSize;

    if (checkCmdLineFlag(argc, (const char **)argv, "block-size"))
    {
        userBlockSize = getCmdLineArgumentInt(argc, argv, "block-size");
        // Constrain to a multiple of k_blockDimX
        userBlockSize = (userBlockSize / k_blockDimX * k_blockDimX);

        // Constrain within allowed bounds
        userBlockSize = MIN(MAX(userBlockSize, k_blockSizeMin), k_blockSizeMax);
    }
    else
    {
        userBlockSize = k_blockSizeMax;
    }

    // Check the device limit on the number of threads
    struct hipFuncAttributes funcAttrib;
    checkHIPErrors(hipFuncGetAttributes(&funcAttrib, reinterpret_cast<const void*>(FiniteDifferencesKernel)));

    userBlockSize = MIN(userBlockSize, funcAttrib.maxThreadsPerBlock);

    // Set the block size
    dimBlock.x = k_blockDimX;
    // Visual Studio 2005 does not like std::min
    //    dimBlock.y = std::min<size_t>(userBlockSize / k_blockDimX, (size_t)k_blockDimMaxY);
    dimBlock.y = ((userBlockSize / k_blockDimX) < (size_t)k_blockDimMaxY) ? (userBlockSize / k_blockDimX) : (size_t)k_blockDimMaxY;
    dimGrid.x  = (unsigned int)ceil((float)dimx / dimBlock.x);
    dimGrid.y  = (unsigned int)ceil((float)dimy / dimBlock.y);
    printf(" set block size to %dx%d\n", dimBlock.x, dimBlock.y);
    printf(" set grid size to %dx%d\n", dimGrid.x, dimGrid.y);

    // Check the block size is valid
    if (dimBlock.x < RADIUS || dimBlock.y < RADIUS)
    {
        printf("invalid block size, x (%d) and y (%d) must be >= radius (%d).\n", dimBlock.x, dimBlock.y, RADIUS);
        exit(EXIT_FAILURE);
    }

    // Copy the input to the device input buffer
    checkHIPErrors(hipMemcpy(bufferIn + padding, input, volumeSize * sizeof(float), hipMemcpyHostToDevice));

    // Copy the input to the device output buffer (actually only need the halo)
    checkHIPErrors(hipMemcpy(bufferOut + padding, input, volumeSize * sizeof(float), hipMemcpyHostToDevice));

    // Copy the coefficients to the device coefficient buffer
    checkHIPErrors(hipMemcpyToSymbol(HIP_SYMBOL(stencil), (void *)coeff, (radius + 1) * sizeof(float)));


#ifdef GPU_PROFILING

    // Create the events
    checkHIPErrors(hipEventCreate(&profileStart));
    checkHIPErrors(hipEventCreate(&profileEnd));

#endif

    // Execute the FDTD
    float *bufferSrc = bufferIn + padding;
    float *bufferDst = bufferOut + padding;
    printf(" GPU FDTD loop\n");


#ifdef GPU_PROFILING
    // Enqueue start event
    checkHIPErrors(hipEventRecord(profileStart, 0));
#endif

    for (int it = 0 ; it < timesteps ; it++)
    {
        printf("\tt = %d ", it);

        // Launch the kernel
        printf("launch kernel\n");
        hipLaunchKernelGGL(FiniteDifferencesKernel, dim3(dimGrid), dim3(dimBlock), 0, 0, bufferDst, bufferSrc, dimx, dimy, dimz);

        // Toggle the buffers
        // Visual Studio 2005 does not like std::swap
        //    std::swap<float *>(bufferSrc, bufferDst);
        float *tmp = bufferDst;
        bufferDst = bufferSrc;
        bufferSrc = tmp;
    }

    printf("\n");

#ifdef GPU_PROFILING
    // Enqueue end event
    checkHIPErrors(hipEventRecord(profileEnd, 0));
#endif

    // Wait for the kernel to complete
    checkHIPErrors(hipDeviceSynchronize());

    // Read the result back, result is in bufferSrc (after final toggle)
    checkHIPErrors(hipMemcpy(output, bufferSrc, volumeSize * sizeof(float), hipMemcpyDeviceToHost));

    // Report time
#ifdef GPU_PROFILING
    float elapsedTimeMS = 0;

    if (profileTimesteps > 0)
    {
        checkHIPErrors(hipEventElapsedTime(&elapsedTimeMS, profileStart, profileEnd));
    }

    if (profileTimesteps > 0)
    {
        // Convert milliseconds to seconds
        double elapsedTime    = elapsedTimeMS * 1.0e-3;
        double avgElapsedTime = elapsedTime / (double)profileTimesteps;
        // Determine number of computations per timestep
        size_t pointsComputed = dimx * dimy * dimz;
        // Determine throughput
        double throughputM    = 1.0e-6 * (double)pointsComputed / avgElapsedTime;
        printf("FDTD3d, Throughput = %.4f MPoints/s, Time = %.5f s, Size = %u Points, NumDevsUsed = %u, Blocksize = %u\n",
               throughputM, avgElapsedTime, pointsComputed, 1, dimBlock.x * dimBlock.y);
    }

#endif

    // Cleanup
    if (bufferIn)
    {
        checkHIPErrors(hipFree(bufferIn));
    }

    if (bufferOut)
    {
        checkHIPErrors(hipFree(bufferOut));
    }

#ifdef GPU_PROFILING

    if (profileStart)
    {
        checkHIPErrors(hipEventDestroy(profileStart));
    }

    if (profileEnd)
    {
        checkHIPErrors(hipEventDestroy(profileEnd));
    }

#endif
    return true;
}
