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
 * This sample demonstrates a combination of Peer-to-Peer (P2P) and
 * Unified Virtual Address Space (UVA) features new to SDK 4.0
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>

// CUDA includes
#include <hip/hip_runtime.h>

// includes, project
#include <helper_hip.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

__global__ void SimpleKernel(float *src, float *dst)
{
    // Just a dummy kernel, doing enough for us to verify that everything
    // worked
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx] * 2.0f;
}

inline bool IsAppBuiltAs64()
{
    return sizeof(void*) == 8;
}

int main(int argc, char **argv)
{
    printf("[%s] - Starting...\n", argv[0]);

    if (!IsAppBuiltAs64())
    {
        printf("%s is only supported with on 64-bit OSs and the application must be built as a 64-bit target.  Test is being waived.\n", argv[0]);
        exit(EXIT_WAIVED);
    }

    // Number of GPUs
    printf("Checking for multiple GPUs...\n");
    int gpu_n;
    checkHIPErrors(hipGetDeviceCount(&gpu_n));
    printf("CUDA-capable device count: %i\n", gpu_n);

    if (gpu_n < 2)
    {
        printf("Two or more GPUs with Peer-to-Peer access capability are required for %s.\n", argv[0]);
        printf("Waiving test.\n");
        exit(EXIT_WAIVED);
    }

    // Query device properties
    hipDeviceProp_t prop[64];
    int gpuid[2]; // we want to find the first two GPU's that can support P2P

    for (int i=0; i < gpu_n; i++)
    {
        checkHIPErrors(hipGetDeviceProperties(&prop[i], i));
    }
    // Check possibility for peer access
    printf("\nChecking GPU(s) for support of peer to peer memory access...\n");

    int can_access_peer;
    int p2pCapableGPUs[2]; // We take only 1 pair of P2P capable GPUs
    p2pCapableGPUs[0] = p2pCapableGPUs[1] = -1;

    // Show all the combinations of supported P2P GPUs
    for (int i = 0; i < gpu_n; i++)
    {
        for (int j = 0; j < gpu_n; j++)
        {
            if (i == j)
            {
                continue;
            }
            checkHIPErrors(hipDeviceCanAccessPeer(&can_access_peer, i, j));
            printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n", prop[i].name, i,
                           prop[j].name, j, can_access_peer ? "Yes" : "No");
            if (can_access_peer && p2pCapableGPUs[0] == -1)
            {
                    p2pCapableGPUs[0] = i;
                    p2pCapableGPUs[1] = j;
            }
        }
    }

    if (p2pCapableGPUs[0] == -1 || p2pCapableGPUs[1] == -1)
    {
        printf("Two or more GPUs with Peer-to-Peer access capability are required for %s.\n", argv[0]);
        printf("Peer to Peer access is not available amongst GPUs in the system, waiving test.\n");

        exit(EXIT_WAIVED);
    }

    // Use first pair of p2p capable GPUs detected.
    gpuid[0] = p2pCapableGPUs[0];
    gpuid[1] = p2pCapableGPUs[1];

    // Enable peer access
    printf("Enabling peer access between GPU%d and GPU%d...\n", gpuid[0], gpuid[1]);
    checkHIPErrors(hipSetDevice(gpuid[0]));
    checkHIPErrors(hipDeviceEnablePeerAccess(gpuid[1], 0));
    checkHIPErrors(hipSetDevice(gpuid[1]));
    checkHIPErrors(hipDeviceEnablePeerAccess(gpuid[0], 0));

    // Allocate buffers
    const size_t buf_size = 1024 * 1024 * 16 * sizeof(float);
    printf("Allocating buffers (%iMB on GPU%d, GPU%d and CPU Host)...\n", int(buf_size / 1024 / 1024), gpuid[0], gpuid[1]);
    checkHIPErrors(hipSetDevice(gpuid[0]));
    float *g0;
    checkHIPErrors(hipMalloc(&g0, buf_size));
    checkHIPErrors(hipSetDevice(gpuid[1]));
    float *g1;
    checkHIPErrors(hipMalloc(&g1, buf_size));
    float *h0;
    checkHIPErrors(hipHostMalloc(&h0, buf_size)); // Automatically portable with UVA

    // Create CUDA event handles
    printf("Creating event handles...\n");
    hipEvent_t start_event, stop_event;
    float time_memcpy;
    int eventflags = hipEventBlockingSync;
    checkHIPErrors(hipEventCreateWithFlags(&start_event, eventflags));
    checkHIPErrors(hipEventCreateWithFlags(&stop_event, eventflags));

    // P2P memcopy() benchmark
    checkHIPErrors(hipEventRecord(start_event, 0));

    for (int i=0; i<100; i++)
    {
        // With UVA we don't need to specify source and target devices, the
        // runtime figures this out by itself from the pointers
        // Ping-pong copy between GPUs
        if (i % 2 == 0)
        {
            checkHIPErrors(hipMemcpy(g1, g0, buf_size, hipMemcpyDefault));
        }
        else
        {
            checkHIPErrors(hipMemcpy(g0, g1, buf_size, hipMemcpyDefault));
        }
    }

    checkHIPErrors(hipEventRecord(stop_event, 0));
    checkHIPErrors(hipEventSynchronize(stop_event));
    checkHIPErrors(hipEventElapsedTime(&time_memcpy, start_event, stop_event));
    printf("hipMemcpyPeer / hipMemcpy between GPU%d and GPU%d: %.2fGB/s\n", gpuid[0], gpuid[1],
           (1.0f / (time_memcpy / 1000.0f)) * ((100.0f * buf_size)) / 1024.0f / 1024.0f / 1024.0f);

    // Prepare host buffer and copy to GPU 0
    printf("Preparing host buffer and memcpy to GPU%d...\n", gpuid[0]);

    for (int i=0; i<buf_size / sizeof(float); i++)
    {
        h0[i] = float(i % 4096);
    }

    checkHIPErrors(hipSetDevice(gpuid[0]));
    checkHIPErrors(hipMemcpy(g0, h0, buf_size, hipMemcpyDefault));

    // Kernel launch configuration
    const dim3 threads(512, 1);
    const dim3 blocks((buf_size / sizeof(float)) / threads.x, 1);

    // Run kernel on GPU 1, reading input from the GPU 0 buffer, writing
    // output to the GPU 1 buffer
    printf("Run kernel on GPU%d, taking source data from GPU%d and writing to GPU%d...\n",
           gpuid[1], gpuid[0], gpuid[1]);
    checkHIPErrors(hipSetDevice(gpuid[1]));
    hipLaunchKernelGGL(SimpleKernel, dim3(blocks), dim3(threads), 0, 0, g0, g1);

    checkHIPErrors(hipDeviceSynchronize());

    // Run kernel on GPU 0, reading input from the GPU 1 buffer, writing
    // output to the GPU 0 buffer
    printf("Run kernel on GPU%d, taking source data from GPU%d and writing to GPU%d...\n",
           gpuid[0], gpuid[1], gpuid[0]);
    checkHIPErrors(hipSetDevice(gpuid[0]));
    hipLaunchKernelGGL(SimpleKernel, dim3(blocks), dim3(threads), 0, 0, g1, g0);

    checkHIPErrors(hipDeviceSynchronize());

    // Copy data back to host and verify
    printf("Copy data back to host from GPU%d and verify results...\n", gpuid[0]);
    checkHIPErrors(hipMemcpy(h0, g0, buf_size, hipMemcpyDefault));

    int error_count = 0;

    for (int i=0; i<buf_size / sizeof(float); i++)
    {
        // Re-generate input data and apply 2x '* 2.0f' computation of both
        // kernel runs
        if (h0[i] != float(i % 4096) * 2.0f * 2.0f)
        {
            printf("Verification error @ element %i: val = %f, ref = %f\n", i, h0[i], (float(i%4096)*2.0f*2.0f));

            if (error_count++ > 10)
            {
                break;
            }
        }
    }

    // Disable peer access (also unregisters memory for non-UVA cases)
    printf("Disabling peer access...\n");
    checkHIPErrors(hipSetDevice(gpuid[0]));
    checkHIPErrors(hipDeviceDisablePeerAccess(gpuid[1]));
    checkHIPErrors(hipSetDevice(gpuid[1]));
    checkHIPErrors(hipDeviceDisablePeerAccess(gpuid[0]));

    // Cleanup and shutdown
    printf("Shutting down...\n");
    checkHIPErrors(hipEventDestroy(start_event));
    checkHIPErrors(hipEventDestroy(stop_event));
    checkHIPErrors(hipSetDevice(gpuid[0]));
    checkHIPErrors(hipFree(g0));
    checkHIPErrors(hipSetDevice(gpuid[1]));
    checkHIPErrors(hipFree(g1));
    checkHIPErrors(hipHostFree(h0));

    for (int i=0; i<gpu_n; i++)
    {
        checkHIPErrors(hipSetDevice(i));
    }

    if (error_count != 0)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        printf("Test passed\n");
        exit(EXIT_SUCCESS);
    }
}

