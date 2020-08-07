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
//
// This sample demonstrates dynamic global memory allocation through device C++ new and delete operators and virtual function declarations available with CUDA 4.0.

#include <stdio.h>

#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;
#include <helper_hip.h>

#include <stdlib.h>

#include <vector>
#include <algorithm>

const char *sSDKsample = "newdelete";

#include "container.hpp"

/////////////////////////////////////////////////////////////////////////////
//
// Kernels to allocate and instantiate Container objects on the device heap
//
////////////////////////////////////////////////////////////////////////////

__global__
void vectorCreate(Container<int> **g_container, int max_size)
{
    // The Vector object and the data storage are allocated in device heap memory.
    // This makes it persistent for the lifetime of the CUDA context.
    // The grid has only one thread as only a single object instance is needed.

    *g_container = new Vector<int>(max_size);
}


/////////////////////////////////////////////////////////////////////////////
//
// Kernels to fill and consume shared Container objects.
//
////////////////////////////////////////////////////////////////////////////


__global__
void containerFill(Container<int> **g_container)
{
    // All threads of the grid cooperatively populate the shared Container object with data.
    if (threadIdx.x == 0)
    {
        (*g_container)->push(blockIdx.x);
    }
}

__global__
void containerConsume(Container<int> **g_container, int *d_result)
{
    // All threads of the grid cooperatively consume the data from the shared Container object.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int v;

    if ((*g_container)->pop(v))
    {
        d_result[idx] = v;
    }
    else
    {
        d_result[idx] = -1;
    }
}


/////////////////////////////////////////////////////////////////////////////
//
// Kernel to delete shared Container objects.
//
////////////////////////////////////////////////////////////////////////////


__global__
void containerDelete(Container<int> **g_container)
{
    delete *g_container;
}


///////////////////////////////////////////////////////////////////////////////////////////
//
// Kernels to using of placement new to put shared Vector objects and data in shared memory
//
///////////////////////////////////////////////////////////////////////////////////////////


__global__
void placementNew(int *d_result)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ unsigned char  __align__(8) s_buffer[sizeof(Vector<int>)];
    __shared__ int __align__(8) s_data[1024];
    __shared__ Vector<int> *s_vector;

    // The first thread of the block initializes the shared Vector object.
    // The placement new operator enables the Vector object and the data array top be placed in shared memory.
    if (threadIdx.x == 0)
    {
        s_vector = new(s_buffer) Vector<int>(1024, s_data);
    }

    cg::sync(cta);

    if ((threadIdx.x & 1) == 0)
    {
        s_vector->push(threadIdx.x >> 1);
    }

    // Need to sync as the vector implementation does not support concurrent push/pop operations.
    cg::sync(cta);

    int v;

    if (s_vector->pop(v))
    {
        d_result[threadIdx.x] = v;
    }
    else
    {
        d_result[threadIdx.x] = -1;
    }

    // Note: deleting objects placed in shared memory is not necessary (lifetime of shared memory is that of the block)
}


struct ComplexType_t
{
    int a;
    int b;
    float c;
    float d;
};


__global__
void complexVector(int *d_result)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ unsigned char __align__(8) s_buffer[sizeof(Vector<ComplexType_t>)];
    __shared__ ComplexType_t __align__(8) s_data[1024];
    __shared__ Vector<ComplexType_t> *s_vector;

    // The first thread of the block initializes the shared Vector object.
    // The placement new operator enables the Vector object and the data array top be placed in shared memory.
    if (threadIdx.x == 0)
    {
        s_vector = new(s_buffer) Vector<ComplexType_t>(1024, s_data);
    }

    cg::sync(cta);

    if ((threadIdx.x & 1) == 0)
    {
        ComplexType_t data;
        data.a = threadIdx.x >> 1;
        data.b = blockIdx.x;
        data.c = threadIdx.x / (float)(blockDim.x);
        data.d = blockIdx.x / (float)(gridDim.x);

        s_vector->push(data);
    }

    cg::sync(cta);

    ComplexType_t v;

    if (s_vector->pop(v))
    {
        d_result[threadIdx.x] = v.a;
    }
    else
    {
        d_result[threadIdx.x] = -1;
    }

    // Note: deleting objects placed in shared memory is not necessary (lifetime of shared memory is that of the block)
}



///////////////////////////////////////////////////////////////////////////////////////////
//
// Host code
//
///////////////////////////////////////////////////////////////////////////////////////////

bool checkResult(int *d_result, int N)
{
    std::vector<int> h_result;
    h_result.resize(N);


    checkHIPErrors(hipMemcpy(&h_result[0], d_result, N*sizeof(int), hipMemcpyDeviceToHost));
    std::sort(h_result.begin(), h_result.end());

    bool success = true;
    bool test = false;

    int value=0;

    for (int i=0; i < N; ++i)
    {
        if (h_result[i] != -1)
        {
            test = true;
        }

        if (test && (value++) != h_result[i])
        {
            success = false;
        }
    }

    return success;
}


bool testContainer(Container<int> **d_container, int blocks, int threads)
{
    int *d_result;
    hipMalloc(&d_result, blocks*threads*sizeof(int));

    hipLaunchKernelGGL(containerFill, dim3(blocks), dim3(threads), 0, 0, d_container);
    hipLaunchKernelGGL(containerConsume, dim3(blocks), dim3(threads), 0, 0, d_container, d_result);
    hipLaunchKernelGGL(containerDelete, dim3(1), dim3(1), 0, 0, d_container);
    checkHIPErrors(hipDeviceSynchronize());

    bool success = checkResult(d_result, blocks*threads);

    hipFree(d_result);

    return success;
}



bool testPlacementNew(int threads)
{
    int *d_result;
    hipMalloc(&d_result, threads*sizeof(int));

    hipLaunchKernelGGL(placementNew, dim3(1), dim3(threads), 0, 0, d_result);
    checkHIPErrors(hipDeviceSynchronize());

    bool success = checkResult(d_result, threads);

    hipFree(d_result);

    return success;
}

bool testComplexType(int threads)
{
    int *d_result;
    hipMalloc(&d_result, threads*sizeof(int));

    hipLaunchKernelGGL(complexVector, dim3(1), dim3(threads), 0, 0, d_result);
    checkHIPErrors(hipDeviceSynchronize());

    bool success = checkResult(d_result, threads);

    hipFree(d_result);

    return success;
}

///////////////////////////////////////////////////////////////////////////////////////////
//
// MAIN
//
///////////////////////////////////////////////////////////////////////////////////////////


int main(int argc, char **argv)
{
    printf("%s Starting...\n\n", sSDKsample);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    findHIPDevice(argc, (const char **)argv);

    // set the heap size for device size new/delete to 128 MB
    checkHIPErrors(hipDeviceSetLimit(hipLimitMallocHeapSize, 128 * (1 << 20)));

    Container<int> **d_container;
    checkHIPErrors(hipMalloc(&d_container, sizeof(Container<int> **)));

    bool bTest = false;
    int test_passed = 0;

    printf(" > Container = Vector test ");
    hipLaunchKernelGGL(vectorCreate, dim3(1), dim3(1), 0, 0, d_container, 128 * 128);
    bTest = testContainer(d_container, 128, 128);
    printf(bTest ? "OK\n\n" : "NOT OK\n\n");
    test_passed += (bTest ? 1 : 0);

    checkHIPErrors(hipFree(d_container));

    printf(" > Container = Vector, using placement new on SMEM buffer test ");
    bTest = testPlacementNew(1024);
    printf(bTest ? "OK\n\n" : "NOT OK\n\n");
    test_passed += (bTest ? 1 : 0);

    printf(" > Container = Vector, with user defined datatype test ");
    bTest = testComplexType(1024);
    printf(bTest ? "OK\n\n" : "NOT OK\n\n");
    test_passed += (bTest ? 1 : 0);

    printf("Test Summary: %d/3 succesfully run\n", test_passed);

    exit(test_passed==3 ? EXIT_SUCCESS : EXIT_FAILURE);
}
