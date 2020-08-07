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

// Shuffle intrinsics CUDA Sample
// This sample demonstrates the use of the shuffle intrinsic
// First, a simple example of a prefix sum using the shuffle to
// perform a scan operation is provided.
// Secondly, a more involved example of computing an integral image
// using the shuffle intrinsic is provided, where the shuffle
// scan operation and shuffle xor operations are used

#include <stdio.h>

#include <hip/hip_runtime.h>

#include <helper_functions.h>
#include <helper_hip.h>
#include "shfl_integral_image.cuh"

// Scan using shfl - takes log2(n) steps
// This function demonstrates basic use of the shuffle intrinsic, __shfl_up,
// to perform a scan operation across a block.
// First, it performs a scan (prefix sum in this case) inside a warp
// Then to continue the scan operation across the block,
// each warp's sum is placed into shared memory.  A single warp
// then performs a shuffle scan on that shared memory.  The results
// are then uniformly added to each warp's threads.
// This pyramid type approach is continued by placing each block's
// final sum in global memory and prefix summing that via another kernel call, then
// uniformly adding across the input data via the uniform_add<<<>>> kernel.

__global__ void shfl_scan_test(int *data, int width, int *partial_sums=NULL)
{
    HIP_DYNAMIC_SHARED( int, sums)
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int lane_id = id % warpSize;
    // determine a warp_id within a block
    int warp_id = threadIdx.x / warpSize;

    // Below is the basic structure of using a shfl instruction
    // for a scan.
    // Record "value" as a variable - we accumulate it along the way
    int value = data[id];

    // Now accumulate in log steps up the chain
    // compute sums, with another thread's value who is
    // distance delta away (i).  Note
    // those threads where the thread 'i' away would have
    // been out of bounds of the warp are unaffected.  This
    // creates the scan sum.

#pragma unroll
    for (int i=1; i<=width; i*=2)
    {
        unsigned int mask = 0xffffffff;
        int n = __shfl_up_sync(mask, value, i, width);

        if (lane_id >= i) value += n;
    }

    // value now holds the scan value for the individual thread
    // next sum the largest values for each warp

    // write the sum of the warp to smem
    if (threadIdx.x % warpSize == warpSize-1)
    {
        sums[warp_id] = value;
    }

    __syncthreads();

    //
    // scan sum the warp sums
    // the same shfl scan operation, but performed on warp sums
    //
    if (warp_id == 0 && lane_id < (blockDim.x / warpSize))
    {
        int warp_sum = sums[lane_id];

        int mask = (1 << (blockDim.x / warpSize)) - 1;
        for (int i=1; i<=(blockDim.x / warpSize); i*=2)
        {
            int n = __shfl_up_sync(mask, warp_sum, i, (blockDim.x / warpSize));

            if (lane_id >= i) warp_sum += n;
        }

        sums[lane_id] = warp_sum;
    }

    __syncthreads();

    // perform a uniform add across warps in the block
    // read neighbouring warp's sum and add it to threads value
    int blockSum = 0;

    if (warp_id > 0)
    {
        blockSum = sums[warp_id-1];
    }

    value += blockSum;

    // Now write out our result
    data[id] = value;

    // last thread has sum, write write out the block's sum
    if (partial_sums != NULL && threadIdx.x == blockDim.x-1)
    {
        partial_sums[blockIdx.x] = value;
    }
}

// Uniform add: add partial sums array
__global__ void uniform_add(int *data, int *partial_sums, int len)
{
    __shared__ int buf;
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if (id > len) return;

    if (threadIdx.x == 0)
    {
        buf = partial_sums[blockIdx.x];
    }

    __syncthreads();
    data[id] += buf;
}

static unsigned int iDivUp(unsigned int dividend, unsigned int divisor)
{
    return ((dividend % divisor) == 0) ?
           (dividend / divisor) :
           (dividend / divisor + 1);
}


// This function verifies the shuffle scan result, for the simple
// prefix sum case.
bool CPUverify(int *h_data, int *h_result, int n_elements)
{
    // cpu verify
    for (int i=0; i<n_elements-1; i++)
    {
        h_data[i+1] = h_data[i] + h_data[i+1];
    }

    int diff = 0;

    for (int i=0 ; i<n_elements; i++)
    {
        diff += h_data[i]-h_result[i];
    }

    printf("CPU verify result diff (GPUvsCPU) = %d\n", diff);
    bool bTestResult = false;

    if (diff == 0) bTestResult = true;

    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (int j=0; j<100; j++)
        for (int i=0; i<n_elements-1; i++)
        {
            h_data[i+1] = h_data[i] + h_data[i+1];
        }

    sdkStopTimer(&hTimer);
    double cput= sdkGetTimerValue(&hTimer);
    printf("CPU sum (naive) took %f ms\n", cput/100);
    return bTestResult;
}


// this verifies the row scan result for synthetic data of all 1's
unsigned int verifyDataRowSums(unsigned int *h_image, int w, int h)
{
    unsigned int diff = 0;

    for (int j=0; j<h; j++)
    {
        for (int i=0; i<w; i++)
        {
            int gold = i+1;
            diff += abs((int)gold-(int)h_image[j*w + i]);
        }
    }

    return diff;
}

bool shuffle_simple_test(int argc, char **argv)
{
    int *h_data, *h_partial_sums, *h_result;
    int *d_data, *d_partial_sums;
    const int n_elements = 65536;
    int sz = sizeof(int)*n_elements;
    int cuda_device = 0;

    printf("Starting shfl_scan\n");

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    cuda_device = findHIPDevice(argc, (const char **)argv);

    hipDeviceProp_t deviceProp;
    checkHIPErrors(hipGetDevice(&cuda_device));

    checkHIPErrors(hipGetDeviceProperties(&deviceProp, cuda_device));

    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // __shfl intrinsic needs SM 3.0 or higher
    if (deviceProp.major < 3)
    {
        printf("> __shfl() intrinsic requires device SM 3.0+\n");
        printf("> Waiving test.\n");
        exit(EXIT_WAIVED);
    }

    checkHIPErrors(hipHostMalloc((void **)&h_data, sizeof(int)*n_elements));
    checkHIPErrors(hipHostMalloc((void **)&h_result, sizeof(int)*n_elements));

    //initialize data:
    printf("Computing Simple Sum test\n");
    printf("---------------------------------------------------\n");

    printf("Initialize test data [1, 1, 1...]\n");

    for (int i=0; i<n_elements; i++)
    {
        h_data[i] = 1;
    }

    int blockSize = 256;
    int gridSize = n_elements/blockSize;
    int nWarps = blockSize/32;
    int shmem_sz = nWarps * sizeof(int);
    int n_partialSums = n_elements/blockSize;
    int partial_sz = n_partialSums*sizeof(int);

    printf("Scan summation for %d elements, %d partial sums\n",
           n_elements, n_elements/blockSize);

    int p_blockSize = min(n_partialSums, blockSize);
    int p_gridSize = iDivUp(n_partialSums, p_blockSize);
    printf("Partial summing %d elements with %d blocks of size %d\n",
           n_partialSums, p_gridSize, p_blockSize);

    // initialize a timer
    hipEvent_t start, stop;
    checkHIPErrors(hipEventCreate(&start));
    checkHIPErrors(hipEventCreate(&stop));
    float et = 0;
    float inc = 0;

    checkHIPErrors(hipMalloc((void **)&d_data, sz));
    checkHIPErrors(hipMalloc((void **)&d_partial_sums, partial_sz));
    checkHIPErrors(hipMemset(d_partial_sums, 0, partial_sz));

    checkHIPErrors(hipHostMalloc((void **)&h_partial_sums, partial_sz));
    checkHIPErrors(hipMemcpy(d_data, h_data, sz, hipMemcpyHostToDevice));

    checkHIPErrors(hipEventRecord(start, 0));
    hipLaunchKernelGGL(shfl_scan_test, dim3(gridSize), dim3(blockSize), shmem_sz, 0, d_data, 32, d_partial_sums);
    hipLaunchKernelGGL(shfl_scan_test, dim3(p_gridSize), dim3(p_blockSize), shmem_sz, 0, d_partial_sums,32);
    hipLaunchKernelGGL(uniform_add, dim3(gridSize-1), dim3(blockSize), 0, 0, d_data+blockSize, d_partial_sums, n_elements);
    checkHIPErrors(hipEventRecord(stop, 0));
    checkHIPErrors(hipEventSynchronize(stop));
    checkHIPErrors(hipEventElapsedTime(&inc, start, stop));
    et+=inc;

    checkHIPErrors(hipMemcpy(h_result, d_data, sz, hipMemcpyDeviceToHost));
    checkHIPErrors(hipMemcpy(h_partial_sums, d_partial_sums, partial_sz,
                               hipMemcpyDeviceToHost));

    printf("Test Sum: %d\n", h_partial_sums[n_partialSums-1]);
    printf("Time (ms): %f\n", et);
    printf("%d elements scanned in %f ms -> %f MegaElements/s\n",
           n_elements, et, n_elements/(et/1000.0f)/1000000.0f);

    bool bTestResult = CPUverify(h_data, h_result, n_elements);

    checkHIPErrors(hipHostFree(h_data));
    checkHIPErrors(hipHostFree(h_result));
    checkHIPErrors(hipHostFree(h_partial_sums));
    checkHIPErrors(hipFree(d_data));
    checkHIPErrors(hipFree(d_partial_sums));

    return bTestResult;
}

// This function tests creation of an integral image using
// synthetic data, of size 1920x1080 pixels greyscale.
bool shuffle_integral_image_test()
{
    char *d_data;
    unsigned int *h_image;
    unsigned int *d_integral_image;
    int w = 1920;
    int h = 1080;
    int n_elements = w*h;
    int sz = sizeof(unsigned int)*n_elements;

    printf("\nComputing Integral Image Test on size %d x %d synthetic data\n", w, h);
    printf("---------------------------------------------------\n");
    checkHIPErrors(hipHostMalloc((void **)&h_image, sz));
    // fill test "image" with synthetic 1's data
    memset(h_image, 0, sz);

    // each thread handles 16 values, use 1 block/row
    int blockSize = iDivUp(w,16);
    // launch 1 block / row
    int gridSize = h;

    // Create a synthetic image for testing
    checkHIPErrors(hipMalloc((void **)&d_data, sz));
    checkHIPErrors(hipMalloc((void **)&d_integral_image, n_elements*sizeof(int)*4));
    checkHIPErrors(hipMemset(d_data, 1, sz));
    checkHIPErrors(hipMemset(d_integral_image, 0, sz));

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float et = 0;
    unsigned int err;


    // Execute scan line prefix sum kernel, and time it
    hipEventRecord(start);
    hipLaunchKernelGGL(shfl_intimage_rows, dim3(gridSize), dim3(blockSize), 0, 0, (uint4 *)d_data, (uint4 *)d_integral_image);
    hipEventRecord(stop);
    checkHIPErrors(hipEventSynchronize(stop));
    checkHIPErrors(hipEventElapsedTime(&et, start, stop));
    printf("Method: Fast  Time (GPU Timer): %f ms ", et);

    // verify the scan line results
    checkHIPErrors(hipMemcpy(h_image, d_integral_image, sz, hipMemcpyDeviceToHost));
    err = verifyDataRowSums(h_image, w, h);
    printf("Diff = %d\n", err);

    // Execute column prefix sum kernel and time it
    dim3 blockSz(32, 8);
    dim3 testGrid(w/blockSz.x, 1);

    hipEventRecord(start);
    hipLaunchKernelGGL(shfl_vertical_shfl, dim3(testGrid), dim3(blockSz), 0, 0, (unsigned int *)d_integral_image, w, h);
    hipEventRecord(stop);
    checkHIPErrors(hipEventSynchronize(stop));
    checkHIPErrors(hipEventElapsedTime(&et, start, stop));
    printf("Method: Vertical Scan  Time (GPU Timer): %f ms ", et);

    // Verify the column results
    checkHIPErrors(hipMemcpy(h_image, d_integral_image, sz, hipMemcpyDeviceToHost));
    printf("\n");

    int finalSum = h_image[w*h-1];
    printf("CheckSum: %d, (expect %dx%d=%d)\n", finalSum, w,h, w *h);

    checkHIPErrors(hipFree(d_data));
    checkHIPErrors(hipFree(d_integral_image));
    checkHIPErrors(hipHostFree(h_image));
    // verify final sum: if the final value in the corner is the same as the size of the
    // buffer (all 1's) then the integral image was generated successfully
    return (finalSum == w*h)? true : false;
}

int main(int argc, char *argv[])
{
    // Initialization.  The shuffle intrinsic is not available on SM < 3.0
    // so waive the test if the hardware is not present.
    int cuda_device = 0;

    printf("Starting shfl_scan\n");

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    cuda_device = findHIPDevice(argc, (const char **)argv);

    hipDeviceProp_t deviceProp;
    checkHIPErrors(hipGetDevice(&cuda_device));

    checkHIPErrors(hipGetDeviceProperties(&deviceProp, cuda_device));

    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // __shfl intrinsic needs SM 3.0 or higher
    if (deviceProp.major < 3)
    {
        printf("> __shfl() intrinsic requires device SM 3.0+\n");
        printf("> Waiving test.\n");
        exit(EXIT_WAIVED);
    }


    bool bTestResult = true;
    bool simpleTest = shuffle_simple_test(argc, argv);
    bool intTest = shuffle_integral_image_test();

    bTestResult = simpleTest & intTest;

    exit((bTestResult) ? EXIT_SUCCESS : EXIT_FAILURE);
}
