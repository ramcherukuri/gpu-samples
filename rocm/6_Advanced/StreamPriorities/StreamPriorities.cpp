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

// std::system includes
#include <cstdio>

// CUDA-C includes
#include <hip/hip_runtime.h>

#include <helper_hip.h>

#define TOTAL_SIZE  256*1024*1024
#define EACH_SIZE   128*1024*1024

// # threadblocks
#define TBLOCKS 1024
#define THREADS  512

// throw error on equality
#define ERR_EQ(X,Y) do { if ((X) == (Y)) { \
            fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
            exit(-1);}} while(0)

// throw error on difference
#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
            fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
            exit(-1);}} while(0)

// copy from source -> destination arrays
__global__ void memcpy_kernel(int *dst, int *src, size_t n)
{
    int num = gridDim.x * blockDim.x;
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = id; i < n / sizeof(int); i += num)
    {
        dst[i] = src[i];
    }
}

// initialise memory
void mem_init(int *buf, size_t n)
{
    for (int i = 0; i < n / sizeof(int); i++)
    {
        buf[i] = i;
    }
}

int main(int argc, char **argv)
{
    hipDeviceProp_t device_prop;
    int dev_id;

    printf("Starting [%s]...\n", argv[0]);

    // set device
    dev_id = findHIPDevice(argc, (const char **) argv);
    checkHIPErrors(hipGetDeviceProperties(&device_prop, dev_id));

    if ((device_prop.major << 4) + device_prop.minor < 0x35)
    {
        fprintf(stderr, "%s requires Compute Capability of SM 3.5 or higher to run.\nexiting...\n", argv[0]);
        exit(EXIT_WAIVED);
    }

    // get the range of priorities available
    // [ greatest_priority, lowest_priority ]
    int priority_low;
    int priority_hi;
    checkHIPErrors(hipDeviceGetStreamPriorityRange(&priority_low, &priority_hi));

    printf("CUDA stream priority range: LOW: %d to HIGH: %d\n",priority_low,priority_hi);

    // create streams with highest and lowest available priorities
    hipStream_t st_low;
    hipStream_t st_hi;
    checkHIPErrors(hipStreamCreateWithPriority(&st_low, hipStreamNonBlocking, priority_low));
    checkHIPErrors(hipStreamCreateWithPriority(&st_hi,  hipStreamNonBlocking, priority_hi));

    size_t size;
    size = TOTAL_SIZE;

    // initialise host data
    int *h_src_low;
    int *h_src_hi;
    ERR_EQ(h_src_low = (int *) malloc(size), NULL);
    ERR_EQ(h_src_hi  = (int *) malloc(size), NULL);
    mem_init(h_src_low, size);
    mem_init(h_src_hi,  size);

    // initialise device data
    int *h_dst_low;
    int *h_dst_hi;
    ERR_EQ(h_dst_low = (int *) malloc(size), NULL);
    ERR_EQ(h_dst_hi  = (int *) malloc(size), NULL);
    memset(h_dst_low, 0, size);
    memset(h_dst_hi,  0, size);

    // copy source data -> device
    int *d_src_low;
    int *d_src_hi;
    checkHIPErrors(hipMalloc(&d_src_low, size));
    checkHIPErrors(hipMalloc(&d_src_hi,  size));
    checkHIPErrors(hipMemcpy(d_src_low, h_src_low, size, hipMemcpyHostToDevice));
    checkHIPErrors(hipMemcpy(d_src_hi,  h_src_hi,  size, hipMemcpyHostToDevice));

    // allocate memory for memcopy destination
    int *d_dst_low;
    int *d_dst_hi;
    checkHIPErrors(hipMalloc(&d_dst_low, size));
    checkHIPErrors(hipMalloc(&d_dst_hi,  size));

    // create some events
    hipEvent_t ev_start_low;
    hipEvent_t ev_start_hi;
    hipEvent_t ev_end_low;
    hipEvent_t ev_end_hi;
    checkHIPErrors(hipEventCreate(&ev_start_low));
    checkHIPErrors(hipEventCreate(&ev_start_hi));
    checkHIPErrors(hipEventCreate(&ev_end_low));
    checkHIPErrors(hipEventCreate(&ev_end_hi));

    /* */

    // call pair of kernels repeatedly (with different priority streams)
    checkHIPErrors(hipEventRecord(ev_start_low, st_low));
    checkHIPErrors(hipEventRecord(ev_start_hi,  st_hi));

    for (int i = 0; i < TOTAL_SIZE; i += EACH_SIZE)
    {
        int j = i / sizeof(int);
        hipLaunchKernelGGL(memcpy_kernel, dim3(TBLOCKS), dim3(THREADS), 0, st_low, d_dst_low + j, d_src_low + j, EACH_SIZE);
        hipLaunchKernelGGL(memcpy_kernel, dim3(TBLOCKS), dim3(THREADS), 0, st_hi , d_dst_hi  + j, d_src_hi  + j, EACH_SIZE);
    }

    checkHIPErrors(hipEventRecord(ev_end_low, st_low));
    checkHIPErrors(hipEventRecord(ev_end_hi,  st_hi));

    checkHIPErrors(hipEventSynchronize(ev_end_low));
    checkHIPErrors(hipEventSynchronize(ev_end_hi));

    /* */

    size = TOTAL_SIZE;
    checkHIPErrors(hipMemcpy(h_dst_low, d_dst_low, size, hipMemcpyDeviceToHost));
    checkHIPErrors(hipMemcpy(h_dst_hi,  d_dst_hi,  size, hipMemcpyDeviceToHost));

    // check results of kernels
    ERR_NE(memcmp(h_dst_low, h_src_low, size), 0);
    ERR_NE(memcmp(h_dst_hi,  h_src_hi,  size), 0);

    // check timings
    float ms_low;
    float ms_hi;
    checkHIPErrors(hipEventElapsedTime(&ms_low, ev_start_low, ev_end_low));
    checkHIPErrors(hipEventElapsedTime(&ms_hi,  ev_start_hi,  ev_end_hi));

    printf("elapsed time of kernels launched to LOW priority stream: %.3lf ms\n", ms_low);
    printf("elapsed time of kernels launched to HI  priority stream: %.3lf ms\n", ms_hi);

    exit(EXIT_SUCCESS);
}
