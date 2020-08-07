#include "hip/hip_runtime.h"
/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <helper_hip.h>
#include "jacobi.h"

namespace cg = cooperative_groups;

// 8 Rows of square-matrix A processed by each CTA.
// This can be max 32 and only power of 2 (i.e., 2/4/8/16/32).
#define ROWS_PER_CTA 8

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

static __global__
void JacobiMethod(const float *A, const double *b,
                  const float conv_threshold, 
                  double *x,
                  double *x_new, double *sum)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ double x_shared[N_ROWS]; // N_ROWS == n
    __shared__ double b_shared[ROWS_PER_CTA+1];

    for (int i = threadIdx.x; i < N_ROWS; i += blockDim.x)
    {
        x_shared[i] = x[i];
    }

    if (threadIdx.x < ROWS_PER_CTA)
    {
        int k=threadIdx.x;
        for (int i = k + (blockIdx.x * ROWS_PER_CTA); (k < ROWS_PER_CTA) && (i < N_ROWS); k+=ROWS_PER_CTA, i+=ROWS_PER_CTA)
        {
            b_shared[i%(ROWS_PER_CTA+1)] = b[i];
        }
    }

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    for (int k=0, i = blockIdx.x * ROWS_PER_CTA; (k < ROWS_PER_CTA) && (i < N_ROWS); k++, i++)
    {
        double rowThreadSum = 0.0;
        for (int j = threadIdx.x; j < N_ROWS; j += blockDim.x)
        {
            rowThreadSum += (A[i*N_ROWS + j] * x_shared[j]);
        }

        for (int offset = tile32.size()/2; offset > 0; offset /= 2) 
        {
             rowThreadSum += tile32.shfl_down(rowThreadSum, offset);
        }

        if (tile32.thread_rank() == 0)
        {
            atomicAdd(&b_shared[i%(ROWS_PER_CTA+1)], -rowThreadSum);
        }
    }

    cg::sync(cta);

    if (threadIdx.x < ROWS_PER_CTA)
    {
        cg::thread_block_tile<ROWS_PER_CTA> tile8 = cg::tiled_partition<ROWS_PER_CTA>(cta);
        double temp_sum = 0.0;

        int k = threadIdx.x;

        for (int i = k + (blockIdx.x * ROWS_PER_CTA); (k < ROWS_PER_CTA) && (i < N_ROWS); k+=ROWS_PER_CTA, i+=ROWS_PER_CTA)
        {
            double dx = b_shared[i%(ROWS_PER_CTA+1)];
            dx /= A[i*N_ROWS + i];

            x_new[i] = (x_shared[i] + dx);
            temp_sum += fabs(dx);
        }

        for (int offset = tile8.size()/2; offset > 0; offset /= 2) 
        {
             temp_sum += tile8.shfl_down(temp_sum, offset);
        }

        if (tile8.thread_rank() == 0)
        {
            atomicAdd(sum, temp_sum);
        }
    }
}

// Thread block size for finalError kernel should be multiple of 32
static __global__ void finalError(double *x, double *g_sum)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    HIP_DYNAMIC_SHARED( double, warpSum)
    double sum = 0.0;

    int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i=globalThreadId; i<N_ROWS; i+=blockDim.x*gridDim.x)
    {
       double d = x[i] - 1.0;
       sum += fabs(d);
    }

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    for (int offset = tile32.size()/2; offset > 0; offset /= 2) 
    {
         sum += tile32.shfl_down(sum, offset);
    }

    if (tile32.thread_rank() == 0)
    {
        warpSum[threadIdx.x / warpSize] = sum;
    }

    cg::sync(cta);

    double blockSum = 0.0;
    if (threadIdx.x < (blockDim.x/warpSize))
    {
        blockSum = warpSum[threadIdx.x];
    }

    if (threadIdx.x < 32)
    {
        for (int offset = tile32.size()/2; offset > 0; offset /= 2) 
        {
            blockSum += tile32.shfl_down(blockSum, offset);
        }
        if (tile32.thread_rank() == 0)
        {
            atomicAdd(g_sum, blockSum);
        }
    }
}

double JacobiMethodGpuCudaGraphExecKernelSetParams(const float *A, const double *b, const float conv_threshold, 
                    const int max_iter, double *x,
                    double *x_new, hipStream_t stream)
{
    // CTA size
    dim3 nthreads(256, 1, 1);
    // grid size
    dim3 nblocks((N_ROWS/ROWS_PER_CTA) + 2, 1, 1);
    cudaGraph_t graph;
    cudaGraphExec_t graphExec = NULL;

    double sum=0.0;
    double *d_sum = NULL;
    checkHIPErrors(hipMalloc(&d_sum, sizeof(double)));

    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t memcpyNode, jacobiKernelNode, memsetNode;
    hipMemcpy3DParms memcpyParams = {0};
    cudaMemsetParams memsetParams = {0};

    memsetParams.dst            = (void*)d_sum;
    memsetParams.value          = 0;
    memsetParams.pitch          = 0;
    // elementSize can be max 4 bytes, so we take sizeof(float) and width=2
    memsetParams.elementSize    = sizeof(float); 
    memsetParams.width          = 2; 
    memsetParams.height         = 1;

    checkHIPErrors(cudaGraphCreate(&graph, 0));
    checkHIPErrors(cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));
    nodeDependencies.push_back(memsetNode);

    cudaKernelNodeParams NodeParams0, NodeParams1;
    NodeParams0.func = (void*)JacobiMethod;
    NodeParams0.gridDim = nblocks;
    NodeParams0.blockDim = nthreads;
    NodeParams0.sharedMemBytes = 0;
    void *kernelArgs0[6] = {(void*)&A, (void*)&b, (void*)&conv_threshold, (void*)&x, (void*)&x_new, (void*)&d_sum};
    NodeParams0.kernelParams = kernelArgs0;
    NodeParams0.extra = NULL;

    checkHIPErrors(cudaGraphAddKernelNode(&jacobiKernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &NodeParams0));

    nodeDependencies.clear();
    nodeDependencies.push_back(jacobiKernelNode);

    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos   = make_hipPos(0,0,0);
    memcpyParams.srcPtr   = make_hipPitchedPtr(d_sum, sizeof(double), 1, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos   = make_hipPos(0,0,0);
    memcpyParams.dstPtr   = make_hipPitchedPtr(&sum, sizeof(double), 1, 1);
    memcpyParams.extent   = make_hipExtent(sizeof(double), 1, 1);
    memcpyParams.kind     = hipMemcpyDeviceToHost;

    checkHIPErrors(cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(), nodeDependencies.size(), &memcpyParams));

    checkHIPErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    NodeParams1.func = (void*)JacobiMethod;
    NodeParams1.gridDim = nblocks;
    NodeParams1.blockDim = nthreads;
    NodeParams1.sharedMemBytes = 0;
    void *kernelArgs1[6] = {(void*)&A, (void*)&b, (void*)&conv_threshold, (void*)&x_new, (void*)&x, (void*)&d_sum};
    NodeParams1.kernelParams = kernelArgs1;
    NodeParams1.extra = NULL;

    int k=0;
    for (k=0; k<max_iter; k++)
    {
        checkHIPErrors(cudaGraphExecKernelNodeSetParams(graphExec, jacobiKernelNode,
                                         ((k&1) == 0) ? &NodeParams0 : &NodeParams1));
        checkHIPErrors(cudaGraphLaunch(graphExec, stream));
        checkHIPErrors(hipStreamSynchronize(stream));

        if(sum <= conv_threshold) 
        {
            checkHIPErrors(hipMemsetAsync(d_sum, 0, sizeof(double), stream));
            nblocks.x  = (N_ROWS / nthreads.x) + 1;
            size_t sharedMemSize = ((nthreads.x/32)+1)*sizeof(double);
            if ((k&1) == 0)
            {
                hipLaunchKernelGGL(finalError, dim3(nblocks), dim3(nthreads), sharedMemSize, stream, x_new, d_sum);
            }
            else
            {
                hipLaunchKernelGGL(finalError, dim3(nblocks), dim3(nthreads), sharedMemSize, stream, x, d_sum);
            }

            checkHIPErrors(hipMemcpyAsync(&sum, d_sum, sizeof(double), hipMemcpyDeviceToHost, stream));
            checkHIPErrors(hipStreamSynchronize(stream));
            printf("GPU iterations : %d\n",k+1);
            printf("GPU error : %.3e\n", sum);
            break;
        }
    }

    checkHIPErrors(hipFree(d_sum));
    return sum;
}

double JacobiMethodGpuCudaGraphExecUpdate(const float *A, const double *b, const float conv_threshold, 
                    const int max_iter, double *x,
                    double *x_new, hipStream_t stream)
{
    // CTA size
    dim3 nthreads(256, 1, 1);
    // grid size
    dim3 nblocks((N_ROWS/ROWS_PER_CTA) + 2, 1, 1);
    cudaGraph_t graph;
    cudaGraphExec_t graphExec = NULL;

    double sum=0.0;
    double *d_sum;
    checkHIPErrors(hipMalloc(&d_sum, sizeof(double)));

    int k=0;
    for (k=0; k<max_iter; k++)
    {
        checkHIPErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        checkHIPErrors(hipMemsetAsync(d_sum, 0, sizeof(double), stream));
        if ((k&1) == 0)
        {
            hipLaunchKernelGGL(JacobiMethod, dim3(nblocks), dim3(nthreads), 0, stream, A, b, conv_threshold,
                                                    x, x_new, d_sum);
        }
        else
        {
            hipLaunchKernelGGL(JacobiMethod, dim3(nblocks), dim3(nthreads), 0, stream, A, b, conv_threshold,
                                                    x_new, x, d_sum);
        }
        checkHIPErrors(hipMemcpyAsync(&sum, d_sum, sizeof(double), hipMemcpyDeviceToHost, stream));
        checkHIPErrors(cudaStreamEndCapture(stream, &graph));

        if (graphExec == NULL)
        {
            checkHIPErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
        }
        else
        {
            cudaGraphExecUpdateResult updateResult_out;
            checkHIPErrors(cudaGraphExecUpdate(graphExec, graph, NULL, &updateResult_out));
            if (updateResult_out != cudaGraphExecUpdateSuccess) {
                if (graphExec != NULL) {
                    checkHIPErrors(cudaGraphExecDestroy(graphExec));
                }
                printf("k = %d graph update failed with error - %d\n", k, updateResult_out);
                checkHIPErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
            }
        }
        checkHIPErrors(cudaGraphLaunch(graphExec, stream));
        checkHIPErrors(hipStreamSynchronize(stream));

        if(sum <= conv_threshold) 
        {
            checkHIPErrors(hipMemsetAsync(d_sum, 0, sizeof(double), stream));
            nblocks.x  = (N_ROWS / nthreads.x) + 1;
            size_t sharedMemSize = ((nthreads.x/32)+1)*sizeof(double);
            if ((k&1) == 0)
            {
                hipLaunchKernelGGL(finalError, dim3(nblocks), dim3(nthreads), sharedMemSize, stream, x_new, d_sum);
            }
            else
            {
                hipLaunchKernelGGL(finalError, dim3(nblocks), dim3(nthreads), sharedMemSize, stream, x, d_sum);
            }

            checkHIPErrors(hipMemcpyAsync(&sum, d_sum, sizeof(double), hipMemcpyDeviceToHost, stream));
            checkHIPErrors(hipStreamSynchronize(stream));
            printf("GPU iterations : %d\n",k+1);
            printf("GPU error : %.3e\n", sum);
            break;
        }
    }

    checkHIPErrors(hipFree(d_sum));
    return sum;
}

double JacobiMethodGpu(const float *A, const double *b, const float conv_threshold, 
                    const int max_iter, double *x,
                    double *x_new, hipStream_t stream)
{
    // CTA size
    dim3 nthreads(256, 1, 1);
    // grid size
    dim3 nblocks((N_ROWS/ROWS_PER_CTA) + 2, 1, 1);

    double sum=0.0;
    double *d_sum;
    checkHIPErrors(hipMalloc(&d_sum, sizeof(double)));
    int k=0;

    for (k=0; k<max_iter; k++)
    {
        checkHIPErrors(hipMemsetAsync(d_sum, 0, sizeof(double), stream));
        if ((k&1) == 0)
        {
            hipLaunchKernelGGL(JacobiMethod, dim3(nblocks), dim3(nthreads), 0, stream, A, b, conv_threshold,
                                                    x, x_new, d_sum);
        }
        else
        {
            hipLaunchKernelGGL(JacobiMethod, dim3(nblocks), dim3(nthreads), 0, stream, A, b, conv_threshold,
                                                    x_new, x, d_sum);
        }
        checkHIPErrors(hipMemcpyAsync(&sum, d_sum, sizeof(double), hipMemcpyDeviceToHost, stream));
        checkHIPErrors(hipStreamSynchronize(stream));

        if(sum <= conv_threshold) 
        {
            checkHIPErrors(hipMemsetAsync(d_sum, 0, sizeof(double), stream));
            nblocks.x  = (N_ROWS / nthreads.x) + 1;
            size_t sharedMemSize = ((nthreads.x/32)+1)*sizeof(double);
            if ((k&1) == 0)
            {
                hipLaunchKernelGGL(finalError, dim3(nblocks), dim3(nthreads), sharedMemSize, stream, x_new, d_sum);
            }
            else
            {
                hipLaunchKernelGGL(finalError, dim3(nblocks), dim3(nthreads), sharedMemSize, stream, x, d_sum);
            }

            checkHIPErrors(hipMemcpyAsync(&sum, d_sum, sizeof(double), hipMemcpyDeviceToHost, stream));
            checkHIPErrors(hipStreamSynchronize(stream));
            printf("GPU iterations : %d\n",k+1);
            printf("GPU error : %.3e\n", sum);
            break;
        }
    }

    checkHIPErrors(hipFree(d_sum));
    return sum;
}
