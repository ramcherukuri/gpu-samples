#include "hip/hip_runtime.h"
/*
 * Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <hip/hip_runtime.h>
#include <helper_hip.h>
#include <vector>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS  3


__global__ void reduce(float *inputVec, double *outputVec, size_t inputSize, size_t outputSize)
{
    __shared__ double tmp[THREADS_PER_BLOCK];

    cg::thread_block cta = cg::this_thread_block();
    size_t globaltid = blockIdx.x*blockDim.x + threadIdx.x;

    double temp_sum = 0.0;
    for (int i=globaltid; i < inputSize; i+=gridDim.x*blockDim.x)
    {
        temp_sum += (double) inputVec[i];
    }
    tmp[cta.thread_rank()] = temp_sum;

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    double beta  = temp_sum;
    double temp;

    for (int i = tile32.size() / 2; i > 0; i >>= 1) {
        if (tile32.thread_rank() < i) {
            temp       = tmp[cta.thread_rank() + i];
            beta       += temp;
            tmp[cta.thread_rank()] = beta;
        }
        cg::sync(tile32);
    }
    cg::sync(cta);

    if (cta.thread_rank() == 0 && blockIdx.x < outputSize) {
        beta  = 0.0;
        for (int i = 0; i < cta.size(); i += tile32.size()) {
            beta  += tmp[i];
        }
        outputVec[blockIdx.x] =  beta;
    }
}

__global__ void reduceFinal(double *inputVec, double *result, size_t inputSize)
{
    __shared__ double tmp[THREADS_PER_BLOCK];

    cg::thread_block cta = cg::this_thread_block();
    size_t globaltid = blockIdx.x*blockDim.x + threadIdx.x;

    double temp_sum = 0.0;
    for (int i=globaltid; i < inputSize; i+=gridDim.x*blockDim.x)
    {
        temp_sum += (double) inputVec[i];
    }
    tmp[cta.thread_rank()] = temp_sum;

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

   // do reduction in shared mem
    if ((blockDim.x >= 512) && (cta.thread_rank() < 256))
    {
        tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 256];
    }

    cg::sync(cta);

    if ((blockDim.x >= 256) &&(cta.thread_rank() < 128))
    {
        tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 128];
    }

    cg::sync(cta);

    if ((blockDim.x >= 128) && (cta.thread_rank() <  64))
    {
       tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() +  64];
    }

    cg::sync(cta);

    if (cta.thread_rank() < 32)
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockDim.x >=  64) temp_sum += tmp[cta.thread_rank() + 32];
        // Reduce final warp using shuffle
        for (int offset = tile32.size()/2; offset > 0; offset /= 2) 
        {
             temp_sum += tile32.shfl_down(temp_sum, offset);
        }
    }
    // write result for this block to global mem
    if (cta.thread_rank() == 0) result[0] = temp_sum;
}

void init_input(float*a, size_t size)
{
    for (size_t i=0; i < size; i++)
        a[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

void cudaGraphsManual(float* inputVec_h, float *inputVec_d, double *outputVec_d, double *result_d, size_t inputSize, size_t numOfBlocks)
{
    hipStream_t streamForGraph;
    cudaGraph_t graph;
    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t memcpyNode, kernelNode, memsetNode;
    double result_h = 0.0;

    checkHIPErrors(hipStreamCreateWithFlags(&streamForGraph, hipStreamNonBlocking));

    cudaKernelNodeParams kernelNodeParams = {0};
    hipMemcpy3DParms memcpyParams = {0};
    cudaMemsetParams memsetParams = {0};

    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos   = make_hipPos(0,0,0);
    memcpyParams.srcPtr   = make_hipPitchedPtr(inputVec_h, sizeof(float)*inputSize, inputSize, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos   = make_hipPos(0,0,0);
    memcpyParams.dstPtr   = make_hipPitchedPtr(inputVec_d, sizeof(float)*inputSize, inputSize, 1);
    memcpyParams.extent   = make_hipExtent(sizeof(float)*inputSize, 1, 1);
    memcpyParams.kind     = hipMemcpyHostToDevice;

    memsetParams.dst            = (void*)outputVec_d;
    memsetParams.value          = 0;
    memsetParams.pitch          = 0;
    memsetParams.elementSize    = sizeof(float); // elementSize can be max 4 bytes
    memsetParams.width          = numOfBlocks*2; 
    memsetParams.height         = 1;

    checkHIPErrors(cudaGraphCreate(&graph, 0));
    checkHIPErrors(cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams));
    checkHIPErrors(cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));

    nodeDependencies.push_back(memsetNode);
    nodeDependencies.push_back(memcpyNode);

    void *kernelArgs[4] = {(void*)&inputVec_d, (void*)&outputVec_d, &inputSize, &numOfBlocks};

    kernelNodeParams.func = (void*)reduce;
    kernelNodeParams.gridDim  = dim3(numOfBlocks, 1, 1);
    kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = (void **)kernelArgs;
    kernelNodeParams.extra = NULL;

    checkHIPErrors(cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams));

    nodeDependencies.clear();
    nodeDependencies.push_back(kernelNode);

    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst            = result_d;
    memsetParams.value          = 0;
    memsetParams.elementSize    = sizeof(float);
    memsetParams.width          = 2;
    memsetParams.height         = 1;
    checkHIPErrors(cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));

    nodeDependencies.push_back(memsetNode);
    
    memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
    kernelNodeParams.func = (void*)reduceFinal;
    kernelNodeParams.gridDim  = dim3(1, 1, 1);
    kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
    kernelNodeParams.sharedMemBytes = 0;
    void *kernelArgs2[3] =  {(void*)&outputVec_d, (void*)&result_d, &numOfBlocks};
    kernelNodeParams.kernelParams = kernelArgs2;
    kernelNodeParams.extra = NULL;

    checkHIPErrors(cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams));
    nodeDependencies.clear();
    nodeDependencies.push_back(kernelNode);

    memset(&memcpyParams, 0, sizeof(memcpyParams));

    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos   = make_hipPos(0,0,0);
    memcpyParams.srcPtr   = make_hipPitchedPtr(result_d, sizeof(double), 1, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos   = make_hipPos(0,0,0);
    memcpyParams.dstPtr   = make_hipPitchedPtr(&result_h, sizeof(double), 1, 1);
    memcpyParams.extent   = make_hipExtent(sizeof(double), 1, 1);
    memcpyParams.kind     = hipMemcpyDeviceToHost;
    checkHIPErrors(cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(), nodeDependencies.size(), &memcpyParams));
    nodeDependencies.clear();
    nodeDependencies.push_back(memcpyNode);

    cudaGraphNode_t *nodes = NULL;
    size_t numNodes = 0;
    checkHIPErrors(cudaGraphGetNodes(graph, nodes, &numNodes));
    printf("\nNum of nodes in the graph created manually = %zu\n", numNodes);

    cudaGraphExec_t graphExec;
    checkHIPErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    cudaGraph_t clonedGraph;
    cudaGraphExec_t clonedGraphExec;
    checkHIPErrors(cudaGraphClone(&clonedGraph, graph));
    checkHIPErrors(cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0));

    for (int i=0; i < GRAPH_LAUNCH_ITERATIONS; i++)
    {
       checkHIPErrors(cudaGraphLaunch(graphExec, streamForGraph));
       checkHIPErrors(hipStreamSynchronize(streamForGraph));
       printf("[cudaGraphsManual] final reduced sum = %lf\n", result_h);
       result_h = 0.0;
    }

    printf("Cloned Graph Output.. \n");
    for (int i=0; i < GRAPH_LAUNCH_ITERATIONS; i++)
    {
       checkHIPErrors(cudaGraphLaunch(clonedGraphExec, streamForGraph));
       checkHIPErrors(hipStreamSynchronize(streamForGraph));
       printf("[cudaGraphsManual] final reduced sum = %lf\n", result_h);
       result_h = 0.0;
    }

    checkHIPErrors(cudaGraphExecDestroy(graphExec));
    checkHIPErrors(cudaGraphExecDestroy(clonedGraphExec));
    checkHIPErrors(cudaGraphDestroy(graph));
    checkHIPErrors(cudaGraphDestroy(clonedGraph));
    checkHIPErrors(hipStreamDestroy(streamForGraph));
}

void cudaGraphsUsingStreamCapture(float* inputVec_h, float *inputVec_d, double *outputVec_d, double *result_d, size_t inputSize, size_t numOfBlocks)
{
    hipStream_t stream1, stream2, stream3, streamForGraph;
    hipEvent_t forkStreamEvent, memsetEvent1, memsetEvent2;
    cudaGraph_t graph;
    double result_h = 0.0;

    checkHIPErrors(hipStreamCreateWithFlags(&stream1, hipStreamNonBlocking));
    checkHIPErrors(hipStreamCreateWithFlags(&stream2, hipStreamNonBlocking));
    checkHIPErrors(hipStreamCreateWithFlags(&stream3, hipStreamNonBlocking));
    checkHIPErrors(hipStreamCreateWithFlags(&streamForGraph, hipStreamNonBlocking));

    checkHIPErrors(hipEventCreate(&forkStreamEvent));
    checkHIPErrors(hipEventCreate(&memsetEvent1));
    checkHIPErrors(hipEventCreate(&memsetEvent2));

    checkHIPErrors(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

    checkHIPErrors(hipEventRecord(forkStreamEvent, stream1));
    checkHIPErrors(hipStreamWaitEvent(stream2, forkStreamEvent, 0));
    checkHIPErrors(hipStreamWaitEvent(stream3, forkStreamEvent, 0));

    checkHIPErrors(hipMemcpyAsync(inputVec_d, inputVec_h, sizeof(float)*inputSize, hipMemcpyDefault, stream1));

    checkHIPErrors(hipMemsetAsync(outputVec_d, 0, sizeof(double)*numOfBlocks, stream2));

    checkHIPErrors(hipEventRecord(memsetEvent1, stream2));

    checkHIPErrors(hipMemsetAsync(result_d, 0, sizeof(double), stream3));
    checkHIPErrors(hipEventRecord(memsetEvent2, stream3));

    checkHIPErrors(hipStreamWaitEvent(stream1, memsetEvent1, 0));

    hipLaunchKernelGGL(reduce, dim3(numOfBlocks), dim3(THREADS_PER_BLOCK), 0, stream1, inputVec_d, outputVec_d, inputSize, numOfBlocks);

    checkHIPErrors(hipStreamWaitEvent(stream1, memsetEvent2, 0));

    hipLaunchKernelGGL(reduceFinal, dim3(1), dim3(THREADS_PER_BLOCK), 0, stream1, outputVec_d, result_d, numOfBlocks);
    checkHIPErrors(hipMemcpyAsync(&result_h, result_d, sizeof(double), hipMemcpyDefault, stream1));

    checkHIPErrors(cudaStreamEndCapture(stream1, &graph));

    cudaGraphNode_t *nodes = NULL;
    size_t numNodes = 0;
    checkHIPErrors(cudaGraphGetNodes(graph, nodes, &numNodes));
    printf("\nNum of nodes in the graph created using stream capture API = %zu\n", numNodes);

    cudaGraphExec_t graphExec;
    checkHIPErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    cudaGraph_t clonedGraph;
    cudaGraphExec_t clonedGraphExec;
    checkHIPErrors(cudaGraphClone(&clonedGraph, graph));
    checkHIPErrors(cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0));

    for (int i=0; i < GRAPH_LAUNCH_ITERATIONS; i++)
    {
       checkHIPErrors(cudaGraphLaunch(graphExec, streamForGraph));
       checkHIPErrors(hipStreamSynchronize(streamForGraph));
       printf("[cudaGraphsUsingStreamCapture] final reduced sum = %lf\n", result_h);
       result_h = 0.0;
    }

    printf("Cloned Graph Output.. \n");
    for (int i=0; i < GRAPH_LAUNCH_ITERATIONS; i++)
    {
       checkHIPErrors(cudaGraphLaunch(clonedGraphExec, streamForGraph));
       checkHIPErrors(hipStreamSynchronize(streamForGraph));
       printf("[cudaGraphsUsingStreamCapture] final reduced sum = %lf\n", result_h);
       result_h = 0.0;
    }

    checkHIPErrors(hipStreamSynchronize(streamForGraph));

    checkHIPErrors(cudaGraphExecDestroy(graphExec));
    checkHIPErrors(cudaGraphExecDestroy(clonedGraphExec));
    checkHIPErrors(cudaGraphDestroy(graph));
    checkHIPErrors(cudaGraphDestroy(clonedGraph));
    checkHIPErrors(hipStreamDestroy(stream1));
    checkHIPErrors(hipStreamDestroy(stream2));
    checkHIPErrors(hipStreamDestroy(streamForGraph));
}

int main(int argc, char **argv)
{
    size_t size = 1<<24;    // number of elements to reduce
    size_t maxBlocks = 512;

    // This will pick the best possible CUDA capable device
    int devID = findHIPDevice(argc, (const char **)argv);

    printf("%zu elements\n", size);
    printf("threads per block  = %d\n", THREADS_PER_BLOCK);
    printf("Graph Launch iterations = %d\n", GRAPH_LAUNCH_ITERATIONS);

    float *inputVec_d = NULL, *inputVec_h = NULL;
    double *outputVec_d = NULL, *result_d;

    inputVec_h = (float*) malloc(sizeof(float)*size);
    checkHIPErrors(hipMalloc(&inputVec_d, sizeof(float)*size));
    checkHIPErrors(hipMalloc(&outputVec_d, sizeof(double)*maxBlocks));
    checkHIPErrors(hipMalloc(&result_d, sizeof(double)));

    init_input(inputVec_h, size);

    cudaGraphsManual(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks);
    cudaGraphsUsingStreamCapture(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks);

    checkHIPErrors(hipFree(inputVec_d));
    checkHIPErrors(hipFree(outputVec_d));
    checkHIPErrors(hipFree(result_d));
    return EXIT_SUCCESS;
}
