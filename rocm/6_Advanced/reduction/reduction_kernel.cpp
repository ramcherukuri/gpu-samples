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
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <stdio.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        HIP_DYNAMIC_SHARED( int, __smem)
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        HIP_DYNAMIC_SHARED( int, __smem)
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        HIP_DYNAMIC_SHARED( double, __smem_d)
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        HIP_DYNAMIC_SHARED( double, __smem_d)
        return (double *)__smem_d;
    }
};

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

/* This reduction interleaves which threads are active by using the modulo
   operator.  This operator is very expensive on GPUs, and the interleaved
   inactivity means that no whole warps are active, which is also very
   inefficient */
template <class T>
__global__ void
reduce0(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/* This version uses contiguous threads, but its interleaved
   addressing results in many shared memory bank conflicts.
*/
template <class T>
__global__ void
reduce1(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version uses sequential addressing -- no divergence or bank conflicts.
*/
template <class T>
__global__ void
reduce2(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version uses n/2 threads --
    it performs the first level of reduction when reading from global memory.
*/
template <class T>
__global__ void
reduce3(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockDim.x < n)
        mySum += g_idata[i+blockDim.x];

    sdata[tid] = mySum;
    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

/*
    This version uses the warp shuffle operation if available to reduce 
    warp synchronization. When shuffle is not available the final warp's
    worth of work is unrolled to reduce looping overhead.

    See http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
    for additional information about using shuffle to perform a reduction
    within a warp.

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize>
__global__ void
reduce4(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockSize < n)
        mySum += g_idata[i+blockSize];

    sdata[tid] = mySum;
    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>32; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }

        cg::sync(cta);
    }

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    if (cta.thread_rank() < 32)
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = tile32.size()/2; offset > 0; offset /= 2) 
        {
             mySum += tile32.shfl_down(mySum, offset);
        }
    }

    // write result for this block to global mem
    if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}

/*
    This version is completely unrolled, unless warp shuffle is available, then
    shuffle is used within a loop.  It uses a template parameter to achieve
    optimal code for any (power of 2) number of threads.  This requires a switch
    statement in the host code to handle all the different thread block sizes at
    compile time. When shuffle is available, it is used to reduce warp synchronization.

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize>
__global__ void
reduce5(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockSize < n)
        mySum += g_idata[i+blockSize];

    sdata[tid] = mySum;
    cg::sync(cta);

    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    cg::sync(cta);

    if ((blockSize >= 256) &&(tid < 128))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

    cg::sync(cta);

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    if (cta.thread_rank() < 32)
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = tile32.size()/2; offset > 0; offset /= 2) 
        {
             mySum += tile32.shfl_down(mySum, offset);
        }
    }

    // write result for this block to global mem
    if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    cg::sync(cta);


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    cg::sync(cta);

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

    cg::sync(cta);

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    if (cta.thread_rank() < 32)
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = tile32.size()/2; offset > 0; offset /= 2) 
        {
             mySum += tile32.shfl_down(mySum, offset);
        }
    }

    // write result for this block to global mem
    if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}


extern "C"
bool isPow2(unsigned int x);


////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void
reduce(int size, int threads, int blocks,
       int whichKernel, T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    switch (whichKernel)
    {
        case 0:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce0<T>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
            break;

        case 1:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce1<T>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
            break;

        case 2:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce2<T>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
            break;

        case 3:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce3<T>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
            break;

        case 4:
            switch (threads)
            {
                case 512:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce4<T, 512>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;

                case 256:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce4<T, 256>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;

                case 128:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce4<T, 128>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;

                case 64:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce4<T,  64>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;

                case 32:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce4<T,  32>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;

                case 16:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce4<T,  16>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;

                case  8:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce4<T,   8>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;

                case  4:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce4<T,   4>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;

                case  2:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce4<T,   2>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;

                case  1:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce4<T,   1>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;
            }

            break;

        case 5:
            switch (threads)
            {
                case 512:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce5<T, 512>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;

                case 256:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce5<T, 256>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;

                case 128:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce5<T, 128>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;

                case 64:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce5<T,  64>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;

                case 32:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce5<T,  32>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;

                case 16:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce5<T,  16>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;

                case  8:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce5<T,   8>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;

                case  4:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce5<T,   4>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;

                case  2:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce5<T,   2>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;

                case  1:
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce5<T,   1>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                    break;
            }

            break;

        case 6:
        default:
            if (isPow2(size))
            {
                switch (threads)
                {
                    case 512:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T, 512, true>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;

                    case 256:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T, 256, true>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;

                    case 128:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T, 128, true>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;

                    case 64:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T,  64, true>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;

                    case 32:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T,  32, true>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;

                    case 16:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T,  16, true>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;

                    case  8:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T,   8, true>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;

                    case  4:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T,   4, true>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;

                    case  2:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T,   2, true>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;

                    case  1:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T,   1, true>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;
                }
            }
            else
            {
                switch (threads)
                {
                    case 512:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T, 512, false>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;

                    case 256:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T, 256, false>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;

                    case 128:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T, 128, false>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;

                    case 64:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T,  64, false>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;

                    case 32:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T,  32, false>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;

                    case 16:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T,  16, false>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;

                    case  8:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T,   8, false>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;

                    case  4:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T,   4, false>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;

                    case  2:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T,   2, false>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;

                    case  1:
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce6<T,   1, false>), dim3(dimGrid), dim3(dimBlock), smemSize , 0, d_idata, d_odata, size);
                        break;
                }
            }

            break;
    }
}

// Instantiate the reduction function for 3 types
template void
reduce<int>(int size, int threads, int blocks,
            int whichKernel, int *d_idata, int *d_odata);

template void
reduce<float>(int size, int threads, int blocks,
              int whichKernel, float *d_idata, float *d_odata);

template void
reduce<double>(int size, int threads, int blocks,
               int whichKernel, double *d_idata, double *d_odata);


#endif // #ifndef _REDUCE_KERNEL_H_
