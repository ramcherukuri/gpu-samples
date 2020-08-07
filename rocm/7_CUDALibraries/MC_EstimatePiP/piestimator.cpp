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

#include "../inc/piestimator.h"

#include <string>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <typeinfo>
#include <hip/hip_runtime.h>
 #include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;
#include <hiprand.h>

using std::string;
using std::vector;

__device__ unsigned int reduce_sum(unsigned int in, cg::thread_block cta)
{
    HIP_DYNAMIC_SHARED( unsigned int, sdata)

    // Perform first level of reduction:
    // - Write to shared memory
    unsigned int ltid = threadIdx.x;

    sdata[ltid] = in;
    cg::sync(cta);

    // Do reduction in shared mem
    for (unsigned int s = blockDim.x / 2 ; s > 0 ; s >>= 1)
    {
        if (ltid < s)
        {
            sdata[ltid] += sdata[ltid + s];
        }

        cg::sync(cta);
    }

    return sdata[0];
}

// Estimator kernel
template <typename Real>
__global__ void computeValue(unsigned int *const results,
                             const Real *const points,
                             const unsigned int numSims)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // Determine thread ID
    unsigned int bid = blockIdx.x;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    // Shift the input/output pointers
    const Real *pointx = points + tid;
    const Real *pointy = pointx + numSims;

    // Count the number of points which lie inside the unit quarter-circle
    unsigned int pointsInside = 0;

    for (unsigned int i = tid ; i < numSims ; i += step, pointx += step, pointy += step)
    {
        Real x = *pointx;
        Real y = *pointy;
        Real l2norm2 = x * x + y * y;

        if (l2norm2 < static_cast<Real>(1))
        {
            pointsInside++;
        }
    }

    // Reduce within the block
    pointsInside = reduce_sum(pointsInside, cta);

    // Store the result
    if (threadIdx.x == 0)
    {
        results[bid] = pointsInside;
    }
}

template <typename Real>
PiEstimator<Real>::PiEstimator(unsigned int numSims, unsigned int device, unsigned int threadBlockSize, unsigned int seed)
    : m_numSims(numSims),
      m_device(device),
      m_threadBlockSize(threadBlockSize),
      m_seed(seed)
{
}

template <typename Real>
Real PiEstimator<Real>::operator()()
{
    hipError_t cudaResult = hipSuccess;
    struct hipDeviceProp_t     deviceProperties;
    struct hipFuncAttributes funcAttributes;

    // Get device properties
    cudaResult = hipGetDeviceProperties(&deviceProperties, m_device);

    if (cudaResult != hipSuccess)
    {
        string msg("Could not get device properties: ");
        msg += hipGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Check precision is valid
    if (typeid(Real) == typeid(double) &&
        (deviceProperties.major < 1 || (deviceProperties.major == 1 && deviceProperties.minor < 3)))
    {
        throw std::runtime_error("Device does not have double precision support");
    }

    // Attach to GPU
    cudaResult = hipSetDevice(m_device);

    if (cudaResult != hipSuccess)
    {
        string msg("Could not set CUDA device: ");
        msg += hipGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Determine how to divide the work between cores
    dim3 block;
    dim3 grid;
    block.x = m_threadBlockSize;
    grid.x  = (m_numSims + m_threadBlockSize - 1) / m_threadBlockSize;

    // Aim to launch around ten or more times as many blocks as there
    // are multiprocessors on the target device.
    unsigned int blocksPerSM = 10;
    unsigned int numSMs      = deviceProperties.multiProcessorCount;

    while (grid.x > 2 * blocksPerSM * numSMs)
    {
        grid.x >>= 1;
    }

    // Get computeValue function properties and check the maximum block size
    cudaResult = hipFuncGetAttributes(&funcAttributes, reinterpret_cast<const void*>(computeValue<Real>));

    if (cudaResult != hipSuccess)
    {
        string msg("Could not get function attributes: ");
        msg += hipGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    if (block.x > (unsigned int)funcAttributes.maxThreadsPerBlock)
    {
        throw std::runtime_error("Block X dimension is too large for computeValue kernel");
    }

    // Check the dimensions are valid
    if (block.x > (unsigned int)deviceProperties.maxThreadsDim[0])
    {
        throw std::runtime_error("Block X dimension is too large for device");
    }

    if (grid.x > (unsigned int)deviceProperties.maxGridSize[0])
    {
        throw std::runtime_error("Grid X dimension is too large for device");
    }

    // Allocate memory for points
    // Each simulation has two random numbers to give X and Y coordinate
    Real *d_points = 0;
    cudaResult = hipMalloc((void **)&d_points, 2 * m_numSims * sizeof(Real));

    if (cudaResult != hipSuccess)
    {
        string msg("Could not allocate memory on device for random numbers: ");
        msg += hipGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Allocate memory for result
    // Each thread block will produce one result
    unsigned int *d_results = 0;
    cudaResult = hipMalloc((void **)&d_results, grid.x * sizeof(unsigned int));

    if (cudaResult != hipSuccess)
    {
        string msg("Could not allocate memory on device for partial results: ");
        msg += hipGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Generate random points in unit square
    hiprandStatus_t curandResult;
    hiprandGenerator_t prng;
    curandResult = hiprandCreateGenerator(&prng, HIPRAND_RNG_PSEUDO_DEFAULT);

    if (curandResult != HIPRAND_STATUS_SUCCESS)
    {
        string msg("Could not create pseudo-random number generator: ");
        msg += curandResult;
        throw std::runtime_error(msg);
    }

    curandResult = hiprandSetPseudoRandomGeneratorSeed(prng, m_seed);

    if (curandResult != HIPRAND_STATUS_SUCCESS)
    {
        string msg("Could not set seed for pseudo-random number generator: ");
        msg += curandResult;
        throw std::runtime_error(msg);
    }

    if (typeid(Real) == typeid(float))
    {
        curandResult = hiprandGenerateUniform(prng, (float *)d_points, 2 * m_numSims);
    }
    else if (typeid(Real) == typeid(double))
    {
        curandResult = hiprandGenerateUniformDouble(prng, (double *)d_points, 2 * m_numSims);
    }
    else
    {
        string msg("Could not generate random numbers of specified type");
        throw std::runtime_error(msg);
    }

    if (curandResult != HIPRAND_STATUS_SUCCESS)
    {
        string msg("Could not generate pseudo-random numbers: ");
        msg += curandResult;
        throw std::runtime_error(msg);
    }

    curandResult = hiprandDestroyGenerator(prng);

    if (curandResult != HIPRAND_STATUS_SUCCESS)
    {
        string msg("Could not destroy pseudo-random number generator: ");
        msg += curandResult;
        throw std::runtime_error(msg);
    }

    // Count the points inside unit quarter-circle
    hipLaunchKernelGGL(HIP_KERNEL_NAME(computeValue<Real>), dim3(grid), dim3(block), block.x *sizeof(unsigned int), 0, d_results, d_points, m_numSims);

    // Copy partial results back
    vector<unsigned int> results(grid.x);
    cudaResult = hipMemcpy(&results[0], d_results, grid.x * sizeof(unsigned int), hipMemcpyDeviceToHost);

    if (cudaResult != hipSuccess)
    {
        string msg("Could not copy partial results to host: ");
        msg += hipGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Complete sum-reduction on host
    Real value = static_cast<Real>(std::accumulate(results.begin(), results.end(), 0));

    // Determine the proportion of points inside the quarter-circle,
    // i.e. the area of the unit quarter-circle
    value /= m_numSims;

    // Value is currently an estimate of the area of a unit quarter-circle, so we can
    // scale to a full circle by multiplying by four. Now since the area of a circle
    // is pi * r^2, and r is one, the value will be an estimate for the value of pi.
    value *= 4;

    // Cleanup
    if (d_points)
    {
        hipFree(d_points);
        d_points = 0;
    }

    if (d_results)
    {
        hipFree(d_results);
        d_results = 0;
    }

    return value;
}

// Explicit template instantiation
template class PiEstimator<float>;
template class PiEstimator<double>;