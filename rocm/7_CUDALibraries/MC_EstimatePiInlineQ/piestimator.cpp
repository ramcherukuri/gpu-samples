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
#include <hiprand_kernel.h>

#include "../inc/cudasharedmem.h"

using std::string;
using std::vector;

// Helper templates to support float and double in same code
template <typename L, typename R>
struct TYPE_IS
{
    static const bool test = false;
};
template <typename L>
struct TYPE_IS<L, L>
{
    static const bool test = true;
};
template <bool, class L, class R>
struct IF
{
    typedef R type;
};
template <class L, class R>
struct IF<true, L, R>
{
    typedef L type;
};

// RNG init kernel
template <typename rngState_t, typename rngDirectionVectors_t>
__global__ void initRNG(rngState_t *const rngStates,
                        rngDirectionVectors_t *const rngDirections,
                        unsigned int numDrawsPerDirection)
{
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    // Determine offset to avoid overlapping sub-sequences
    unsigned int offset = tid * ((numDrawsPerDirection + step - 1) / step);

    // Initialise the RNG
    hiprand_init(rngDirections[0], offset, &rngStates[tid]);
    hiprand_init(rngDirections[1], offset, &rngStates[tid + step]);
}

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

__device__ inline void getPoint(float &x, float &y, hiprandStateSobol32 &state1, hiprandStateSobol32 &state2)
{
    x = hiprand_uniform(&state1);
    y = hiprand_uniform(&state2);
}
__device__ inline void getPoint(double &x, double &y, curandStateSobol64 &state1, curandStateSobol64 &state2)
{
    x = hiprand_uniform_double(&state1);
    y = hiprand_uniform_double(&state2);
}

// Estimator kernel
template <typename Real, typename rngState_t>
__global__ void computeValue(unsigned int *const results,
                             rngState_t *const rngStates,
                             const unsigned int numSims)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // Determine thread ID
    unsigned int bid = blockIdx.x;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    // Initialise the RNG
    rngState_t localState1 = rngStates[tid];
    rngState_t localState2 = rngStates[tid + step];

    // Count the number of points which lie inside the unit quarter-circle
    unsigned int pointsInside = 0;

    for (unsigned int i = tid ; i < numSims ; i += step)
    {
        Real x;
        Real y;
        getPoint(x, y, localState1, localState2);
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
PiEstimator<Real>::PiEstimator(unsigned int numSims, unsigned int device, unsigned int threadBlockSize)
    : m_numSims(numSims),
      m_device(device),
      m_threadBlockSize(threadBlockSize)
{
}

template <typename Real>
Real PiEstimator<Real>::operator()()
{
    hipError_t cudaResult = hipSuccess;
    struct hipDeviceProp_t     deviceProperties;
    struct hipFuncAttributes funcAttributes;

    // Determine type of generator to use (32- or 64-bit)
    typedef typename IF<TYPE_IS<Real, double>::test, curandStateSobol64_t, hiprandStateSobol32_t>::type             curandStateSobol_sz;
    typedef typename IF<TYPE_IS<Real, double>::test, curandDirectionVectors64_t, hiprandDirectionVectors32_t>::type curandDirectionVectors_sz;

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

    // Get initRNG function properties and check the maximum block size
    cudaResult = hipFuncGetAttributes(&funcAttributes, reinterpret_cast<const void*>(initRNG<curandStateSobol_sz), curandDirectionVectors_sz>);

    if (cudaResult != hipSuccess)
    {
        string msg("Could not get function attributes: ");
        msg += hipGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    if (block.x > (unsigned int)funcAttributes.maxThreadsPerBlock)
    {
        throw std::runtime_error("Block X dimension is too large for initRNG kernel");
    }

    // Get computeValue function properties and check the maximum block size
    cudaResult = hipFuncGetAttributes(&funcAttributes, reinterpret_cast<const void*>(computeValue<Real), curandStateSobol_sz>);

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

    // Allocate memory for RNG states and direction vectors
    curandStateSobol_sz       *d_rngStates     = 0;
    curandDirectionVectors_sz *d_rngDirections = 0;
    cudaResult = hipMalloc((void **)&d_rngStates, 2 * grid.x * block.x * sizeof(curandStateSobol_sz));

    if (cudaResult != hipSuccess)
    {
        string msg("Could not allocate memory on device for RNG states: ");
        msg += hipGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    cudaResult = hipMalloc((void **)&d_rngDirections, 2 * sizeof(curandDirectionVectors_sz));

    if (cudaResult != hipSuccess)
    {
        string msg("Could not allocate memory on device for RNG direction vectors: ");
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

    // Generate direction vectors on the host and copy to the device
    if (typeid(Real) == typeid(float))
    {
        hiprandDirectionVectors32_t *rngDirections;
        hiprandStatus_t curandResult = curandGetDirectionVectors32(&rngDirections, CURAND_DIRECTION_VECTORS_32_JOEKUO6);

        if (curandResult != HIPRAND_STATUS_SUCCESS)
        {
            string msg("Could not get direction vectors for quasi-random number generator: ");
            msg += curandResult;
            throw std::runtime_error(msg);
        }

        cudaResult = hipMemcpy(d_rngDirections, rngDirections, 2 * sizeof(hiprandDirectionVectors32_t), hipMemcpyHostToDevice);

        if (cudaResult != hipSuccess)
        {
            string msg("Could not copy direction vectors to device: ");
            msg += hipGetErrorString(cudaResult);
            throw std::runtime_error(msg);
        }
    }
    else if (typeid(Real) == typeid(double))
    {
        curandDirectionVectors64_t *rngDirections;
        hiprandStatus_t curandResult = curandGetDirectionVectors64(&rngDirections, CURAND_DIRECTION_VECTORS_64_JOEKUO6);

        if (curandResult != HIPRAND_STATUS_SUCCESS)
        {
            string msg("Could not get direction vectors for quasi-random number generator: ");
            msg += curandResult;
            throw std::runtime_error(msg);
        }

        cudaResult = hipMemcpy(d_rngDirections, rngDirections, 2 * sizeof(curandDirectionVectors64_t), hipMemcpyHostToDevice);

        if (cudaResult != hipSuccess)
        {
            string msg("Could not copy direction vectors to device: ");
            msg += hipGetErrorString(cudaResult);
            throw std::runtime_error(msg);
        }
    }
    else
    {
        string msg("Could not get direction vectors for random number generator of specified type");
        throw std::runtime_error(msg);
    }

    // Initialise RNG
    hipLaunchKernelGGL(initRNG, dim3(grid), dim3(block), 0, 0, d_rngStates, d_rngDirections, m_numSims);

    // Count the points inside unit quarter-circle
    hipLaunchKernelGGL(HIP_KERNEL_NAME(computeValue<Real>), dim3(grid), dim3(block), block.x *sizeof(unsigned int), 0, d_results, d_rngStates, m_numSims);

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
    if (d_rngStates)
    {
        hipFree(d_rngStates);
        d_rngStates = 0;
    }

    if (d_rngDirections)
    {
        hipFree(d_rngDirections);
        d_rngDirections = 0;
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
