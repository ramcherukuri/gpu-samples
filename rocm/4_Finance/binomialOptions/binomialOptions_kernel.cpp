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

////////////////////////////////////////////////////////////////////////////////
// Global types and parameters
////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

#include <helper_hip.h>
#include "binomialOptions_common.h"
#include "realtype.h"


//Preprocessed input option data
typedef struct
{
    real S;
    real X;
    real vDt;
    real puByDf;
    real pdByDf;
} __TOptionData;
static __constant__ __TOptionData d_OptionData[MAX_OPTIONS];
static __device__           real d_CallValue[MAX_OPTIONS];



////////////////////////////////////////////////////////////////////////////////
// Overloaded shortcut functions for different precision modes
////////////////////////////////////////////////////////////////////////////////
#ifndef DOUBLE_PRECISION
__device__ inline float expiryCallValue(float S, float X, float vDt, int i)
{
    float d = S * __expf(vDt * (2.0f * i - NUM_STEPS)) - X;
    return (d > 0.0F) ? d : 0.0F;
}
#else
__device__ inline double expiryCallValue(double S, double X, double vDt, int i)
{
    double d = S * exp(vDt * (2.0 * i - NUM_STEPS)) - X;
    return (d > 0.0) ? d : 0.0;
}
#endif


////////////////////////////////////////////////////////////////////////////////
// GPU kernel
////////////////////////////////////////////////////////////////////////////////
#define THREADBLOCK_SIZE 128
#define ELEMS_PER_THREAD (NUM_STEPS/THREADBLOCK_SIZE)
#if NUM_STEPS % THREADBLOCK_SIZE
#error Bad constants
#endif

__global__ void binomialOptionsKernel()
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ real call_exchange[THREADBLOCK_SIZE + 1];

    const int     tid = threadIdx.x;
    const real      S = d_OptionData[blockIdx.x].S;
    const real      X = d_OptionData[blockIdx.x].X;
    const real    vDt = d_OptionData[blockIdx.x].vDt;
    const real puByDf = d_OptionData[blockIdx.x].puByDf;
    const real pdByDf = d_OptionData[blockIdx.x].pdByDf;

    real call[ELEMS_PER_THREAD + 1];
    #pragma unroll
    for(int i = 0; i < ELEMS_PER_THREAD; ++i)
        call[i] = expiryCallValue(S, X, vDt, tid * ELEMS_PER_THREAD + i);

    if (tid == 0)
        call_exchange[THREADBLOCK_SIZE] = expiryCallValue(S, X, vDt, NUM_STEPS);

    int final_it = max(0, tid * ELEMS_PER_THREAD - 1);

    #pragma unroll 16
    for(int i = NUM_STEPS; i > 0; --i)
    {
        call_exchange[tid] = call[0];
        cg::sync(cta);
        call[ELEMS_PER_THREAD] = call_exchange[tid + 1];
        cg::sync(cta);

        if (i > final_it)
        {
           #pragma unroll
           for(int j = 0; j < ELEMS_PER_THREAD; ++j)
              call[j] = puByDf * call[j + 1] + pdByDf * call[j];
        }
    }

    if (tid == 0)
    {
        d_CallValue[blockIdx.x] = call[0];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Host-side interface to GPU binomialOptions
////////////////////////////////////////////////////////////////////////////////
extern "C" void binomialOptionsGPU(
    real *callValue,
    TOptionData  *optionData,
    int optN
)
{
    __TOptionData h_OptionData[MAX_OPTIONS];

    for (int i = 0; i < optN; i++)
    {
        const real      T = optionData[i].T;
        const real      R = optionData[i].R;
        const real      V = optionData[i].V;

        const real     dt = T / (real)NUM_STEPS;
        const real    vDt = V * sqrt(dt);
        const real    rDt = R * dt;
        //Per-step interest and discount factors
        const real     If = exp(rDt);
        const real     Df = exp(-rDt);
        //Values and pseudoprobabilities of upward and downward moves
        const real      u = exp(vDt);
        const real      d = exp(-vDt);
        const real     pu = (If - d) / (u - d);
        const real     pd = (real)1.0 - pu;
        const real puByDf = pu * Df;
        const real pdByDf = pd * Df;

        h_OptionData[i].S      = (real)optionData[i].S;
        h_OptionData[i].X      = (real)optionData[i].X;
        h_OptionData[i].vDt    = (real)vDt;
        h_OptionData[i].puByDf = (real)puByDf;
        h_OptionData[i].pdByDf = (real)pdByDf;
    }

    checkHIPErrors(hipMemcpyToSymbol(HIP_SYMBOL(d_OptionData), h_OptionData, optN * sizeof(__TOptionData)));
    hipLaunchKernelGGL(binomialOptionsKernel, dim3(optN), dim3(THREADBLOCK_SIZE), 0, 0);
    getLastHIPError("binomialOptionsKernel() execution failed.\n");
    checkHIPErrors(hipMemcpyFromSymbol(callValue, HIP_SYMBOL(d_CallValue), optN *sizeof(real)));
}