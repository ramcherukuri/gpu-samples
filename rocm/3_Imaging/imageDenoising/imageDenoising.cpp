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
 * This sample demonstrates two adaptive image denoising techniques:
 * KNN and NLM, based on computation of both geometric and color distance
 * between texels. While both techniques are already implemented in the
 * DirectX SDK using shaders, massively speeded up variation
 * of the latter technique, taking advantage of shared memory, is implemented
 * in addition to DirectX counterparts.
 * See supplied whitepaper for more explanations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_hip.h>
#include "imageDenoising.h"


////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
float Max(float x, float y)
{
    return (x > y) ? x : y;
}

float Min(float x, float y)
{
    return (x < y) ? x : y;
}

int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__device__ float lerpf(float a, float b, float c)
{
    return a + (b - a) * c;
}

__device__ float vecLen(float4 a, float4 b)
{
    return (
               (b.x - a.x) * (b.x - a.x) +
               (b.y - a.y) * (b.y - a.y) +
               (b.z - a.z) * (b.z - a.z)
           );
}

__device__ TColor make_color(float r, float g, float b, float a)
{
    return
        ((int)(a * 255.0f) << 24) |
        ((int)(b * 255.0f) << 16) |
        ((int)(g * 255.0f) <<  8) |
        ((int)(r * 255.0f) <<  0);
}



////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//Texture object and channel descriptor for image texture
hipTextureObject_t texImage;
hipChannelFormatDesc uchar4tex = hipCreateChannelDesc<uchar4>();

//CUDA array descriptor
hipArray *a_Src;

////////////////////////////////////////////////////////////////////////////////
// Filtering kernels
////////////////////////////////////////////////////////////////////////////////
#include "imageDenoising_copy_kernel.cuh"
#include "imageDenoising_knn_kernel.cuh"
#include "imageDenoising_nlm_kernel.cuh"
#include "imageDenoising_nlm2_kernel.cuh"

extern "C"
hipError_t CUDA_MallocArray(uchar4 **h_Src, int imageW, int imageH)
{
    hipError_t error;

    error = hipMallocArray(&a_Src, &uchar4tex, imageW, imageH);
    error = hipMemcpyToArray(a_Src, 0, 0,
                              *h_Src, imageW * imageH * sizeof(uchar4),
                              hipMemcpyHostToDevice
                             );

    hipResourceDesc            texRes;
    memset(&texRes,0,sizeof(hipResourceDesc));

    texRes.resType            = hipResourceTypeArray;
    texRes.res.array.array    = a_Src;

    hipTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(hipTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = hipFilterModeLinear;
    texDescr.addressMode[0] = hipAddressModeWrap;
    texDescr.addressMode[1] = hipAddressModeWrap;
    texDescr.readMode = hipReadModeNormalizedFloat;

    checkHIPErrors(hipCreateTextureObject(&texImage, &texRes, &texDescr, NULL));

    return error;
}


extern "C"
hipError_t CUDA_FreeArray()
{
    return hipFreeArray(a_Src);
}
