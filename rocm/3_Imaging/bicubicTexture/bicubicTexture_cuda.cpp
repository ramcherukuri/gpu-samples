#include "hip/hip_runtime.h"
/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#ifndef _BICUBICTEXTURE_CU_
#define _BICUBICTEXTURE_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_math.h>

// includes, cuda
#include <helper_hip.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#include "bicubicTexture_kernel.cuh"

hipArray *d_imageArray = 0;

extern "C"
void initTexture(int imageWidth, int imageHeight, uchar *h_data)
{
    // allocate array and copy image data
    hipChannelFormatDesc channelDesc = hipCreateChannelDesc(8, 0, 0, 0, hipChannelFormatKindUnsigned);
    checkHIPErrors(hipMallocArray(&d_imageArray, &channelDesc, imageWidth, imageHeight));
    checkHIPErrors(hipMemcpy2DToArray(d_imageArray, 0, 0, h_data, imageWidth * sizeof(uchar), 
                                        imageWidth * sizeof(uchar), imageHeight, hipMemcpyHostToDevice));
    free(h_data);

    hipResourceDesc            texRes;
    memset(&texRes,0,sizeof(hipResourceDesc));

    texRes.resType            = hipResourceTypeArray;
    texRes.res.array.array    = d_imageArray;

    hipTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(hipTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = hipFilterModeLinear;
    texDescr.addressMode[0] = hipAddressModeClamp;
    texDescr.addressMode[1] = hipAddressModeClamp;
    texDescr.readMode = hipReadModeNormalizedFloat;

    checkHIPErrors(hipCreateTextureObject(&texObjLinear, &texRes, &texDescr, NULL));

    memset(&texDescr,0,sizeof(hipTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode       = hipFilterModePoint;
    texDescr.addressMode[0] = hipAddressModeClamp;
    texDescr.addressMode[1] = hipAddressModeClamp;
    texDescr.readMode = hipReadModeNormalizedFloat;

    checkHIPErrors(hipCreateTextureObject(&texObjPoint, &texRes, &texDescr, NULL));
}

extern "C"
void freeTexture()
{
    checkHIPErrors(hipDestroyTextureObject(texObjPoint));
    checkHIPErrors(hipDestroyTextureObject(texObjLinear));
    checkHIPErrors(hipFreeArray(d_imageArray));
}


// render image using CUDA
extern "C"
void render(int width, int height, float tx, float ty, float scale, float cx, float cy,
            dim3 blockSize, dim3 gridSize, int filter_mode, uchar4 *output)
{
    // call CUDA kernel, writing results to PBO memory
    switch (filter_mode)
    {
        case MODE_NEAREST:
            hipLaunchKernelGGL(d_render, dim3(gridSize), dim3(blockSize), 0, 0, output, width, height, tx, ty, scale, cx, cy, texObjPoint);
            break;

        case MODE_BILINEAR:
            hipLaunchKernelGGL(d_render, dim3(gridSize), dim3(blockSize), 0, 0, output, width, height, tx, ty, scale, cx, cy, texObjLinear);
            break;

        case MODE_BICUBIC:
            hipLaunchKernelGGL(d_renderBicubic, dim3(gridSize), dim3(blockSize), 0, 0, output, width, height, tx, ty, scale, cx, cy, texObjPoint);
            break;

        case MODE_FAST_BICUBIC:
            hipLaunchKernelGGL(d_renderFastBicubic, dim3(gridSize), dim3(blockSize), 0, 0, output, width, height, tx, ty, scale, cx, cy, texObjLinear);
            break;

        case MODE_CATROM:
            hipLaunchKernelGGL(d_renderCatRom, dim3(gridSize), dim3(blockSize), 0, 0, output, width, height, tx, ty, scale, cx, cy, texObjPoint);
            break;
    }

    getLastHIPError("kernel failed");
}

#endif
