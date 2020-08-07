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


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_hip.h>
#include "convolutionFFT2D_common.h"
#include "convolutionFFT2D.cuh"

////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
extern "C" void padKernel(
    float *d_Dst,
    float *d_Src,
    int fftH,
    int fftW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
)
{
    assert(d_Src != d_Dst);
    dim3 threads(32, 8);
    dim3 grid(iDivUp(kernelW, threads.x), iDivUp(kernelH, threads.y));

    SET_FLOAT_BASE;
#if (USE_TEXTURE)
    hipTextureObject_t texFloat;
    hipResourceDesc            texRes;
    memset(&texRes,0,sizeof(hipResourceDesc));

    texRes.resType            = hipResourceTypeLinear;
    texRes.res.linear.devPtr    = d_Src;
    texRes.res.linear.sizeInBytes = sizeof(float)*kernelH*kernelW;
    texRes.res.linear.desc = hipCreateChannelDesc<float>();

    hipTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(hipTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = hipFilterModeLinear;
    texDescr.addressMode[0] = hipAddressModeWrap;
    texDescr.readMode = hipReadModeElementType;

    checkHIPErrors(hipCreateTextureObject(&texFloat, &texRes, &texDescr, NULL));
#endif

    hipLaunchKernelGGL(padKernel_kernel, dim3(grid), dim3(threads), 0, 0, 
        d_Dst,
        d_Src,
        fftH,
        fftW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
#if (USE_TEXTURE)
        , texFloat
#endif
    );
    getLastHIPError("padKernel_kernel<<<>>> execution failed\n");

#if (USE_TEXTURE)
    checkHIPErrors(hipDestroyTextureObject(texFloat));
#endif
}



////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
extern "C" void padDataClampToBorder(
    float *d_Dst,
    float *d_Src,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
    int kernelW,
    int kernelH,
    int kernelY,
    int kernelX
)
{
    assert(d_Src != d_Dst);
    dim3 threads(32, 8);
    dim3 grid(iDivUp(fftW, threads.x), iDivUp(fftH, threads.y));

#if (USE_TEXTURE)
    hipTextureObject_t texFloat;
    hipResourceDesc            texRes;
    memset(&texRes,0,sizeof(hipResourceDesc));

    texRes.resType            = hipResourceTypeLinear;
    texRes.res.linear.devPtr    = d_Src;
    texRes.res.linear.sizeInBytes = sizeof(float)*dataH*dataW;
    texRes.res.linear.desc = hipCreateChannelDesc<float>();

    hipTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(hipTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = hipFilterModeLinear;
    texDescr.addressMode[0] = hipAddressModeWrap;
    texDescr.readMode = hipReadModeElementType;

    checkHIPErrors(hipCreateTextureObject(&texFloat, &texRes, &texDescr, NULL));
#endif

    hipLaunchKernelGGL(padDataClampToBorder_kernel, dim3(grid), dim3(threads), 0, 0, 
        d_Dst,
        d_Src,
        fftH,
        fftW,
        dataH,
        dataW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
#if (USE_TEXTURE)
       ,texFloat
#endif
    );
    getLastHIPError("padDataClampToBorder_kernel<<<>>> execution failed\n");

#if (USE_TEXTURE)
    checkHIPErrors(hipDestroyTextureObject(texFloat));
#endif
}



////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
extern "C" void modulateAndNormalize(
    fComplex *d_Dst,
    fComplex *d_Src,
    int fftH,
    int fftW,
    int padding
)
{
    assert(fftW % 2 == 0);
    const int dataSize = fftH * (fftW / 2 + padding);

    hipLaunchKernelGGL(modulateAndNormalize_kernel, dim3(iDivUp(dataSize), dim3(256)), 256, 0, 
        d_Dst,
        d_Src,
        dataSize,
        1.0f / (float)(fftW *fftH)
    );
    getLastHIPError("modulateAndNormalize() execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// 2D R2C / C2R post/preprocessing kernels
////////////////////////////////////////////////////////////////////////////////
static const double PI = 3.1415926535897932384626433832795;
static const uint BLOCKDIM = 256;

extern "C" void spPostprocess2D(
    void *d_Dst,
    void *d_Src,
    uint DY,
    uint DX,
    uint padding,
    int dir
)
{
    assert(d_Src != d_Dst);
    assert(DX % 2 == 0);

#if(POWER_OF_TWO)
    uint log2DX, log2DY;
    uint factorizationRemX = factorRadix2(log2DX, DX);
    uint factorizationRemY = factorRadix2(log2DY, DY);
    assert(factorizationRemX == 1 && factorizationRemY == 1);
#endif

    const uint threadCount = DY * (DX / 2);
    const double phaseBase = dir * PI / (double)DX;

#if (USE_TEXTURE)
    hipTextureObject_t texComplex;
    hipResourceDesc            texRes;
    memset(&texRes,0,sizeof(hipResourceDesc));

    texRes.resType            = hipResourceTypeLinear;
    texRes.res.linear.devPtr    = d_Src;
    texRes.res.linear.sizeInBytes = sizeof(fComplex)*DY*(DX+padding);
    texRes.res.linear.desc = hipCreateChannelDesc<fComplex>();

    hipTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(hipTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = hipFilterModeLinear;
    texDescr.addressMode[0]   = hipAddressModeWrap;
    texDescr.readMode         = hipReadModeElementType;

    checkHIPErrors(hipCreateTextureObject(&texComplex, &texRes, &texDescr, NULL));
#endif

    hipLaunchKernelGGL(spPostprocess2D_kernel, dim3(iDivUp(threadCount), dim3(BLOCKDIM)), BLOCKDIM, 0, 
        (fComplex *)d_Dst,
        (fComplex *)d_Src,
        DY, DX, threadCount, padding,
        (float)phaseBase
#if (USE_TEXTURE)
        , texComplex
#endif
    );
    getLastHIPError("spPostprocess2D_kernel<<<>>> execution failed\n");

#if (USE_TEXTURE)
    checkHIPErrors(hipDestroyTextureObject(texComplex));
#endif
}

extern "C" void spPreprocess2D(
    void *d_Dst,
    void *d_Src,
    uint DY,
    uint DX,
    uint padding,
    int dir
)
{
    assert(d_Src != d_Dst);
    assert(DX % 2 == 0);

#if(POWER_OF_TWO)
    uint log2DX, log2DY;
    uint factorizationRemX = factorRadix2(log2DX, DX);
    uint factorizationRemY = factorRadix2(log2DY, DY);
    assert(factorizationRemX == 1 && factorizationRemY == 1);
#endif

    const uint threadCount = DY * (DX / 2);
    const double phaseBase = -dir * PI / (double)DX;

#if (USE_TEXTURE)
    hipTextureObject_t     texComplex;
    hipResourceDesc        texRes;
    memset(&texRes,0,sizeof(hipResourceDesc));

    texRes.resType                  = hipResourceTypeLinear;
    texRes.res.linear.devPtr        = d_Src;
    texRes.res.linear.sizeInBytes   = sizeof(fComplex)*DY*(DX+padding);
    texRes.res.linear.desc = hipCreateChannelDesc<fComplex>();

    hipTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(hipTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = hipFilterModeLinear;
    texDescr.addressMode[0] = hipAddressModeWrap;
    texDescr.readMode = hipReadModeElementType;

    checkHIPErrors(hipCreateTextureObject(&texComplex, &texRes, &texDescr, NULL));
#endif
    hipLaunchKernelGGL(spPreprocess2D_kernel, dim3(iDivUp(threadCount), dim3(BLOCKDIM)), BLOCKDIM, 0, 
        (fComplex *)d_Dst,
        (fComplex *)d_Src,
        DY, DX, threadCount, padding,
        (float)phaseBase
#if (USE_TEXTURE)
        , texComplex
#endif
    );
    getLastHIPError("spPreprocess2D_kernel<<<>>> execution failed\n");

#if (USE_TEXTURE)
    checkHIPErrors(hipDestroyTextureObject(texComplex));
#endif
}



////////////////////////////////////////////////////////////////////////////////
// Combined spPostprocess2D + modulateAndNormalize + spPreprocess2D
////////////////////////////////////////////////////////////////////////////////
extern "C" void spProcess2D(
    void *d_Dst,
    void *d_SrcA,
    void *d_SrcB,
    uint DY,
    uint DX,
    int dir
)
{
    assert(DY % 2 == 0);

#if(POWER_OF_TWO)
    uint log2DX, log2DY;
    uint factorizationRemX = factorRadix2(log2DX, DX);
    uint factorizationRemY = factorRadix2(log2DY, DY);
    assert(factorizationRemX == 1 && factorizationRemY == 1);
#endif

    const uint threadCount = (DY / 2) * DX;
    const double phaseBase = dir * PI / (double)DX;

#if (USE_TEXTURE)
    hipTextureObject_t texComplexA, texComplexB;
    hipResourceDesc            texRes;
    memset(&texRes,0,sizeof(hipResourceDesc));

    texRes.resType            = hipResourceTypeLinear;
    texRes.res.linear.devPtr    = d_SrcA;
    texRes.res.linear.sizeInBytes = sizeof(fComplex)*DY*DX;
    texRes.res.linear.desc = hipCreateChannelDesc<fComplex>();

    hipTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(hipTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = hipFilterModeLinear;
    texDescr.addressMode[0] = hipAddressModeWrap;
    texDescr.readMode = hipReadModeElementType;

    checkHIPErrors(hipCreateTextureObject(&texComplexA, &texRes, &texDescr, NULL));

    memset(&texRes,0,sizeof(hipResourceDesc));

    texRes.resType            = hipResourceTypeLinear;
    texRes.res.linear.devPtr    = d_SrcB;
    texRes.res.linear.sizeInBytes = sizeof(fComplex)*DY*DX;
    texRes.res.linear.desc = hipCreateChannelDesc<fComplex>();

    memset(&texDescr,0,sizeof(hipTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = hipFilterModeLinear;
    texDescr.addressMode[0] = hipAddressModeWrap;
    texDescr.readMode = hipReadModeElementType;

    checkHIPErrors(hipCreateTextureObject(&texComplexB, &texRes, &texDescr, NULL));
#endif
    hipLaunchKernelGGL(spProcess2D_kernel, dim3(iDivUp(threadCount), dim3(BLOCKDIM)), BLOCKDIM, 0, 
        (fComplex *)d_Dst,
        (fComplex *)d_SrcA,
        (fComplex *)d_SrcB,
        DY, DX, threadCount,
        (float)phaseBase,
        0.5f / (float)(DY *DX)
#if (USE_TEXTURE)
        , texComplexA
        , texComplexB
#endif    
        );
    getLastHIPError("spProcess2D_kernel<<<>>> execution failed\n");

#if (USE_TEXTURE)
    checkHIPErrors(hipDestroyTextureObject(texComplexA));
    checkHIPErrors(hipDestroyTextureObject(texComplexB));
#endif
}
