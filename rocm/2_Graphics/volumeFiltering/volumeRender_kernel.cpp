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

// Simple 3D volume renderer

#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_

#include <helper_hip.h>
#include <helper_math.h>
#include "volumeRender.h"

#define VOLUMERENDER_TFS              2
#define VOLUMERENDER_TF_PREINTSIZE    1024
#define VOLUMERENDER_TF_PREINTSTEPS   1024
#define VOLUMERENDER_TF_PREINTRAY     4

enum TFMode
{
    TF_SINGLE_1D = 0,         // single 1D TF for everything
    TF_LAYERED_2D_PREINT = 1, // layered 2D TF uses pre-integration
    TF_LAYERED_2D = 2,        // layered 2D TF without pre-integration behavior
};

typedef unsigned int  uint;
typedef unsigned char uchar;

static bool usePreInt = true;
static hipArray *d_transferIntegrate = 0;
static hipArray *d_transferFunc = 0;
static hipArray *d_transferArray = 0;

// 1D transfer function texture
hipTextureObject_t transferTex;
// 1D transfer integration texture
hipTextureObject_t transferIntegrateTex;
hipSurfaceObject_t transferIntegrateSurf;
// 2D layered preintegrated transfer function texture
hipTextureObject_t transferLayerPreintTex;
hipSurfaceObject_t transferLayerPreintSurf;

typedef struct
{
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray
{
    float3 o;    // origin
    float3 d;    // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin  = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}


template <int TFMODE >
__device__ void
d_render(uint *d_output, uint imageW, uint imageH,
         float density, float brightness,
         float transferOffset, float transferScale, hipTextureObject_t volumeTex,
         hipTextureObject_t transferTex, hipTextureObject_t transferLayerPreintTex,
         float transferWeight = 0.0f)
{
    const float rayscale =  float(TFMODE != TF_SINGLE_1D ? VOLUMERENDER_TF_PREINTRAY : 1);
    const int maxSteps = 512;
    const float tstep = 0.01f * rayscale;
    const float opacityThreshold = 0.95f;
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

    density *= rayscale;

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;

    float lastsample = 0;

    //lastsample = (lastsample-transferOffset)*transferScale;
    for (int i=0; i<maxSteps; i++)
    {
        // read from 3D texture
        // remap position to [0, 1] coordinates
        float3 coord = make_float3(pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
        float sample = tex3D<float>(volumeTex, coord.x, coord.y, coord.z);
        //sample = (sample-transferOffset)*transferScale;
        //sample *= 64.0f;    // scale for 10-bit data

        // lookup in transfer function texture
        float4 col;
        int tfid = (pos.x < 0);

        if (TFMODE != TF_SINGLE_1D)
        {
            col = tex2DLayered<float4>(transferLayerPreintTex, sample, TFMODE==TF_LAYERED_2D ? sample : lastsample, tfid);
            col.w *= density;
            lastsample = sample;
        }
        else
        {
            col = tex1D<float4>(transferTex, sample);
            col.w *= 0;
        }

        // "under" operator for back-to-front blending
        //sum = lerp(sum, col, col.w);


        // pre-multiply alpha
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;
        // "over" operator for front-to-back blending
        sum = sum + col*(1.0f - sum.w);

        // exit early if opaque
        if (sum.w > opacityThreshold)
            break;

        t += tstep;

        if (t > tfar) break;

        pos += step;
    }

    sum *= brightness;

    // write output color
    d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

__global__ void
d_render_regular(uint *d_output, uint imageW, uint imageH,
                 float density, float brightness,
                 float transferOffset, float transferScale, hipTextureObject_t volumeTex,
                 hipTextureObject_t transferTex, hipTextureObject_t transferLayerPreintTex,
                 float transferWeight = 0.0f)
{
    d_render<TF_SINGLE_1D>(d_output,imageW,imageH,density,brightness,transferOffset,transferScale,volumeTex,
                             transferTex, transferLayerPreintTex, transferWeight);
}

__global__ void
d_render_preint(uint *d_output, uint imageW, uint imageH,
                float density, float brightness,
                float transferOffset, float transferScale, hipTextureObject_t volumeTex,
                hipTextureObject_t transferTex, hipTextureObject_t transferLayerPreintTex, 
                float transferWeight = 0.0f)
{
    d_render<TF_LAYERED_2D_PREINT>(d_output,imageW,imageH,density,brightness,transferOffset,transferScale,volumeTex,
                                transferTex, transferLayerPreintTex, transferWeight);
}

__global__ void
d_render_preint_off(uint *d_output, uint imageW, uint imageH,
                    float density, float brightness,
                    float transferOffset, float transferScale, hipTextureObject_t volumeTex,
                    hipTextureObject_t transferTex, hipTextureObject_t transferLayerPreintTex,
                    float transferWeight = 0.0f)
{
    d_render<TF_LAYERED_2D>(d_output,imageW,imageH,density,brightness,transferOffset,transferScale,volumeTex,
                            transferTex, transferLayerPreintTex, transferWeight);
}

//////////////////////////////////////////////////////////////////////////

__global__ void
d_integrate_trapezoidal(hipExtent extent, hipTextureObject_t transferTex,
                        hipSurfaceObject_t transferIntegrateSurf)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;

    // for higher speed could use hierarchical approach for sum
    if (x >= extent.width)
    {
        return;
    }

    float stepsize = 1.0/float(extent.width-1);
    float to = float(x) * stepsize;

    float4 outclr = make_float4(0,0,0,0);
    float incr = stepsize;

    float4 lastval = tex1D<float4>(transferTex,0);

    float cur = incr;

    while (cur < to + incr * 0.5)
    {
        float4 val = tex1D<float4>(transferTex,cur);
        float4 trapezoid = (lastval+val)/2.0f;
        lastval = val;

        outclr += trapezoid;
        cur += incr;
    }

    // surface writes need byte offsets for x!
    surf1Dwrite(outclr,transferIntegrateSurf,x * sizeof(float4));
}

__global__ void
d_preintegrate(int layer, float steps, hipExtent extent, hipTextureObject_t transferTex,
                hipTextureObject_t transferIntegrateTex, hipSurfaceObject_t transferLayerPreintSurf)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= extent.width || y >= extent.height)
    {
        return;
    }

    float sx = float(x)/float(extent.width);
    float sy = float(y)/float(extent.height);

    float smax = max(sx,sy);
    float smin = min(sx,sy);

    float4 iv;

    if (x != y)
    {
        // assumes square textures!
        float fracc = smax - smin;
        fracc = 1.0 /(fracc*steps);

        float4 intmax = tex1D<float4>(transferIntegrateTex,smax);
        float4 intmin = tex1D<float4>(transferIntegrateTex,smin);
        iv.x = (intmax.x - intmin.x)*fracc;
        iv.y = (intmax.y - intmin.y)*fracc;
        iv.z = (intmax.z - intmin.z)*fracc;
        //iv.w = (intmax.w - intmin.w)*fracc;
        iv.w   = (1.0 - exp(-(intmax.w - intmin.w) * fracc));
    }
    else
    {
        float4 sample = tex1D<float4>(transferTex,smin);
        iv.x = sample.x;
        iv.y = sample.y;
        iv.z = sample.z;
        //iv.w = sample.w;
        iv.w   = (1.0 - exp(-sample.w));
    }

    iv.x =  __saturatef(iv.x);
    iv.y =  __saturatef(iv.y);
    iv.z =  __saturatef(iv.z);
    iv.w =  __saturatef(iv.w);

    // surface writes need byte offsets for x!
    surf2DLayeredwrite(iv,transferLayerPreintSurf, x * sizeof(float4), y, layer);
}

//////////////////////////////////////////////////////////////////////////


void VolumeRender_setTextureFilterMode(bool bLinearFilter, Volume* vol)
{
    if (vol->volumeTex)
    {
        checkHIPErrors(hipDestroyTextureObject(vol->volumeTex));
    }
    hipResourceDesc            texRes;
    memset(&texRes,0,sizeof(hipResourceDesc));

    texRes.resType            = hipResourceTypeArray;
    texRes.res.array.array    = vol->content;

    hipTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(hipTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode       = bLinearFilter ? hipFilterModeLinear : hipFilterModePoint;

    texDescr.addressMode[0] = hipAddressModeWrap;
    texDescr.addressMode[1] = hipAddressModeWrap;
    texDescr.addressMode[2] = hipAddressModeWrap;

    texDescr.readMode = VolumeTypeInfo<VolumeType>::readMode;

    checkHIPErrors(hipCreateTextureObject(&vol->volumeTex, &texRes, &texDescr, NULL));
}

static unsigned int iDivUp(size_t a, size_t b)
{
    size_t val = (a % b != 0) ? (a / b + 1) : (a / b);
    if (val > UINT_MAX)
    {
        fprintf(stderr, "\nUINT_MAX limit exceeded in iDivUp() exiting.....\n");
        exit(EXIT_FAILURE);    // val exceeds limit
    }

    return static_cast<unsigned int>(val);
}

void VolumeRender_updateTF(int tfIdx, int numColors, float4 *colors)
{

    if (d_transferFunc)
    {
        checkHIPErrors(hipFreeArray(d_transferFunc));
        d_transferFunc = 0;
    }

    hipChannelFormatDesc channelFloat4 = hipCreateChannelDesc<float4>();
    checkHIPErrors(hipMallocArray(&d_transferFunc, &channelFloat4, numColors, 1));
    checkHIPErrors(hipMemcpyToArray(d_transferFunc, 0, 0, colors, sizeof(float4)*numColors, hipMemcpyHostToDevice));

    hipResourceDesc            texRes;
    memset(&texRes,0,sizeof(hipResourceDesc));

    texRes.resType            = hipResourceTypeArray;
    texRes.res.array.array    = d_transferFunc;

    hipTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(hipTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode       = hipFilterModeLinear;
    texDescr.addressMode[0] = hipAddressModeClamp;
    texDescr.readMode = hipReadModeElementType;

    checkHIPErrors(hipCreateTextureObject(&transferTex, &texRes, &texDescr, NULL));

    if (tfIdx < 0 || tfIdx >= VOLUMERENDER_TFS)
    {
        return;
    }

    {
        hipExtent extent = {VOLUMERENDER_TF_PREINTSTEPS, 0,0};
        dim3 blockSize(32,1,1);
        dim3 gridSize(iDivUp(extent.width,blockSize.x),1,1);
        hipLaunchKernelGGL(d_integrate_trapezoidal, dim3(gridSize), dim3(blockSize), 0, 0, extent, transferTex, transferIntegrateSurf);
    }

    {
        hipExtent extent = {VOLUMERENDER_TF_PREINTSIZE, VOLUMERENDER_TF_PREINTSIZE,VOLUMERENDER_TFS};
        dim3 blockSize(16,16,1);
        dim3 gridSize(iDivUp(extent.width,blockSize.x),iDivUp(extent.height,blockSize.y),1);
        hipLaunchKernelGGL(d_preintegrate, dim3(gridSize), dim3(blockSize), 0, 0, tfIdx, float(VOLUMERENDER_TF_PREINTSTEPS), extent, transferTex, transferIntegrateTex, 
                                                transferLayerPreintSurf);
    }

}

void VolumeRender_init()
{
    hipResourceDesc            texRes;
    hipTextureDesc             texDescr;

    hipChannelFormatDesc channelFloat4 = hipCreateChannelDesc<float4>();
    hipExtent extent = {VOLUMERENDER_TF_PREINTSIZE, VOLUMERENDER_TF_PREINTSIZE,VOLUMERENDER_TFS};
    checkHIPErrors(hipMalloc3DArray(&d_transferArray, &channelFloat4, extent, hipArrayLayered | hipArraySurfaceLoadStore));

    memset(&texRes,0,sizeof(hipResourceDesc));

    texRes.resType            = hipResourceTypeArray;
    texRes.res.array.array    = d_transferArray;

    memset(&texDescr,0,sizeof(hipTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode       = hipFilterModeLinear;
    texDescr.addressMode[0] = hipAddressModeClamp;
    texDescr.addressMode[1] = hipAddressModeClamp;
    texDescr.readMode = hipReadModeElementType;

    checkHIPErrors(hipCreateTextureObject(&transferLayerPreintTex, &texRes, &texDescr, NULL));

    hipResourceDesc    surfRes;
    memset(&surfRes, 0, sizeof(hipResourceDesc));
    surfRes.resType = hipResourceTypeArray;
    surfRes.res.array.array = d_transferArray;

    checkHIPErrors(hipCreateSurfaceObject(&transferLayerPreintSurf, &surfRes));

    checkHIPErrors(hipMallocArray(&d_transferIntegrate, &channelFloat4, VOLUMERENDER_TF_PREINTSTEPS,0,hipArraySurfaceLoadStore));

    memset(&texRes,0,sizeof(hipResourceDesc));

    texRes.resType            = hipResourceTypeArray;
    texRes.res.array.array    = d_transferIntegrate;

    memset(&texDescr,0,sizeof(hipTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode       = hipFilterModeLinear;

    texDescr.addressMode[0] = hipAddressModeClamp;
    texDescr.addressMode[1] = hipAddressModeClamp;
    texDescr.addressMode[2] = hipAddressModeClamp;

    texDescr.readMode = hipReadModeElementType;

    checkHIPErrors(hipCreateTextureObject(&transferIntegrateTex, &texRes, &texDescr, NULL));

    memset(&surfRes, 0, sizeof(hipResourceDesc));
    surfRes.resType = hipResourceTypeArray;
    surfRes.res.array.array = d_transferIntegrate;

    checkHIPErrors(hipCreateSurfaceObject(&transferIntegrateSurf, &surfRes));

    // create transfer function texture
    float4 transferFunc0[] =
    {
        {  0.0, 0.0, 0.0, 0.0, },
        {  1.0, 0.0, 0.0, 1.0, },
        {  1.0, 0.5, 0.0, 1.0, },
        {  1.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 1.0, 1.0, },
        {  0.0, 0.0, 1.0, 1.0, },
        {  1.0, 0.0, 1.0, 1.0, },
        {  0.0, 0.0, 0.0, 0.0, },
    };

    float4 transferFunc1[] =
    {
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 1.0, 0.0, 0.125, },
        {  0.0, 0.5, 1.0, 0.125, },
        {  0.0, 1.0, 1.0, 0.125, },
        {  0.0, 1.0, 0.0, 0.125, },
        {  0.25, 0.75, 0.0, 1.0, },
        {  0.75, 0.25, 0.0, 0.125, },
        {  1.0, 0.75, 0.0, 0.125, },
        {  0.0, 0.0, 0.0, 0.0, },
    };

    VolumeRender_updateTF(1,sizeof(transferFunc1)/sizeof(float4),transferFunc1);
    VolumeRender_updateTF(0,sizeof(transferFunc0)/sizeof(float4),transferFunc0);
}

void VolumeRender_deinit()
{
    checkHIPErrors(hipDestroyTextureObject(transferTex));
    checkHIPErrors(hipDestroyTextureObject(transferIntegrateTex));
    checkHIPErrors(hipDestroySurfaceObject(transferIntegrateSurf));
    checkHIPErrors(hipDestroyTextureObject(transferLayerPreintTex));
    checkHIPErrors(hipDestroySurfaceObject(transferLayerPreintSurf)); 
    checkHIPErrors(hipFreeArray(d_transferFunc));
    checkHIPErrors(hipFreeArray(d_transferArray));
    checkHIPErrors(hipFreeArray(d_transferIntegrate));
    d_transferArray = 0;
    d_transferFunc = 0;
    d_transferIntegrate = 0;
}



void VolumeRender_setPreIntegrated(int state)
{
    usePreInt = !!state;
}

void VolumeRender_render(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                         float density, float brightness, float transferOffset, float transferScale, 
                         hipTextureObject_t volumeTex)
{
    if (usePreInt)
    {
        hipLaunchKernelGGL(d_render_preint, dim3(gridSize), dim3(blockSize), 0, 0, d_output, imageW, imageH, density,
                                                 brightness, transferOffset, transferScale, 
                                                 volumeTex, transferTex, transferLayerPreintTex);
    }
    else
    {
        hipLaunchKernelGGL(d_render_preint_off, dim3(gridSize), dim3(blockSize), 0, 0, d_output, imageW, imageH, density,
                                                     brightness, transferOffset, transferScale, 
                                                     volumeTex, transferTex, transferLayerPreintTex);
    }

}

void VolumeRender_copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    checkHIPErrors(hipMemcpyToSymbol(HIP_SYMBOL(c_invViewMatrix), invViewMatrix, sizeofMatrix));
}


#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
