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
    This sample has two kernels, one doing the rendering every frame, and another
    one used to generate the mip map levels at startup.

    For rendering we use a "virtual" texturing approach, where one 2d texture
    stores pointers to the actual textures used. This can be achieved by the
    new cudaTextureObject introduced in CUDA 5.0 and requiring sm3+ hardware.

    The mipmap generation kernel uses cudaSurfaceObject and cudaTextureObject
    passed as kernel arguments to compute the higher mip map level based on
    the lower.

*/

#ifndef _BINDLESSTEXTURE_KERNEL_CU_
#define _BINDLESSTEXTURE_KERNEL_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <vector>

#include <helper_hip.h>
#include <helper_math.h>

#include "bindlessTexture.h"

// set this to just see the mipmap chain of first image
//#define SHOW_MIPMAPS


// local references to resources

Image               atlasImage;
std::vector<Image>  contentImages;
float               highestLod = 1.0f;

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

//////////////////////////////////////////////////////////////////////////

__host__ __device__ __inline__ uint2 encodeTextureObject(hipTextureObject_t obj)
{
    return make_uint2((uint)(obj & 0xFFFFFFFF), (uint)(obj >> 32));
}

__host__ __device__ __inline__ hipTextureObject_t decodeTextureObject(uint2 obj)
{
    return (((hipTextureObject_t)obj.x) | ((hipTextureObject_t)obj.y) << 32);
}

__device__ __inline__ float4 to_float4(uchar4 vec)
{
    return make_float4(vec.x, vec.y, vec.z, vec.w);
}

__device__ __inline__ uchar4 to_uchar4(float4 vec)
{
    return make_uchar4((uchar)vec.x, (uchar)vec.y, (uchar)vec.z, (uchar)vec.w);
}

//////////////////////////////////////////////////////////////////////////
// Rendering

// the atlas texture stores the 64 bit cudaTextureObjects
// we use it for "virtual" texturing

__global__ void
d_render(uchar4 *d_output, uint imageW, uint imageH, float lod, hipTextureObject_t atlasTexture)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x / (float) imageW;
    float v = y / (float) imageH;

    if ((x < imageW) && (y < imageH))
    {
        // read from 2D atlas texture and decode texture object
        uint2 texCoded = tex2D<uint2>(atlasTexture, u, v);
        hipTextureObject_t tex = decodeTextureObject(texCoded);

        // read from cuda texture object, use template to specify what data will be
        // returned. tex2DLod allows us to pass the lod (mip map level) directly.
        // There is other functions with CUDA 5, e.g. tex2DGrad,    that allow you
        // to pass derivatives to perform automatic mipmap/anisotropic filtering.
        float4 color = tex2DLod<float4>(tex, u, 1-v, lod);
        // In our sample tex is always valid, but for something like your own
        // sparse texturing you would need to make sure to handle the zero case.

        // write output color
        uint i = y * imageW + x;
        d_output[i] = to_uchar4(color * 255.0);
    }
}

extern "C"
void renderAtlasImage(dim3 gridSize, dim3 blockSize, uchar4 *d_output, uint imageW, uint imageH, float lod)
{
    // psuedo animate lod
    lod = fmodf(lod,highestLod*2);
    lod = highestLod-fabs(lod-highestLod);

#ifdef SHOW_MIPMAPS
    lod = 0.0f;
#endif

    hipLaunchKernelGGL(d_render, dim3(gridSize), dim3(blockSize), 0, 0, d_output, imageW, imageH, lod, atlasImage.textureObject);

    checkHIPErrors(hipGetLastError());
}

//////////////////////////////////////////////////////////////////////////
// MipMap Generation


//  A key benefit of using the new surface objects is that we don't need any global
//  binding points anymore. We can directly pass them as function arguments.

__global__ void
d_mipmap(hipSurfaceObject_t mipOutput, hipTextureObject_t mipInput, uint imageW, uint imageH)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    float px = 1.0/float(imageW);
    float py = 1.0/float(imageH);


    if ((x < imageW) && (y < imageH))
    {
        // take the average of 4 samples

        // we are using the normalized access to make sure non-power-of-two textures
        // behave well when downsized.
        float4 color =
            (tex2D<float4>(mipInput,(x + 0) * px, (y + 0) * py)) +
            (tex2D<float4>(mipInput,(x + 1) * px, (y + 0) * py)) +
            (tex2D<float4>(mipInput,(x + 1) * px, (y + 1) * py)) +
            (tex2D<float4>(mipInput,(x + 0) * px, (y + 1) * py));


        color /= 4.0;
        color *= 255.0;
        color = fminf(color,make_float4(255.0));

        surf2Dwrite(to_uchar4(color),mipOutput,x * sizeof(uchar4),y);
    }
}

void generateMipMaps(hipMipmappedArray_t mipmapArray, hipExtent size)
{
    size_t width    = size.width;
    size_t height   = size.height;

#ifdef SHOW_MIPMAPS
    hipArray_t levelFirst;
    checkHIPErrors(cudaGetMipmappedArrayLevel(&levelFirst, mipmapArray, 0));
#endif

    uint level = 0;

    while (width != 1 || height != 1)
    {
        width     /= 2;
        width      = MAX((size_t)1,width);
        height    /= 2;
        height     = MAX((size_t)1,height);

        hipArray_t levelFrom;
        checkHIPErrors(cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level));
        hipArray_t levelTo;
        checkHIPErrors(cudaGetMipmappedArrayLevel(&levelTo,   mipmapArray, level + 1));

        hipExtent  levelToSize;
        checkHIPErrors(cudaArrayGetInfo(NULL,&levelToSize,NULL,levelTo));
        checkHost(levelToSize.width  == width);
        checkHost(levelToSize.height == height);
        checkHost(levelToSize.depth  == 0);

        // generate texture object for reading
        hipTextureObject_t         texInput;
        hipResourceDesc            texRes;
        memset(&texRes,0,sizeof(hipResourceDesc));

        texRes.resType            = hipResourceTypeArray;
        texRes.res.array.array    = levelFrom;

        hipTextureDesc             texDescr;
        memset(&texDescr,0,sizeof(hipTextureDesc));

        texDescr.normalizedCoords = 1;
        texDescr.filterMode       = hipFilterModeLinear;

        texDescr.addressMode[0] = hipAddressModeClamp;
        texDescr.addressMode[1] = hipAddressModeClamp;
        texDescr.addressMode[2] = hipAddressModeClamp;

        texDescr.readMode = hipReadModeNormalizedFloat;

        checkHIPErrors(hipCreateTextureObject(&texInput, &texRes, &texDescr, NULL));

        // generate surface object for writing

        hipSurfaceObject_t surfOutput;
        hipResourceDesc    surfRes;
        memset(&surfRes,0,sizeof(hipResourceDesc));
        surfRes.resType = hipResourceTypeArray;
        surfRes.res.array.array = levelTo;

        checkHIPErrors(hipCreateSurfaceObject(&surfOutput,&surfRes));

        // run mipmap kernel
        dim3 blockSize(16,16,1);
        dim3 gridSize(((uint)width+blockSize.x-1)/blockSize.x,((uint)height+blockSize.y-1)/blockSize.y,1);

        hipLaunchKernelGGL(d_mipmap, dim3(gridSize), dim3(blockSize), 0, 0, surfOutput, texInput, (uint)width, (uint)height);

        checkHIPErrors(hipDeviceSynchronize());
        checkHIPErrors(hipGetLastError());

        checkHIPErrors(hipDestroySurfaceObject(surfOutput));

        checkHIPErrors(hipDestroyTextureObject(texInput));

#ifdef SHOW_MIPMAPS
        // we blit the current mipmap back into first level
        hipMemcpy3DParms copyParams = {0};
        copyParams.dstArray     = levelFirst;
        copyParams.srcArray     = levelTo;
        copyParams.extent       = make_hipExtent(width,height,1);
        copyParams.kind         = hipMemcpyDeviceToDevice;
        checkHIPErrors(hipMemcpy3D(&copyParams));
#endif

        level++;
    }
}

uint getMipMapLevels(hipExtent size)
{
    size_t sz = MAX(MAX(size.width,size.height),size.depth);

    uint levels = 0;

    while (sz)
    {
        sz /= 2;
        levels++;
    }

    return levels;
}


//////////////////////////////////////////////////////////////////////////
// Initalization

extern "C"
void randomizeAtlas()
{
    uint2 *h_data = (uint2 *) atlasImage.h_data;

    // assign random texture object handles to our atlas image tiles
    for (size_t i = 0; i < atlasImage.size.width * atlasImage.size.height; i++)
    {
#ifdef SHOW_MIPMAPS
        h_data[i] = encodeTextureObject(contentImages[ 0 ].textureObject);
#else
        h_data[i] = encodeTextureObject(contentImages[ rand() % contentImages.size() ].textureObject);
#endif
    }

    // copy data to atlas array
    hipMemcpy3DParms copyParams = {0};
    copyParams.srcPtr       = make_hipPitchedPtr(atlasImage.h_data, atlasImage.size.width * sizeof(uint2), atlasImage.size.width, atlasImage.size.height);
    copyParams.dstArray     = atlasImage.dataArray;
    copyParams.extent       = atlasImage.size;
    copyParams.extent.depth = 1;
    copyParams.kind         = hipMemcpyHostToDevice;
    checkHIPErrors(hipMemcpy3D(&copyParams));

};

extern "C"
void deinitAtlasAndImages()
{
    for (size_t i = 0; i < contentImages.size(); i++)
    {
        Image &image = contentImages[i];

        if (image.h_data)
        {
            free(image.h_data);
        }

        if (image.textureObject)
        {
            checkHIPErrors(hipDestroyTextureObject(image.textureObject));
        }

        if (image.mipmapArray)
        {
            checkHIPErrors(cudaFreeMipmappedArray(image.mipmapArray));
        }
    }

    if (atlasImage.h_data)
    {
        free(atlasImage.h_data);
    }

    if (atlasImage.textureObject)
    {
        checkHIPErrors(hipDestroyTextureObject(atlasImage.textureObject));
    }

    if (atlasImage.dataArray)
    {
        checkHIPErrors(hipFreeArray(atlasImage.dataArray));
    }
}

extern "C"
void initAtlasAndImages(const Image *images, size_t numImages, hipExtent atlasSize)
{
    // create individual textures
    contentImages.resize(numImages);

    for (size_t i = 0; i < numImages; i++)
    {
        Image &image = contentImages[i];
        image.size = images[i].size;
        image.size.depth = 0;
        image.type = hipResourceTypeMipmappedArray;

        // how many mipmaps we need
        uint levels = getMipMapLevels(image.size);
        highestLod  = MAX(highestLod, (float) levels-1);

        hipChannelFormatDesc desc = hipCreateChannelDesc<uchar4>();
        checkHIPErrors(cudaMallocMipmappedArray(&image.mipmapArray, &desc, image.size, levels));

        // upload level 0
        hipArray_t level0;
        checkHIPErrors(cudaGetMipmappedArrayLevel(&level0,image.mipmapArray,0));

        hipMemcpy3DParms copyParams = {0};
        copyParams.srcPtr       = make_hipPitchedPtr(images[i].h_data, image.size.width * sizeof(uchar4), image.size.width, image.size.height);
        copyParams.dstArray     = level0;
        copyParams.extent       = image.size;
        copyParams.extent.depth = 1;
        copyParams.kind         = hipMemcpyHostToDevice;
        checkHIPErrors(hipMemcpy3D(&copyParams));

        // compute rest of mipmaps based on level 0
        generateMipMaps(image.mipmapArray, image.size);

        // generate bindless texture object

        hipResourceDesc            resDescr;
        memset(&resDescr,0,sizeof(hipResourceDesc));

        resDescr.resType            = hipResourceTypeMipmappedArray;
        resDescr.res.mipmap.mipmap  = image.mipmapArray;

        hipTextureDesc             texDescr;
        memset(&texDescr,0,sizeof(hipTextureDesc));

        texDescr.normalizedCoords = 1;
        texDescr.filterMode       = hipFilterModeLinear;
        texDescr.mipmapFilterMode = hipFilterModeLinear;

        texDescr.addressMode[0] = hipAddressModeClamp;
        texDescr.addressMode[1] = hipAddressModeClamp;
        texDescr.addressMode[2] = hipAddressModeClamp;

        texDescr.maxMipmapLevelClamp = float(levels - 1);

        texDescr.readMode = hipReadModeNormalizedFloat;

        checkHIPErrors(hipCreateTextureObject(&image.textureObject, &resDescr, &texDescr, NULL));
    }

    // create atlas array
    hipChannelFormatDesc channelDesc = hipCreateChannelDesc<uint2>();
    checkHIPErrors(hipMallocArray(&atlasImage.dataArray, &channelDesc, atlasSize.width, atlasSize.height));
    atlasImage.h_data              = malloc(atlasSize.width * atlasSize.height * sizeof(uint2));
    atlasImage.type                = hipResourceTypeArray;
    atlasImage.size                = atlasSize;

    hipResourceDesc            texRes;
    memset(&texRes,0,sizeof(hipResourceDesc));

    texRes.resType            = hipResourceTypeArray;
    texRes.res.array.array    = atlasImage.dataArray;

    hipTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(hipTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode       = hipFilterModePoint;
    texDescr.addressMode[0] = hipAddressModeClamp;
    texDescr.addressMode[1] = hipAddressModeClamp;
    texDescr.addressMode[1] = hipAddressModeClamp;
    texDescr.readMode = hipReadModeElementType;

    checkHIPErrors(hipCreateTextureObject(&atlasImage.textureObject, &texRes, &texDescr, NULL));

    randomizeAtlas();
}


#endif // #ifndef _SIMPLETEXTURE3D_KERNEL_CU_
