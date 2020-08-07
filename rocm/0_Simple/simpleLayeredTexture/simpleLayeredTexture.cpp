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
* This sample demonstrates how to use texture fetches from layered 2D textures in CUDA C
*
* This sample first generates a 3D input data array for the layered texture
* and the expected output. Then it starts CUDA C kernels, one for each layer,
* which fetch their layer's texture data (using normalized texture coordinates)
* transform it to the expected output, and write it to a 3D output data array.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, kernels
#include <hip/hip_runtime.h>

// includes, project
#include <helper_hip.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

static const char *sSDKname = "simpleLayeredTexture";


////////////////////////////////////////////////////////////////////////////////
//! Transform a layer of a layered 2D texture using texture lookups
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
transformKernel(float *g_odata, int width, int height, int layer, hipTextureObject_t tex)
{
    // calculate this thread's data point
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // 0.5f offset and division are necessary to access the original data points
    // in the texture (such that bilinear interpolation will not be activated).
    // For details, see also CUDA Programming Guide, Appendix D
    float u = (x+0.5f) / (float) width;
    float v = (y+0.5f) / (float) height;

    // read from texture, do expected transformation and write to global memory
    g_odata[layer*width*height + y*width + x] = -tex2DLayered<float>(tex, u, v, layer) + layer;
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    printf("[%s] - Starting...\n", sSDKname);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findHIPDevice(argc, (const char **)argv);

    bool bResult = true;

    // get number of SMs on this GPU
    hipDeviceProp_t deviceProps;

    checkHIPErrors(hipGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s] has %d Multi-Processors ", deviceProps.name, deviceProps.multiProcessorCount);
    printf("SM %d.%d\n", deviceProps.major, deviceProps.minor);

    // generate input data for layered texture
    unsigned int width=512, height=512, num_layers = 5;
    unsigned int size = width * height * num_layers * sizeof(float);
    float *h_data = (float *) malloc(size);

    for (unsigned int layer = 0; layer < num_layers; layer++)
        for (int i = 0; i < (int)(width * height); i++)
        {
            h_data[layer*width*height + i] = (float)i;
        }

    // this is the expected transformation of the input data (the expected output)
    float *h_data_ref = (float *) malloc(size);

    for (unsigned int layer = 0; layer < num_layers; layer++)
        for (int i = 0; i < (int)(width * height); i++)
        {
            h_data_ref[layer*width*height + i] = -h_data[layer*width*height + i] + layer;
        }

    // allocate device memory for result
    float *d_data = NULL;
    checkHIPErrors(hipMalloc((void **) &d_data, size));

    // allocate array and copy image data
    hipChannelFormatDesc channelDesc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
    hipArray *cu_3darray;
    checkHIPErrors(hipMalloc3DArray(&cu_3darray, &channelDesc, make_hipExtent(width, height, num_layers), hipArrayLayered));
    hipMemcpy3DParms myparms = {0};
    myparms.srcPos = make_hipPos(0,0,0);
    myparms.dstPos = make_hipPos(0,0,0);
    myparms.srcPtr = make_hipPitchedPtr(h_data, width * sizeof(float), width, height);
    myparms.dstArray = cu_3darray;
    myparms.extent = make_hipExtent(width, height, num_layers);
    myparms.kind = hipMemcpyHostToDevice;
    checkHIPErrors(hipMemcpy3D(&myparms));

    hipTextureObject_t         tex;
    hipResourceDesc            texRes;
    memset(&texRes,0,sizeof(hipResourceDesc));

    texRes.resType            = hipResourceTypeArray;
    texRes.res.array.array    = cu_3darray;

    hipTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(hipTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode       = hipFilterModeLinear;
    texDescr.addressMode[0] = hipAddressModeWrap;
    texDescr.addressMode[1] = hipAddressModeWrap;
    texDescr.readMode = hipReadModeElementType;

    checkHIPErrors(hipCreateTextureObject(&tex, &texRes, &texDescr, NULL));

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    printf("Covering 2D data array of %d x %d: Grid size is %d x %d, each block has 8 x 8 threads\n",
           width, height, dimGrid.x, dimGrid.y);

    hipLaunchKernelGGL(transformKernel, dim3(dimGrid), dim3(dimBlock ), 0, 0, d_data, width, height, 0, tex);  // warmup (for better timing)

    // check if kernel execution generated an error
    getLastHIPError("warmup Kernel execution failed");

    checkHIPErrors(hipDeviceSynchronize());

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // execute the kernel
    for (unsigned int layer = 0; layer < num_layers; layer++)
        hipLaunchKernelGGL(transformKernel, dim3(dimGrid), dim3(dimBlock), 0 , 0, d_data, width, height, layer, tex);

    // check if kernel execution generated an error
    getLastHIPError("Kernel execution failed");

    checkHIPErrors(hipDeviceSynchronize());
    sdkStopTimer(&timer);
    printf("Processing time: %.3f msec\n", sdkGetTimerValue(&timer));
    printf("%.2f Mtexlookups/sec\n", (width *height *num_layers / (sdkGetTimerValue(&timer) / 1000.0f) / 1e6));
    sdkDeleteTimer(&timer);

    // allocate mem for the result on host side
    float *h_odata = (float *) malloc(size);
    // copy result from device to host
    checkHIPErrors(hipMemcpy(h_odata, d_data, size, hipMemcpyDeviceToHost));

    // write regression file if necessary
    if (checkCmdLineFlag(argc, (const char **)argv, "regression"))
    {
        // write file for regression test
        sdkWriteFile<float>("./data/regression.dat", h_odata, width*height, 0.0f, false);
    }
    else
    {
        printf("Comparing kernel output to expected data\n");

#define MIN_EPSILON_ERROR 5e-3f
        bResult = compareData(h_odata, h_data_ref, width*height*num_layers, MIN_EPSILON_ERROR, 0.0f);
    }

    // cleanup memory
    free(h_data);
    free(h_data_ref);
    free(h_odata);

    checkHIPErrors(hipDestroyTextureObject(tex));
    checkHIPErrors(hipFree(d_data));
    checkHIPErrors(hipFreeArray(cu_3darray));

    exit(bResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
