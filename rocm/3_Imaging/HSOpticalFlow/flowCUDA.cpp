/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

#include "common.h"

// include kernels
#include "downscaleKernel.cuh"
#include "upscaleKernel.cuh"
#include "warpingKernel.cuh"
#include "derivativesKernel.cuh"
#include "solverKernel.cuh"
#include "addKernel.cuh"

///////////////////////////////////////////////////////////////////////////////
/// \brief method logic
///
/// handles memory allocations, control flow
/// \param[in]  I0           source image
/// \param[in]  I1           tracked image
/// \param[in]  width        images width
/// \param[in]  height       images height
/// \param[in]  stride       images stride
/// \param[in]  alpha        degree of displacement field smoothness
/// \param[in]  nLevels      number of levels in a pyramid
/// \param[in]  nWarpIters   number of warping iterations per pyramid level
/// \param[in]  nSolverIters number of solver iterations (Jacobi iterations)
/// \param[out] u            horizontal displacement
/// \param[out] v            vertical displacement
///////////////////////////////////////////////////////////////////////////////
void ComputeFlowCUDA(const float *I0,
                     const float *I1,
                     int width, int height, int stride,
                     float alpha,
                     int nLevels,
                     int nWarpIters,
                     int nSolverIters,
                     float *u,
                     float *v)
{
    printf("Computing optical flow on GPU...\n");

    // pI0 and pI1 will hold device pointers
    const float **pI0 = new const float *[nLevels];
    const float **pI1 = new const float *[nLevels];

    int *pW = new int [nLevels];
    int *pH = new int [nLevels];
    int *pS = new int [nLevels];

    // device memory pointers
    float *d_tmp;
    float *d_du0;
    float *d_dv0;
    float *d_du1;
    float *d_dv1;

    float *d_Ix;
    float *d_Iy;
    float *d_Iz;

    float *d_u;
    float *d_v;
    float *d_nu;
    float *d_nv;

    const int dataSize = stride * height * sizeof(float);

    checkHIPErrors(hipMalloc(&d_tmp, dataSize));
    checkHIPErrors(hipMalloc(&d_du0, dataSize));
    checkHIPErrors(hipMalloc(&d_dv0, dataSize));
    checkHIPErrors(hipMalloc(&d_du1, dataSize));
    checkHIPErrors(hipMalloc(&d_dv1, dataSize));

    checkHIPErrors(hipMalloc(&d_Ix, dataSize));
    checkHIPErrors(hipMalloc(&d_Iy, dataSize));
    checkHIPErrors(hipMalloc(&d_Iz, dataSize));

    checkHIPErrors(hipMalloc(&d_u , dataSize));
    checkHIPErrors(hipMalloc(&d_v , dataSize));
    checkHIPErrors(hipMalloc(&d_nu, dataSize));
    checkHIPErrors(hipMalloc(&d_nv, dataSize));

    // prepare pyramid

    int currentLevel = nLevels - 1;
    // allocate GPU memory for input images
    checkHIPErrors(hipMalloc(pI0 + currentLevel, dataSize));
    checkHIPErrors(hipMalloc(pI1 + currentLevel, dataSize));

    checkHIPErrors(hipMemcpy((void *)pI0[currentLevel], I0, dataSize, hipMemcpyHostToDevice));
    checkHIPErrors(hipMemcpy((void *)pI1[currentLevel], I1, dataSize, hipMemcpyHostToDevice));

    pW[currentLevel] = width;
    pH[currentLevel] = height;
    pS[currentLevel] = stride;

    for (; currentLevel > 0; --currentLevel)
    {
        int nw = pW[currentLevel] / 2;
        int nh = pH[currentLevel] / 2;
        int ns = iAlignUp(nw);

        checkHIPErrors(hipMalloc(pI0 + currentLevel - 1, ns * nh * sizeof(float)));
        checkHIPErrors(hipMalloc(pI1 + currentLevel - 1, ns * nh * sizeof(float)));

        Downscale(pI0[currentLevel], pW[currentLevel], pH[currentLevel],
                  pS[currentLevel], nw, nh, ns, (float *)pI0[currentLevel - 1]);

        Downscale(pI1[currentLevel], pW[currentLevel], pH[currentLevel],
                  pS[currentLevel], nw, nh, ns, (float *)pI1[currentLevel - 1]);

        pW[currentLevel - 1] = nw;
        pH[currentLevel - 1] = nh;
        pS[currentLevel - 1] = ns;
    }

    checkHIPErrors(hipMemset(d_u, 0, stride * height * sizeof(float)));
    checkHIPErrors(hipMemset(d_v, 0, stride * height * sizeof(float)));

    // compute flow
    for (; currentLevel < nLevels; ++currentLevel)
    {

        for (int warpIter = 0; warpIter < nWarpIters; ++warpIter)
        {
            checkHIPErrors(hipMemset(d_du0, 0, dataSize));
            checkHIPErrors(hipMemset(d_dv0, 0, dataSize));

            checkHIPErrors(hipMemset(d_du1, 0, dataSize));
            checkHIPErrors(hipMemset(d_dv1, 0, dataSize));

            // on current level we compute optical flow
            // between frame 0 and warped frame 1
            WarpImage(pI1[currentLevel], pW[currentLevel], pH[currentLevel],
                      pS[currentLevel], d_u, d_v, d_tmp);

            ComputeDerivatives(pI0[currentLevel], d_tmp, pW[currentLevel],
                               pH[currentLevel], pS[currentLevel], d_Ix, d_Iy, d_Iz);

            for (int iter = 0; iter < nSolverIters; ++iter)
            {
                SolveForUpdate(d_du0, d_dv0, d_Ix, d_Iy, d_Iz,
                               pW[currentLevel], pH[currentLevel], pS[currentLevel], alpha, d_du1, d_dv1);

                Swap(d_du0, d_du1);
                Swap(d_dv0, d_dv1);
            }

            // update u, v
            Add(d_u, d_du0, pH[currentLevel] * pS[currentLevel], d_u);
            Add(d_v, d_dv0, pH[currentLevel] * pS[currentLevel], d_v);
        }

        if (currentLevel != nLevels - 1)
        {
            // prolongate solution
            float scaleX = (float)pW[currentLevel + 1]/(float)pW[currentLevel];

            Upscale(d_u, pW[currentLevel], pH[currentLevel], pS[currentLevel],
                    pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1], scaleX, d_nu);

            float scaleY = (float)pH[currentLevel + 1]/(float)pH[currentLevel];

            Upscale(d_v, pW[currentLevel], pH[currentLevel], pS[currentLevel],
                    pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1], scaleY, d_nv);

            Swap(d_u, d_nu);
            Swap(d_v, d_nv);
        }
    }

    checkHIPErrors(hipMemcpy(u, d_u, dataSize, hipMemcpyDeviceToHost));
    checkHIPErrors(hipMemcpy(v, d_v, dataSize, hipMemcpyDeviceToHost));

    // cleanup
    for (int i = 0; i < nLevels; ++i)
    {
        checkHIPErrors(hipFree((void *)pI0[i]));
        checkHIPErrors(hipFree((void *)pI1[i]));
    }

    delete [] pI0;
    delete [] pI1;
    delete [] pW;
    delete [] pH;
    delete [] pS;

    checkHIPErrors(hipFree(d_tmp));
    checkHIPErrors(hipFree(d_du0));
    checkHIPErrors(hipFree(d_dv0));
    checkHIPErrors(hipFree(d_du1));
    checkHIPErrors(hipFree(d_dv1));
    checkHIPErrors(hipFree(d_Ix));
    checkHIPErrors(hipFree(d_Iy));
    checkHIPErrors(hipFree(d_Iz));
    checkHIPErrors(hipFree(d_nu));
    checkHIPErrors(hipFree(d_nv));
    checkHIPErrors(hipFree(d_u));
    checkHIPErrors(hipFree(d_v));
}
