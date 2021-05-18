
/*
 *
 * http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
 * https://github.com/ROCmSoftwarePlatform/hipDNN/blob/e7579a4ed8d38acf054c2a4c2796ce895d03cfcf/library/src/hcc_detail/hipdnn_miopen.cpp
 * https://intel.github.io/mkl-dnn/understanding_memory_formats.html
 */


#include <opencv2/opencv.hpp>
#include "hip/hip_runtime.h"
#include <miopen/miopen.h>
#include <string>
#include <iostream>

#define checkMIOPEN(expression)                               \
  {                                                          \
    miopenStatus_t status = (expression);                     \
    if (status != miopenStatusSuccess ) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << miopenGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


// 
cv::Mat convert2nchw(cv::Mat image) {

    cv::Mat image2 = image.clone(); 

    float* ptr = reinterpret_cast<float*>(image.ptr());
    float* output = reinterpret_cast<float*>(image2.ptr()); 

    int channel = image.channels();
    int height = image.rows;
    int width = image.cols;
    std::cout << "channel:" << channel << std::endl;
    std::cout << "height:" << height << std::endl;
    std::cout << "width:" << width << std::endl;
    for (int c = 0; c < channel; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                output[ c*height*width + h*width + w ] = ptr[ h*width*channel + w*channel + c ];
            }
        }
    }
    std::cout << "channel2:" << image2.channels() << std::endl;
    std::cout << "height2:" << image2.rows << std::endl;
    std::cout << "width2:" << image2.cols << std::endl;
  
 
    return image2;
}


cv::Mat convert2nhwc(cv::Mat image) {

    cv::Mat image2 = image.clone();

    float* ptr = reinterpret_cast<float*>(image.ptr());
    float* output = reinterpret_cast<float*>(image2.ptr());

    int channel = image.channels();
    int height = image.rows;
    int width = image.cols;
    std::cout << "channel:" << channel << std::endl;
    std::cout << "height:" << height << std::endl;
    std::cout << "width:" << width << std::endl;
    for (int c = 0; c < channel; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                output[ h*width*channel + w*channel + c ] = ptr[ c*height*width + h*width + w ];
            }
        }
    }
    std::cout << "channel2:" << image2.channels() << std::endl;
    std::cout << "height2:" << image2.rows << std::endl;
    std::cout << "width2:" << image2.cols << std::endl;

    return image2;
}

/**************************************************
 *
 * Here is a function you can use to save the image:
 *
 *************************************************/
void save_image(const char* output_filename, float* buffer, 
                int height, int width) {

  cv::Mat output_image(height, width, CV_32FC3, buffer);

  // Make negative values zero.
  cv::threshold(output_image,  output_image,
                /*threshold=*/0, /*maxval=*/0,
                cv::THRESH_TOZERO);

  cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
  output_image = convert2nhwc(output_image);
  output_image.convertTo(output_image, CV_8UC3);
  cv::imwrite(output_filename, output_image);
}

/**************************************************
 * 
 * Here is a helper function to load the image:
 *
 *************************************************/
cv::Mat load_image(const char* image_path) {
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  image.convertTo(image, CV_32FC3);
  cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);

  // Conver from NHWC format to NCHW
  image = convert2nchw(image);
  return image;
}


/************************************************
 *
 * main
 */
int main(int argc, char const *argv[]) {

  if(argc!=2) {
	  std::cout << "usage: ./" << *argv[0] <<"[PATH]/[FILE-NAME].png " << std::endl;
	  std::cout << "   Example: ./" << *argv[0] <<"./mario.png " << std::endl;
	return 1;
  }

  std::string fname(argv[1]);
  cv::Mat image = load_image(fname.c_str());

/* 
 * We need to describe the three data structures that participate in the convolution operation: 
 * 	the input tensor, 
 * 	the output tensor and 
 * 	the kernel tensor. 
 *
 * Particularly important when comes to the memory layout, which can be either 
 * 	NHWC (N, Height, width, channel) or 
 * 	NCHW. 
 *
 * In either of these two formats, N specifies the batch dimension. 
 *
 * MIOPEN generally allows its operations to be performed on batches of data (often images), 
 * so the first dimension would be for individual images. H and W stand for the height 
 * and width dimension. Lastly, C is the channels axis, e.g. for red, green and 
 * blue (RGB) channels of a color image. Some deep learning frameworks, like 
 * TensorFlow, prefer to store tensors in NHWC format (where channels change 
 * most frequently), while others prefer putting channels first. 
 *
 * Let's take a look how to describe the input tensor, which will later on store the image we
 * want to convolve:
 */

  miopenHandle_t miopenHandle;
  miopenCreate(&miopenHandle);

 /* 
 * The first thing we need to do is declare and initialize a miopenTensorDescriptor_t. 
 */
miopenTensorDescriptor_t  input_descriptor;
checkMIOPEN(miopenCreateTensorDescriptor(&input_descriptor));

miopenTensorDescriptor_t  output_descriptor;
checkMIOPEN(miopenCreateTensorDescriptor(&output_descriptor));

miopenTensorDescriptor_t  kernel_descriptor;
checkMIOPEN(miopenCreateTensorDescriptor(&kernel_descriptor));


/* Then, we use miopenSet4dTensorDescriptor to actually specify the properties of
 * the tensor. The remainder of the options tell miopen that we'll be convolving
 * a single image with three (color) channels, whose pixels are represented as
 * floating point values (between 0 and 1). We also configure the height and width
 * of the tensor. We do the same for the output image:
 *
 * miopenStatus_t miopenSet4dTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
 *                                            miopenDataType_t dataType,
 *                                            int n, int c, int h, int w)
 */

checkMIOPEN(miopenSet4dTensorDescriptor(input_descriptor,
                                      /*dataType=*/miopenFloat,
                                      /*batch_size=*/1,
                                      /*channels=*/3,
                                      /*image_height=*/image.rows,
                                      /*image_width=*/image.cols));

checkMIOPEN(miopenSet4dTensorDescriptor(output_descriptor,
                                      /*dataType=*/miopenFloat,
                                      /*batch_size=*/1,
                                      /*channels=*/3,
                                      /*image_height=*/image.rows,
                                      /*image_width=*/image.cols));

/*
 * Leaving us with the kernel tensor. CNN has specialized construction and initialization 
 * routines for kernels (filters): The parameters are essentially the same, since kernels 
 * are small images themselves. The only difference is that the batch_size is now the number 
 * of output channels (out_channels) or feature maps. Note, however, that I've switched the 
 * format argument to NCHW. This will make it easier to define the kernel weights later on.
 */
checkMIOPEN(miopenSet4dTensorDescriptor(kernel_descriptor,
                                      /*dataType=*/miopenFloat,
                                      /*out_channels=*/3,
                                      /*in_channels=*/3,
                                      /*kernel_height=*/3,
                                      /*kernel_width=*/3));

/*
 * With the parameter description out of the way, we now need to tell MIOPEN what kind of 
 * (convolution) operation we want to perform. For this, we again declare and configure a 
 * descriptor, which, you may notice, is an overarching pattern in code:
 *
 */

miopenConvolutionDescriptor_t convolution_descriptor;
checkMIOPEN(miopenCreateConvolutionDescriptor(&convolution_descriptor));
checkMIOPEN(miopenInitConvolutionDescriptor(convolution_descriptor, 
                                           /*mode=*/miopenConvolution,
                                           /*pad_height=*/1,
                                           /*pad_width=*/1,
                                           /*vertical_stride=*/1,
                                           /*horizontal_stride=*/1,
                                           /*dilation_height=*/1,
                                           /*dilation_width=*/1));
/*
 * If you're unfamiliar with the common hyperparameters (read: knobs) of convolutions, you 
 * can find out more about them here: http://cs231n.github.io/convolutional-networks/ . 
 * If you're a pro at convolutions, you will understand that the first two parameters to 
 * miopenInitConvolutionDescriptor after the descriptor control the zero-padding around the 
 * image, the subsequent two control the kernel stride and the next two the dilation. The 
 * mode argument can be either 
 * miopenConvolution = 0  Cross-Correlation convolution
 * miopenTranspose = 1  Transpose convolutions deconvolution
 * miopenGroupConv = 2  Deprecated Group convolution legacy, ToBe Removed
 * miopenDepthwise = 3  Deprecated Depthwise convolution legacy, ToBe Removed
 *
 * These are basically the two ways we can compute the weighted sum that makes up a single 
 * convolution pass for our purposes (and convolutions in CNNs as we know them) we want miopenConvolution. 
 *
 * Ok, are we finally done? Why is this so much code compared to tf.nn.conv2d? 
 * Well, the answers are (1) no and (2) because we're many layers beneath the 
 * level of abstraction we usually enjoy working with. We need two more things: 
 * a more detailed description of the convolution algorithm we want to use and 
 * the physical memory to operate on. Let’s begin with the first:
 *
 *
 *
 * miopenStatus_t miopenFindConvolutionForwardAlgorithm(miopenHandle_t handle, 
 *                                                      const miopenTensorDescriptor_t xDesc, const void *x, 
 *                                                      const miopenTensorDescriptor_t wDesc, const void *w, 
 *                                                      const miopenConvolutionDescriptor_t convDesc, 
 *                                                      const miopenTensorDescriptor_t yDesc, void *y, 
 *                                                      const int requestAlgoCount, 
 *                                                      int *returnedAlgoCount, 
 *                                                      miopenConvAlgoPerf_t *perfResults, 
 *                                                      void *workSpace, 
 *                                                      size_t workSpaceSize, bool exhaustiveSearch)
 *
 * Parameters:
 *     handle: MIOpen handle (input)
 *     xDesc: Tensor descriptor for data input tensor x (input)
 *     x: Data tensor x (input)
 *     wDesc: Tensor descriptor for weight tensor w (input)
 *     w: Weights tensor w (input)
 *     convDesc: Convolution layer descriptor (input)
 *     yDesc: Tensor descriptor for output data tensor y (input)
 *     y: Data tensor y (output)
 *     requestAlgoCount: Number of algorithms to return kernel times (input)
 *     returnedAlgoCount: Pointer to number of algorithms returned (output)
 *     perfResults: Pointer to union of best algorithm for forward and backwards (input)
 *     workSpace: Pointer to workspace required for the search (output)
 *     workSpaceSize: Size in bytes of the memory needed for find (output)
 *     exhaustiveSearch: A boolean to toggle a full search of all algorithms and configurations (input)                                                      
 */

/*
 * At this point, we need to allocate the required resources for the convolution.
 * The number of buffers and memory requirements for each buffer will differ depending
 * on which algorithm we use for the convolution. In our case, we need four buffers for
 * the workspace, the input and output image as well as the kernel. Let's allocate the
 * first three on the device directly:
 */

    // Memory for workspace 
    size_t workspace_bytes = 0;

/*
 * miopenStatus_t miopenConvolutionForwardGetWorkSpaceSize(
 *                                            miopenHandle_t handle, 
 *                                            const miopenTensorDescriptor_t wDesc, 
 *                                            const miopenTensorDescriptor_t xDesc, 
 *                                            const miopenConvolutionDescriptor_t convDesc, 
 *                                            const miopenTensorDescriptor_t yDesc, 
 *                                            size_t *workSpaceSize)
 */ 


    checkMIOPEN(miopenConvolutionForwardGetWorkSpaceSize(
                                                   miopenHandle,
                                                   kernel_descriptor,
                                                   input_descriptor,
                                                   convolution_descriptor,
                                                   output_descriptor,
                                                   &workspace_bytes));

    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
          << std::endl;


    void* d_workspace{nullptr};
    hipMalloc(&d_workspace, workspace_bytes);

/*
 *
 * miopenStatus_t miopenGetConvolutionForwardOutputDim(miopenConvolutionDescriptor_t convDesc, 
 *                                                     const miopenTensorDescriptor_t inputTensorDesc, 
 *                                                     const miopenTensorDescriptor_t filterDesc, 
 *                                                     int *n, int *c, int *h, int *w)
 *
 */

    int batch_size=0, channels=0, height=0, width=0;

    checkMIOPEN(miopenGetConvolutionForwardOutputDim(convolution_descriptor,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     &batch_size, &channels, &height, &width)); 

    int image_bytes = batch_size * channels * height * width * sizeof(float);
    
    // Memory for INPUT
    float* d_input{nullptr};
    hipMalloc(&d_input, image_bytes);
    hipMemcpy(d_input, image.ptr<float>(0), image_bytes, hipMemcpyHostToDevice);

    // Memory for OUTPUT
    float* d_output{nullptr};
    hipMalloc(&d_output, image_bytes);
 //   hipMemset(d_output, 0, image_bytes);

/*
 * Note that I get the batch_size, channels, height and width variables from the 
 * miopenGetConvolutionForwardOutputDim function, which tells you the dimension of 
 * the image after the convolution.
 */

/*
 * The kernel we'll want to first allocate and populate on the host and then copy 
 * to the GPU device:
 *
 */

  // Memory for Kernel
// Edge detection kernel
#define EDGE
#ifdef EDGE
    const float kernel_template[3][3] = {
       {1,  1, 1},
       {1, -8, 1},
       {1,  1, 1}
    };
#endif
/*  X - Axis */
#ifdef XAXIS
    const float kernel_template[3][3] = {
       {-1,  0, 1},
       {-2,  0, 2},
       {-1,  0, 1}
    };
#endif
#ifdef YAXIS
/*  Y - Axis */
    const float kernel_template[3][3] = {
       {-1, -2, -1},
       { 0,  0,  0},
       { 1,  2,  1}
    };
#endif

    float h_kernel[3][3][3][3];
    for (int kernel = 0; kernel < 3; ++kernel) {
      for (int channel = 0; channel < 3; ++channel) {
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) {
            h_kernel[kernel][channel][row][column] = kernel_template[row][column];
          }
        }
      }
    }

    float* d_kernel{nullptr};
    hipMalloc(&d_kernel, sizeof(h_kernel));
    hipMemcpy(d_kernel, h_kernel, sizeof(h_kernel), hipMemcpyHostToDevice);

/*
 * Note how I first declare a template for the 3 by 3 kernel use and then copy 
 * it into the three input and three output dimensions of the actual kernel buffer. 
 * That is, we'll have the same pattern three times, once for each channel of the input 
 * image (red, green and blue), and that whole kernel three times, once for each output 
 * feature map we want to produce.
 *
 * Can you guess what that kernel does? Read on to find out!
 *
 */

    const int requestedAlgoCount = 1;
    int returnedAlgoCount;


    miopenConvAlgoPerf_t *perfResults =
        new miopenConvAlgoPerf_t[requestedAlgoCount];


    miopenFindConvolutionForwardAlgorithm(
            //miopenHandle_t handle(INPUT),
            miopenHandle,
            //const miopenTensorDescriptor_t xDesc(INPUT), const void *x(INPUT),
            input_descriptor, d_input,
            //const miopenTensorDescriptor_t wDesc(INPUT), const void *w(INPUT),
            kernel_descriptor, d_kernel,
            //const miopenConvolutionDescriptor_t convDesc(INPUT),
            convolution_descriptor,
            //const miopenTensorDescriptor_t yDesc(INPUT), void *y(OUTPUT),
            output_descriptor, d_output,
            //const int requestAlgoCount(INPUT), int *returnedAlgoCount(OUTPUT),
            requestedAlgoCount, &returnedAlgoCount,
            //miopenConvAlgoPerf_t *perfResults(INPUT),
            perfResults,
            //void *workSpace(OUTPUT), size_t workSpaceSize(OUTPUT), 
            d_workspace, workspace_bytes,
            //bool exhaustiveSearch
            false  
           );

/*
 * enum miopenConvFwdAlgorithm_t
 * Convolutional algorithm mode for forward propagation. MIOpen use cross-correlation for its convolution implementation.
 *
 * Values:
 * miopenConvolutionFwdAlgoGEMM = 0          GEMM variant
 * miopenConvolutionFwdAlgoDirect = 1        Direct convolutions
 * miopenConvolutionFwdAlgoFFT = 2           Fast Fourier Transform indirect convolutions
 * miopenConvolutionFwdAlgoWinograd = 3      Winograd indirect convolutions
 * miopenConvolutionFwdAlgoImplicitGEMM = 5  Implicit GEMM convolutions, fp32 only and disabled by default
 *
 */

/*
 * struct miopenConvAlgoPerf_t: Perf struct for forward, backward filter, or backward data algorithms.
 *
 * Contains the union to hold the selected convolution algorithm for forward, or backwards layers, 
 * and also contains the time it took to run the algorithm and the workspace required to run the algorithm. 
 * The workspace in this structure can be used when executing the convolution layer.
 * https://github.com/ROCmSoftwarePlatform/MIOpen/blob/027cd281323e9cd35c6af0f500dd8d5b8dc0fa88/include/miopen/miopen.h#L861
 */
    miopenConvFwdAlgorithm_t convolution_algorithm=perfResults[0].fwd_algo;


 /*
 * At last, we can perform the actual convolution operation:
 */

const float alpha = 1, beta = 0;


/*
 * miopenStatus_t miopenConvolutionForward(miopenHandle_t handle, 
 *                                         const void *alpha, 
 *                                         const miopenTensorDescriptor_t xDesc, const void *x, 
 *                                         const miopenTensorDescriptor_t wDesc, const void *w, 
 *                                         const miopenConvolutionDescriptor_t convDesc, 
 *                                         miopenConvFwdAlgorithm_t algo, 
 *                                         const void *beta, 
 *                                         const miopenTensorDescriptor_t yDesc, void *y, 
 *                                         void *workSpace, size_t workSpaceSize)
 *
 */

checkMIOPEN(miopenConvolutionForward(miopenHandle,
                                   &alpha,
                                   input_descriptor, d_input,
                                   kernel_descriptor, d_kernel,
                                   convolution_descriptor,
                                   convolution_algorithm,
                                   &beta,
                                   output_descriptor, d_output,
                                   d_workspace,
                                   workspace_bytes
                                   ));

/*
 * Here, alpha and beta are parameters that can be used to mix the input and output 
 * buffers (they're not really useful for us). The rest of the parameters are basically 
 * everything we declared and configured up to this point. Note that the d_workspace is 
 * allowed to be nullptr if we pick a convolution algorithm that does not require additional 
 * memory.
 *
 * The only thing left to do at this point is copy the resulting image back to the host and 
 * do something with it. We also need to release any resources we allocated, of course:
 *
 */

float* h_output = new float[image_bytes];
hipMemcpy(h_output, d_output, image_bytes, hipMemcpyDeviceToHost);

// Do something with h_output ...
save_image("out.png", h_output, height, width);


delete[] h_output;
hipFree(d_kernel);
hipFree(d_input);
hipFree(d_output);
hipFree(d_workspace);

miopenDestroyTensorDescriptor(input_descriptor);
miopenDestroyTensorDescriptor(output_descriptor);
miopenDestroyTensorDescriptor(kernel_descriptor);
miopenDestroyConvolutionDescriptor(convolution_descriptor);

miopenDestroy(miopenHandle);

}

