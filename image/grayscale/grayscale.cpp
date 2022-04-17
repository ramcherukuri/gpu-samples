
#include <opencv2/opencv.hpp>
#include "hip/hip_runtime.h"
#include <miopen/miopen.h>

__global__ void grayscale(const float * input, float * output, int height, int width)
{
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((Row < height) && (Col < width)) {
    int grayOffset = Row * width + Col;
    int rgbOffset = grayOffset * 3;
    output[grayOffset]= 0.21*input[rgbOffset] + 
                        0.71*input[rgbOffset+1] + 
                        0.07*input[rgbOffset+2];
  }
}

int main(int argc, char const *argv[]) {

  float* d_input{nullptr};
  float* d_output{nullptr};
  int THREADS = 32;

  cv::Mat image = cv::imread("./test.png", cv::IMREAD_COLOR);
  image.convertTo(image, CV_32FC3);

  // Matrix size;
  int image_size = image.rows * image.cols;
  int input_image_bytes = image.channels() * image_size * sizeof(float);
  int output_image_bytes = image_size * sizeof(float);
    
  // Memory for INPUT
  hipMalloc(&d_input, input_image_bytes);
  hipMemcpy(d_input, image.ptr<float>(0), input_image_bytes, hipMemcpyHostToDevice);

  // Memory for OUTPUT
  hipMalloc(&d_output, output_image_bytes);

  int BLOCKS = image_size / THREADS;
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  grayscale<<<blocks, threads>>>(d_input, d_output, image.rows, image.cols);

  float* h_output = new float[output_image_bytes];
  hipMemcpy(h_output, d_output, output_image_bytes, hipMemcpyDeviceToHost);

  cv::Mat output_image(image.rows, image.cols, CV_32FC1, h_output);
  cv::imwrite("out.png", output_image);

  delete[] h_output;
  hipFree(d_input);
  hipFree(d_output);
}

