
#include <opencv2/opencv.hpp>
#include "hip/hip_runtime.h"
#include <miopen/miopen.h>

#define BLUR_SIZE 16

__global__ void blur(const float * input, float * output, int height, int width)
{
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((Row < height) && (Col < width)) {
     int rgbOffset = (Row * width + Col) * 3;

     int rpixVal = 0;
     int gpixVal = 0;
     int bpixVal = 0;
     int pixels = 0;

     // Get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
     for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
         for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
             int curRow = Row + blurRow;
             int curCol = Col + blurCol;
	     int crgbOffset = (curRow * width + curCol) * 3;
             // Verify we have a valid image pixel
             if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                 rpixVal += input[crgbOffset];
                 gpixVal += input[crgbOffset+1];
                 bpixVal += input[crgbOffset+1];
		 // Keep track of number of pixels in the accumulated total
                 pixels++; 
             }
          }
      }
      // Write our new pixel value out
      output[rgbOffset] = (unsigned char)(rpixVal / pixels);
      output[rgbOffset+1] = (unsigned char)(gpixVal / pixels);
      output[rgbOffset+2] = (unsigned char)(bpixVal / pixels);
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
  int image_bytes = image.channels() * image_size * sizeof(float);
    
  // Memory for INPUT
  hipMalloc(&d_input, image_bytes);
  hipMemcpy(d_input, image.ptr<float>(0), image_bytes, hipMemcpyHostToDevice);

  // Memory for OUTPUT
  hipMalloc(&d_output, image_bytes);

  int BLOCKS = image_size / THREADS;
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  blur<<<blocks, threads>>>(d_input, d_output, image.rows, image.cols);

  float* h_output = new float[image_bytes];
  hipMemcpy(h_output, d_output, image_bytes, hipMemcpyDeviceToHost);

  cv::Mat output_image(image.rows, image.cols, CV_32FC3, h_output);
  cv::imwrite("out.png", output_image);

  delete[] h_output;
  hipFree(d_input);
  hipFree(d_output);
}

