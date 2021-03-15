#ifndef MAT_OPERATIONS_H
#define MAT_OPERATIONS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include <iostream>
#include <stdio.h>
#include <sys/time.h>

using namespace std;
using namespace cv;
using namespace cuda;

__host__ void add_gpu(cv::cuda::GpuMat &A, cv::cuda::GpuMat &B, cv::cuda::GpuMat &dst);
__host__ void add_gpu(cv::cuda::GpuMat &A, float b, cv::cuda::GpuMat &dst);

__host__ void subtract_gpu(cv::cuda::GpuMat &A, cv::cuda::GpuMat &B, cv::cuda::GpuMat &dst);
__host__ void subtract_gpu(cv::cuda::GpuMat &A, float b, cv::cuda::GpuMat &dst);

__host__ void divide_gpu(cv::cuda::GpuMat &A, cv::cuda::GpuMat &B, cv::cuda::GpuMat &dst);
__host__ void divide_gpu(cv::cuda::GpuMat &A, float b, cv::cuda::GpuMat &dst);

__host__ void multiply_gpu(cv::cuda::GpuMat &A, cv::cuda::GpuMat &B, cv::cuda::GpuMat &dst);
__host__ void multiply_gpu(cv::cuda::GpuMat &A, float b, cv::cuda::GpuMat &dst);

#endif