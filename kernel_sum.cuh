#ifndef KERNEL_SUM_H
#define KERNEL_SUM_H

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

__host__ void kernel_sum_gpu(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, int r);

__global__ void kernel_sum_x(const cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst, int r, int cols);
__global__ void kernel_sum_y(const cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst, int r, int rows);

#endif