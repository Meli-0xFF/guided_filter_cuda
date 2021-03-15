#ifndef BOX_FILTER_H
#define BOX_FILTER_H

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
#include "integral_image.cuh"
#include "kernel_sum.cuh"
#include "mat_operations.cuh"
#include "mat_corrections.cuh"

using namespace std;
using namespace cv;
using namespace cuda;

__host__ void box_filter_gpu(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, int r);

#endif