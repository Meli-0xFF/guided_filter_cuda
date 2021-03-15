#ifndef MAT_CORRECTIONS_H
#define MAT_CORRECTIONS_H

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

__host__ void copy_zeros_gpu(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);
__host__ void remove_negatives_gpu(cv::cuda::GpuMat &src);

#endif