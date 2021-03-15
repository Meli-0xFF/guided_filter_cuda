#ifndef INTEGRAL_IMAGE_H
#define INTEGRAL_IMAGE_H

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

#define BLOCK_SIZE 32

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
 ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

__host__ void integral_image_gpu(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);

__global__ void inclusive_scan_x(const cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst,  cv::cuda::PtrStepSzf block_sums, int cols);
__global__ void exclusive_scan_x(const cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst, int n);
__global__ void add_scans_x(cv::cuda::PtrStepSzf dst, cv::cuda::PtrStepSzf block_sums_scan, int cols);
__global__ void inclusive_scan_y(const cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst,  cv::cuda::PtrStepSzf block_sums, int rows);
__global__ void exclusive_scan_y(const cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst, int n);
__global__ void add_scans_y(cv::cuda::PtrStepSzf dst, cv::cuda::PtrStepSzf block_sums_scan, int rows);


#endif