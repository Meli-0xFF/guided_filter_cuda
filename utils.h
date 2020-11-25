#ifndef GUIDED_FILTER_CUDA_UTILS_H
#define GUIDED_FILTER_CUDA_UTILS_H
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

void cumulative_sum(Mat &srcMat, Mat &dstMat, char dim);

#endif //GUIDED_FILTER_CUDA_UTILS_H
