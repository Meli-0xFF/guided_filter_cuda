#ifndef GUIDED_FILTER_CUDA_UTILS_H
#define GUIDED_FILTER_CUDA_UTILS_H
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <type_traits>

using namespace cv;

#define MAT_TYPE CV_32F

float max_val(Mat &mat);
void cumulative_sum(Mat &srcMat, Mat &dstMat, char dim);
Mat box_filter(Mat &srcMat, int r);
Mat show(Mat &mat, String name);
void print_val(Mat &mat, String name);
void integralImage(Mat &i, Mat &ii, Mat &S, int x, int y);
Mat mat_box_filter(Mat &srcMat, int r);
#endif //GUIDED_FILTER_CUDA_UTILS_H
