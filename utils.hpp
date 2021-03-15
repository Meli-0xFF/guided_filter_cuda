#ifndef GUIDED_FILTER_CUDA_UTILS_H
#define GUIDED_FILTER_CUDA_UTILS_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include "opencv2/core/cuda.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include "integral_image.cuh"
#include "mat_operations.cuh"
#include "mat_corrections.cuh"
#include "box_filter.cuh"
#include <math.h> 

using namespace cv;
using namespace std;

#define MAT_TYPE CV_32F

Mat box_filter(Mat &srcMat, int r);

// Getting and printing minimum and maximum value of matrix
float max_val(Mat &mat);
void print_val(Mat &mat, String name);

// Element-wise arithmetic operations on matrices
Mat add(Mat &A, Mat &B);
Mat subtract(Mat &A, Mat &B);
Mat multiply(Mat &A, Mat &B);
Mat divide(Mat &A, Mat &B);
Mat absMat(Mat &A);

// Visuaization of matrices
Mat show_rainbow(Mat &mat, String name);
Mat show_grayscale(Mat &mat, String name);

// Handling undefined or invalid values on matrices
void copy_zeros(Mat &src, Mat &dst);
void remove_negatives(Mat &mat);

// Matrices difference measurement
Mat test(Mat &A, Mat &B);

#endif //GUIDED_FILTER_CUDA_UTILS_H
