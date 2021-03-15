#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <iomanip>
#include "utils.hpp"


using namespace cv;
using namespace std;
using namespace cv::cuda;

int r;
float eps, eps_p, eps_I;


Mat guided_filter_cpu(Mat &I, Mat &p, int r) {
    Mat Ones = Mat::ones(p.rows, p.cols, MAT_TYPE);

    copy_zeros(p, Ones);
    copy_zeros(p, I);
    
    Mat N = box_filter(Ones, r);    
  
    Mat sum_I = box_filter(I, r);
    Mat mean_I = divide(sum_I, N);
    
    Mat sum_p = box_filter(p, r);
    Mat mean_p = divide(sum_p, N);

    Mat Ip = multiply(I, p);
    Mat sum_Ip = box_filter(Ip, r);
    Mat mean_Ip = divide(sum_Ip, N);

    Mat mean_I_p = multiply(mean_I, mean_p);
    Mat cov_Ip = subtract(mean_Ip, mean_I_p);
    
    Mat II = multiply(I, I);
    Mat sum_II = box_filter(II, r);
    Mat mean_II = divide(sum_II, N);
    
    Mat mean_I_I = multiply(mean_I, mean_I);
    Mat var_I = subtract(mean_II, mean_I_I);
    
    remove_negatives(var_I);

    Mat var_I_eps = var_I + eps;
    Mat a = divide(cov_Ip, var_I_eps);    

    Mat a_mean_I = multiply(a, mean_I);
    Mat b = subtract(mean_p, a_mean_I);    

    Mat sum_a = box_filter(a, r);
    Mat mean_a = divide(sum_a, N);

    Mat sum_b = box_filter(b, r);
    Mat mean_b = divide(sum_b, N);

    Mat q = Mat(p.rows, p.cols, MAT_TYPE);
    Mat mean_a_I = multiply(mean_a, I);
    q = add(mean_a_I, mean_b);

    remove_negatives(q);

    return q;
}

Mat guided_filter_gpu(Mat &I, Mat &p, int r) {
    GpuMat d_p;
    GpuMat d_I;
    GpuMat d_ones;
    
    Mat Ones = Mat::ones(p.rows, p.cols, CV_32FC1);

    d_p.upload(p);
    d_I.upload(I);
    d_ones.upload(Ones);
    
    struct timeval t1, t2;
    gettimeofday(&t1, 0);
    
    copy_zeros_gpu(d_p, d_ones);
    copy_zeros_gpu(d_p, d_I);
    
    GpuMat N(p.rows, p.cols, MAT_TYPE);
    box_filter_gpu(d_ones, N, r);        
  
    GpuMat sum_I(p.rows, p.cols, MAT_TYPE);
    box_filter_gpu(d_I, sum_I, r);
    GpuMat mean_I(p.rows, p.cols, MAT_TYPE);
    divide_gpu(sum_I, N, mean_I);
    
    GpuMat sum_p(p.rows, p.cols, MAT_TYPE);
    box_filter_gpu(d_p, sum_p, r);
    GpuMat mean_p(p.rows, p.cols, MAT_TYPE);
    divide_gpu(sum_p, N, mean_p);

    GpuMat Ip(p.rows, p.cols, MAT_TYPE);
    multiply_gpu(d_I, d_p, Ip);   
    
    GpuMat sum_Ip(p.rows, p.cols, MAT_TYPE);
    box_filter_gpu(Ip, sum_Ip, r);
    GpuMat mean_Ip(p.rows, p.cols, MAT_TYPE);
    divide_gpu(sum_Ip, N, mean_Ip);

    GpuMat mean_I_p(p.rows, p.cols, MAT_TYPE);
    multiply_gpu(mean_I, mean_p, mean_I_p);
    
    GpuMat cov_Ip(p.rows, p.cols, MAT_TYPE);
    subtract_gpu(mean_Ip, mean_I_p, cov_Ip);
    
    GpuMat II(p.rows, p.cols, MAT_TYPE);
    multiply_gpu(d_I, d_I, II);
    
    GpuMat sum_II(p.rows, p.cols, MAT_TYPE);
    box_filter_gpu(II, sum_II, r);
    GpuMat mean_II(p.rows, p.cols, MAT_TYPE);
    divide_gpu(sum_II, N, mean_II);
    
    GpuMat mean_I_I(p.rows, p.cols, MAT_TYPE);
    multiply_gpu(mean_I, mean_I, mean_I_I);
    
    GpuMat var_I(p.rows, p.cols, MAT_TYPE);
    subtract_gpu(mean_II, mean_I_I, var_I);
    
    remove_negatives_gpu(var_I);

    GpuMat var_I_eps(p.rows, p.cols, MAT_TYPE);
    add_gpu(var_I, eps, var_I_eps);
    
    GpuMat a(p.rows, p.cols, MAT_TYPE);
    divide_gpu(cov_Ip, var_I_eps, a);    

    GpuMat a_mean_I(p.rows, p.cols, MAT_TYPE);
    multiply_gpu(a, mean_I, a_mean_I);
    
    GpuMat b(p.rows, p.cols, MAT_TYPE);
    subtract_gpu(mean_p, a_mean_I, b);  

    GpuMat sum_a(p.rows, p.cols, MAT_TYPE);
    box_filter_gpu(a, sum_a, r);
    GpuMat mean_a(p.rows, p.cols, MAT_TYPE);
    divide_gpu(sum_a, N, mean_a);

    GpuMat sum_b(p.rows, p.cols, MAT_TYPE);
    box_filter_gpu(b, sum_b, r);
    GpuMat mean_b(p.rows, p.cols, MAT_TYPE);
    divide_gpu(sum_b, N, mean_b);

    GpuMat q_d(p.rows, p.cols, CV_32FC1);
    GpuMat mean_a_I(p.rows, p.cols, CV_32FC1);
    multiply_gpu(mean_a, d_I, mean_a_I);
    add_gpu(mean_a_I, mean_b, q_d);

    remove_negatives_gpu(q_d);
           
    gettimeofday(&t2, 0);

    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

    printf("BOX_FILTER GPU time:  %3.6f ms \n", time);
    
    Mat q;
    q_d.download(q);
    
    return q;
}


int main(int argc, char *argv[]) {
    
    Mat p = imread("data_sets/guided-filter-bilateral-off/scan-16-in-depth.tif", IMREAD_ANYDEPTH);
    Mat I = imread("data_sets/guided-filter-bilateral-off/scan-16-in-guide.tif", IMREAD_ANYDEPTH);
    Mat bilateral = imread("data_sets/guided-filter-bilateral-off/scan-16-out-depth.tif", IMREAD_ANYDEPTH);
    
    I.convertTo(I, MAT_TYPE, 1.0);
    
    cin >> r >> eps;
    Mat res1;
    Mat res2;
    struct timeval t1, t2;
    
    res1 = guided_filter_gpu(I, p, r);
        
    gettimeofday(&t1, 0);
    
    res2 = guided_filter_cpu(I, p, r);
    
    gettimeofday(&t2, 0);

    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

    printf("BOX_FILTER CPU time:  %3.6f ms \n", time);

    res1.convertTo(res1, CV_32F, 1.0);
    show_grayscale(I, "I");
    show_rainbow(res1, "q GPU rainbow");
    show_rainbow(res2, "q CPU rainbow");
    show_rainbow(bilateral, "bilateral rainbow");
    
    print_val(res1, "q");

    char k = waitKey();
    
    return 0;
}
