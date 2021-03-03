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


Mat guided_filter(Mat &I, Mat &p, int r) {
    Mat Ones = Mat::ones(p.rows, p.cols, MAT_TYPE);

    copy_zeros(p, Ones);
    copy_zeros(p, I);
    print_val(I, "I");
    
    
    Mat N = box_filter(Ones, r);    
  
    Mat sum_I = box_filter(I, r);
    Mat mean_I = divide(sum_I, N);
    
    Mat sum_p = box_filter(p, r);
    Mat mean_p = divide(sum_p, N);
    
    print_val(mean_I, "mean_I");
    print_val(mean_p, "mean_p");

    Mat Ip = multiply(I, p);
    Mat sum_Ip = box_filter(Ip, r);
    Mat mean_Ip = divide(sum_Ip, N);

    Mat mean_I_p = multiply(mean_I, mean_p);
    Mat cov_Ip = subtract(mean_Ip, mean_I_p);
    
    Mat II = multiply(I, I);
    print_val(II, "II");
    Mat sum_II = box_filter(II, r);
    Mat mean_II = divide(sum_II, N);
    
    
    Mat mean_I_I = multiply(mean_I, mean_I);
    Mat var_I = subtract(mean_II, mean_I_I);
    
    remove_negatives(var_I);
    
    print_val(sum_II, "sumII");
    print_val(mean_II, "meanII");
    print_val(var_I, "var_I");
    
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


int main(int argc, char *argv[]) {
    Mat I;
    Mat p;
    Mat q;

    I = imread("data_sets/guided-filter-bilateral-off/scan-16-in-guide.tif", IMREAD_ANYDEPTH);
    p = imread("data_sets/guided-filter-bilateral-off/scan-16-in-depth.tif", IMREAD_ANYDEPTH);
    Mat bilateral = imread("data_sets/guided-filter-bilateral-off/scan-16-out-depth.tif", IMREAD_ANYDEPTH);
    I.convertTo(I, MAT_TYPE, 1.0);
    p.convertTo(p, MAT_TYPE, 1.0);

    
    while (1) {
        cin >> r >> eps;
        Mat A = multiply(p, p);
        Mat new_I = A;/// sqrtMat(A);
        //Mat I = multiply(p, p);
        float p_max = max_val(p);
        float I_max = max_val(new_I);
        
        new_I.convertTo(new_I, MAT_TYPE, 1.0/I_max);
        p.convertTo(p, MAT_TYPE, 1.0/p_max);
        
        show_grayscale(new_I, "I");
        
        show(bilateral, "b");

        Mat res = Mat(p.rows, p.cols, MAT_TYPE);
         
        res = guided_filter(new_I, p, r);
        //float max_res = max_val(res);
       // res.convertTo(res, MAT_TYPE, 1.0/max_res);
        show_grayscale(res, "q");
        
        print_val(res, "res");
       // res = res.mul(2515.0);
        show(res, "q rainbow");
        res.convertTo(res, CV_16UC1, 65535);
        imwrite("output-new.png", res);
       
        cout << "done" << endl;
         char k = waitKey();       
    }
    /*
    GpuMat src;
    GpuMat dst(4096, 4096, CV_32FC1);
    Mat res;
    Mat Ones = Mat::ones(4096, 4096, CV_32FC1);
    
    src.upload(Ones);
    cum_sum_gpu(src, dst);
    dst.download(res);*/
   //cout << res << endl;

    return 0;
}
