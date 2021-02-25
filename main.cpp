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
float eps;

Mat Iii;
Mat Ii;
Mat I;
Mat p;
Mat q;

Mat guided_filter() {
    Mat Ones = Mat::ones(p.rows, p.cols, MAT_TYPE);

    Mat N_full = box_filter(Ones, r);

    copy_zeros(p, Ones);
    copy_zeros(p, I);
    show_grayscale(I, "I");
    
    Mat N = box_filter(Ones, r);    
  
    Mat sum_I = box_filter(I, r);
    Mat mean_I = divide(sum_I, N);
    
    Mat sum_p = box_filter(p, r);
    Mat mean_p = divide(sum_p, N);
    
    print_val(mean_I, "mean_I");
    print_val(mean_p, "mean_p");

    Mat Ip = I.mul(p);
    Mat sum_Ip = box_filter(Ip, r);
    Mat mean_Ip = divide(sum_Ip, N);

    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    
    Mat sum_I_full = box_filter(Iii, r);
    Mat mean_I_full = divide(sum_I_full, N_full);

    Mat II_full = Iii.mul(Iii);
    Mat mean_II_full = box_filter(II_full, r) / N_full;
    Mat var_I_full = mean_II_full - mean_I_full.mul(mean_I_full);
    
    Mat II = I.mul(I);
    Mat sum_II = box_filter(II, r);
    Mat mean_II = divide(sum_II, N);
    
    Mat var_I = mean_II - mean_I.mul(mean_I);
    
    print_val(sum_II, "sumII");
    print_val(var_I, "var_I");
    
    Mat var_I_eps = var_I + eps;
    Mat a = cov_Ip / var_I_eps;
    
    copy_zeros(p, a);
    
    Mat b = mean_p - a.mul(mean_I);
    
    copy_zeros(p, b);
    
    Mat sum_a = box_filter(a, r);
    Mat mean_a = divide(sum_a, N);
    
    Mat sum_b = box_filter(b, r);
    Mat mean_b = divide(sum_b, N);
    
    Mat q = Mat(p.rows, p.cols, MAT_TYPE);
    Mat q_final = Mat(p.rows, p.cols, MAT_TYPE);
    q = mean_a.mul(Ii) + mean_b;
    
    copy_zeros(p, q);
    
    print_val(var_I_full, "blbost");
    
    float treshold = mat_treshold(var_I_full, 10.0);
    cout << treshold << endl;
    noise_to_zero(q, var_I_full, N, q_final, treshold, 10.0);
    
    return q_final;
}


int main(int argc, char *argv[]) {
    I = imread("data_sets/guided-filter-bilateral-off/scan-16-in-guide.tif", IMREAD_ANYDEPTH);
    p = imread("data_sets/guided-filter-bilateral-off/scan-16-in-depth.tif", IMREAD_ANYDEPTH);
    Mat bilateral = imread("data_sets/guided-filter-bilateral-off/scan-16-out-depth.tif", IMREAD_ANYDEPTH);
    I.convertTo(I, MAT_TYPE, 1.0);
    p.convertTo(p, MAT_TYPE, 1.0);

    float p_max = max_val(p);
    float I_max = max_val(I);
    
  //  I.convertTo(I, MAT_TYPE, 1.0/I_max);
  //  p.convertTo(p, MAT_TYPE, 1.0/p_max);
 //   show_grayscale(I, "I");
  //  show_grayscale(p, "p");
    
    Ii = I.clone();
    Iii = I.clone();
    
    while (1) {
        cin >> r >> eps;
        
        show(bilateral, "b");
        show(p, "p");

        Mat res = Mat(p.rows, p.cols, MAT_TYPE);
        
        res = guided_filter();
        show_grayscale(res, "q");
        float max_res = max_val(res);
        print_val(res, "res");
       // res = res.mul(1521.0);
        show(res, "res rainbow");
        res.convertTo(res, CV_16UC1, 65535);
        imwrite("output.png", res);
       
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
