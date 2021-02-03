#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <iomanip>
#include "utils.h"

using namespace cv;
using namespace std;

int r;
float eps;

Mat I;
Mat p;
Mat q;

Mat guided_filter() {
    Mat Ones = Mat::ones(p.rows, p.cols, CV_32F);
    Mat N = box_filter(Ones, r);

    Mat mean_I = box_filter(I, r) / N;
    Mat mean_p = box_filter(p, r) / N;

    Mat Ip = I.mul(p);
    Mat mean_Ip = box_filter(Ip, r) / N;

    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

    Mat II = I.mul(I);
    Mat mean_II = box_filter(II, r) / N;
    Mat var_I = mean_II - mean_I.mul(mean_I);
    Mat a = cov_Ip / (var_I + eps);
    Mat b = mean_p - a.mul(mean_I);
    Mat mean_a = box_filter(a, r) / N;
    Mat mean_b = box_filter(b, r) / N;
    Mat q = Mat(p.rows, p.cols, CV_64F);
    q = mean_a.mul(I) + mean_b;

    return q;
}

int main() {
    I = imread("data_sets/guided-filter-bilateral-off/scan-16-in-guide.tif", IMREAD_ANYDEPTH);
    p = imread("data_sets/guided-filter-bilateral-off/scan-16-in-depth.tif", IMREAD_ANYDEPTH);
    Mat bilateral = imread("data_sets/guided-filter-bilateral-off/scan-16-out-depth.tif", IMREAD_ANYDEPTH);
    I.convertTo(I, CV_32F, 1/65535.0);

    while (1) {
        cin >> r >> eps;
       // 1 0.0000056
        Mat res = guided_filter();
        show(res, "q");

       // show(p, "p");
       // show(I, "I");

      //  print_val(p, "p");
      //  print_val(I, "I");
       // print_val(res, "q");


      //  print_val(bilateral, "bilateral");

     //   Mat Ones = Mat::ones(10, 10, CV_32F);
       // Ones *= 1000000000;
      //  Ones += 0.1f;

      //  Mat N = box_filter(Ones, r);
       // cout << Ones << endl;
   //     cout << N;

     /*   Mat Ones = Mat::ones(p.rows, p.cols, CV_32F);
        Mat N = box_filter(Ones, r);
        Mat mean_I = box_filter(I, r) / N;
        Mat mean_p = box_filter(p, r);
        print_val(N, "N");
        print_val(p, "p");
        print_val(mean_p, "mean_p");*/
      //  print_val(mean_I, "mean_I");

      //  Mat Ip = I.mul(p);
     //   Mat mean_Ip = box_filter(Ip, r) / N;
     //   show(mean_p, "mean_p");
   //     show(mean_I, "mean_I");

   //     print_val(Ip, "Ip");
   //     print_val(mean_Ip, "mean_Ip");

        char k = waitKey();
    }

    return 0;
}
