#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "utils.h"

using namespace cv;
using namespace std;

int main() {
    Mat src = Mat::ones(5, 5, CV_8U);
    Mat dst = Mat::zeros(5, 5, CV_8U);
    Mat dst2 = Mat::zeros(5, 5, CV_8U);
    cumulative_sum(src, dst, 'x');
    cumulative_sum(src, dst2, 'y');
    cout << "src = " << endl << " " << src << endl;
    cout << "dst = " << endl << " " << dst << endl;
    cout << "dst2 = " << endl << " " << dst2 << endl;

    return 0;
}
