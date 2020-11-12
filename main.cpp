#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

int main() {
    Mat img = imread("data_sets/guided-filter-bilateral-off/scan-12-in-depth.tif", IMREAD_ANYDEPTH);
    if(img.empty()) {
        std::cout << "Could not read the image: " << std::endl;
        return 1;
    }
    imshow("Display window", img);
    char a = waitKey();

    return 0;
}
