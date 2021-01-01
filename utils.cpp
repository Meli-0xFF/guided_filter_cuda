#include "utils.h"

using namespace std;

void print_val(Mat &mat, String name) {
    double minVal;
    double maxVal;

    minMaxLoc(mat, &minVal, &maxVal);
    cout << name << " min = " << minVal << " max = " << maxVal << endl;
}

void cumulative_sum(Mat &srcMat, Mat &dstMat, char dim) {
    if (dim == 'x') {
        for (int i = 0; i < srcMat.rows; i++) {
            double row_cum_sum = 0;
            for (int j = 0; j < srcMat.cols; j++) {
                dstMat.at<double>(i, j) = srcMat.at<double>(i, j) + row_cum_sum;
                row_cum_sum += srcMat.at<double>(i, j);
            }
        }
    } else if (dim == 'y') {
        for (int i = 0; i < srcMat.cols; i++) {
            double col_cum_sum = 0;
            for (int j = 0; j < srcMat.rows; j++) {
                dstMat.at<double>(j, i) = srcMat.at<double>(j, i) + col_cum_sum;
                col_cum_sum += srcMat.at<double>(j, i);
            }
        }
    } else {
        std::cout << "Invalid dim parameter: You can choose from x or y dimension parameter.";
    }
}

Mat box_filter(Mat &srcMat, int r) {
    Mat cumMat = Mat::zeros(srcMat.rows, srcMat.cols, CV_64F);
    Mat dstMat = Mat();


    cumulative_sum(srcMat, cumMat, 'y');
    dstMat.push_back(cumMat.rowRange(r, 2*r + 1));
    dstMat.push_back((cumMat.rowRange(2*r + 1, cumMat.rows) - cumMat.rowRange(0, cumMat.rows - 2*r - 1)));
    for (int i = cumMat.rows - 2*r - 1; i < cumMat.rows - r - 1; i++)
        dstMat.push_back(cumMat.row(cumMat.rows - 1) - cumMat.row(i));



    cumulative_sum(dstMat, cumMat, 'x');
    rotate(cumMat, cumMat, cv::ROTATE_90_CLOCKWISE);
    dstMat = Mat();
    dstMat.push_back(cumMat.rowRange(r, 2*r + 1));
    dstMat.push_back((cumMat.rowRange(2*r + 1, cumMat.rows) - cumMat.rowRange(0, cumMat.rows - 2*r - 1)));
    for (int i = cumMat.rows - 2*r - 1; i < cumMat.rows - r - 1; i++)
        dstMat.push_back(cumMat.row(cumMat.rows - 1) - cumMat.row(i));


    rotate(dstMat, dstMat, cv::ROTATE_90_COUNTERCLOCKWISE);

    return dstMat;
}



void show (Mat &mat, String name) {
    double minVal;
    double maxVal;
    minMaxLoc(mat, &minVal, &maxVal);
    Mat res = Mat(mat.rows, mat.cols, CV_8U);
    mat.convertTo(res, CV_8U, 255.0/maxVal);

    imshow("Display " + name, res);
}

