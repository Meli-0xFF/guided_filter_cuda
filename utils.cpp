#include "utils.h"

void cumulative_sum(Mat &srcMat, Mat &dstMat, char dim) {
    CV_Assert(srcMat.depth() == CV_8U);

    if (dim == 'x') {
        for (int i = 0; i < srcMat.rows; i++) {
            int row_cum_sum = 0;
            for (int j = 0; j < srcMat.cols; j++) {
                dstMat.at<uchar>(i, j) = srcMat.at<uchar>(i, j) + row_cum_sum;
                row_cum_sum += srcMat.at<uchar>(i, j);
            }
        }
    } else if (dim == 'y') {
        for (int i = 0; i < srcMat.cols; i++) {
            int col_cum_sum = 0;
            for (int j = 0; j < srcMat.rows; j++) {
                dstMat.at<uchar>(j, i) = srcMat.at<uchar>(j, i) + col_cum_sum;
                col_cum_sum += srcMat.at<uchar>(j, i);
            }
        }
    } else {
        std::cout << "Invalid dim parameter: You can choose from x or y dimension parameter.";
    }
}


