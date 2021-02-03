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

Mat mat_box_filter(Mat &srcMat, int r) {
    Mat cumMat = Mat::zeros(srcMat.rows, srcMat.cols, CV_32F);
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

float S(Mat &i, Mat &sMat, int x, int y, int offset) {
    if (y <= offset - 1) return 0;
    if (sMat.at<float>(x, y) != -1) return sMat.at<float>(x, y);
    sMat.at<float>(x, y) = (y >= offset ? S(i, sMat, x, y - 1, offset) : 0) + ((y >= i.cols + offset) ? 0 : i.at<float>(x - offset, y - offset));
    return sMat.at<float>(x, y);
}

float ii(Mat &i, Mat &iiMat, Mat &sMat, int x, int y, int offset) {
    if (x <= offset - 1) {
        iiMat.at<float>(x, y) = 0;
        return 0;
    }
    if (iiMat.at<float>(x, y) != -1) return iiMat.at<float>(x, y);
    iiMat.at<float>(x, y) = ((x >= i.rows + offset) ? 0 : S(i, sMat, x, y, offset)) + (x >= offset ? ii(i, iiMat, sMat, x - 1, y, offset) : 0);
    return iiMat.at<float>(x, y);
}

Mat box_filter(Mat &srcMat, int r) {
    Mat iiMat = Mat::ones(srcMat.rows + 2*r + 1, srcMat.cols + 2*r + 1, CV_32F);
    iiMat *= -1.0f;
    Mat sMat = Mat::ones(srcMat.rows + 2*r + 1, srcMat.cols + 2*r + 1, CV_32F);
    sMat *= -1.0f;
    Mat resMat = Mat::zeros(srcMat.rows, srcMat.cols, CV_32F);

    for (int x = r + 1; x < iiMat.rows - r; x++) {
        for (int y = r + 1; y < iiMat.cols - r; y++) {
            resMat.at<float>(x - r - 1, y - r - 1) = ii(srcMat, iiMat, sMat, x + r, y + r, r + 1) -
                                                        ii(srcMat, iiMat, sMat, x - r - 1, y + r, r + 1) -
                                                        ii(srcMat, iiMat, sMat, x + r, y - r - 1, r + 1) +
                                                        ii(srcMat, iiMat, sMat, x - r - 1, y - r - 1, r + 1);
        }
    }
    return resMat;
}

// Convert grayscale depthmap to repeated rainbow
cv::Mat ConvertToRainbow(cv::Mat grayscale)
{
    const float rainbowMultiplier = 32; // how many times to repeat rainbow in whole intervat
    const double distanceMin = 100;
    const double distanceMax = 1500;
    const double distanceToLightnessCoefficient = 256.0 / (distanceMax - distanceMin);
    cv::Mat shiftedGrayscale;
    grayscale.convertTo(shiftedGrayscale, CV_32F);
    shiftedGrayscale = (shiftedGrayscale - distanceMin) * distanceToLightnessCoefficient;
    for(int i=0; i<shiftedGrayscale.rows; i++)
        for(int j=0; j<shiftedGrayscale.cols; j++)
            shiftedGrayscale.at<float>(i, j) = static_cast<double>(static_cast<int32_t>(shiftedGrayscale.at<float>(i, j) * rainbowMultiplier) % 176);
    shiftedGrayscale.convertTo(shiftedGrayscale, CV_8U);
    grayscale.convertTo(grayscale, CV_8U);
    cv::Mat rainbow;
    cv::cvtColor(grayscale, rainbow, CV_GRAY2BGR);
    cv::cvtColor(rainbow, rainbow, CV_BGR2HSV);
    std::vector<cv::Mat> hsvChannels;
    cv::split(rainbow, hsvChannels);
    hsvChannels[0] = shiftedGrayscale;
    cv::threshold(grayscale, hsvChannels[1], 0, 255, cv::THRESH_BINARY);
    cv::threshold(grayscale, hsvChannels[2], 0, 255, cv::THRESH_BINARY);
    cv::merge(hsvChannels, rainbow);
    cv::cvtColor(rainbow, rainbow, CV_HSV2BGR);
    return rainbow;
}

void show (Mat &mat, String name) {
    double minVal;
    double maxVal;
    minMaxLoc(mat, &minVal, &maxVal);
    Mat res = Mat(mat.rows, mat.cols, CV_8U);

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            if (mat.at<float>(i, j) < 0) mat.at<float>(i, j) = 0;
        }
    }



    mat.convertTo(res, CV_8U, 255 / maxVal);

    imshow("Display " + name, ConvertToRainbow(mat));
}

