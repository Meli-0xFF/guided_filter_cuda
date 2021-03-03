#include "utils.hpp"


using namespace std;


Mat divide(Mat &A, Mat &B) {
    Mat C = Mat(A.rows, A.cols, MAT_TYPE);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            if (B.at<float>(i, j) == 0) C.at<float>(i, j) = 0;
            else if (A.at<float>(i, j) == 0) C.at<float>(i, j) = 0;
            else C.at<float>(i, j) = A.at<float>(i, j) / B.at<float>(i, j);
        }
    }

    return C;
}

Mat subtract(Mat &A, Mat &B) {
    Mat C = Mat(A.rows, A.cols, MAT_TYPE);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            if (B.at<float>(i, j) == A.at<float>(i, j)) C.at<float>(i, j) = 0;
            else if (B.at<float>(i, j) == 0) C.at<float>(i, j) = A.at<float>(i, j);
            else if (A.at<float>(i, j) == 0) C.at<float>(i, j) = (-1) * B.at<float>(i, j);
            else C.at<float>(i, j) = A.at<float>(i, j) - B.at<float>(i, j);
        }
    }

    return C;
}

Mat add(Mat &A, Mat &B) {
    Mat C = Mat(A.rows, A.cols, MAT_TYPE);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            if (B.at<float>(i, j) == 0 && A.at<float>(i, j) == 0) C.at<float>(i, j) = 0;
            else if (B.at<float>(i, j) == 0) C.at<float>(i, j) = A.at<float>(i, j);
            else if (A.at<float>(i, j) == 0) C.at<float>(i, j) = B.at<float>(i, j);
            else C.at<float>(i, j) = A.at<float>(i, j) + B.at<float>(i, j);
        }
    }

    return C;
}

Mat multiply(Mat &A, Mat &B) {
    Mat C = Mat::zeros(A.rows, A.cols, MAT_TYPE);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            if (A.at<float>(i, j) == 0) continue;
            if (B.at<float>(i, j) == 0) continue;
            
            C.at<float>(i, j) = A.at<float>(i, j) * B.at<float>(i, j);
        }
    }

    return C;
}

Mat test(Mat &A, Mat &B) {
    Mat C = Mat::zeros(A.rows, A.cols, MAT_TYPE);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            if (B.at<float>(i, j) == 0) continue;
            else if ((abs(A.at<float>(i, j) - B.at<float>(i, j)) / B.at<float>(i, j)) < 0.0025) C.at<float>(i, j) = B.at<float>(i, j);
            else C.at<float>(i, j) = A.at<float>(i, j);
        }
    }

    return C;
}



Mat sqrtMat(Mat &A) {
     Mat B = Mat::zeros(A.rows, A.cols, MAT_TYPE);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            if (A.at<float>(i, j) > 0) B.at<float>(i, j) = sqrt(A.at<float>(i, j));
            else B.at<float>(i, j) = 0;
        }
    }

    return B;
}

Mat cbrtMat(Mat &A) {
     Mat B = Mat::zeros(A.rows, A.cols, MAT_TYPE);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            if (A.at<float>(i, j) > 0) B.at<float>(i, j) = cbrt(A.at<float>(i, j));
            else B.at<float>(i, j) = 0;
        }
    }

    return B;
}

Mat absMat(Mat &A) {
     Mat B = Mat::zeros(A.rows, A.cols, MAT_TYPE);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            if (A.at<float>(i, j) < 0) B.at<float>(i, j) = 0;//A.at<float>(i, j) * (-1);
            else B.at<float>(i, j) = A.at<float>(i, j);
        }
    }

    return B;
}

void copy_zeros(Mat &src, Mat &dst) {
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (src.at<float>(i, j) == 0) dst.at<float>(i, j) = 0;
        }
    }
}

float max_val(Mat &mat) {
    float maxVal = 0;

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            if(mat.at<float>(i,j) >= maxVal) maxVal = mat.at<float>(i,j);
        }
    }
    return maxVal;
}

void print_val(Mat &mat, String name) {
    float minVal = 10000000;
    float maxVal = 0;

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            if(mat.at<float>(i,j) >= maxVal) maxVal = mat.at<float>(i,j);
            if(mat.at<float>(i,j) < minVal) minVal = mat.at<float>(i,j);
        }
    }

    cout << name << " min = " << minVal << " max = " << maxVal << endl;
}

float mat_treshold(Mat &mat, float part) {
    float minVal = 10000000;
    float maxVal = 0;

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            if(mat.at<float>(i,j) >= maxVal) maxVal = mat.at<float>(i,j);
            if(mat.at<float>(i,j) < minVal) minVal = mat.at<float>(i,j);
        }
    }

    return (maxVal / part);
}

void noise_to_zero(Mat &src, Mat &guide_var, Mat &kernel_pixels, Mat &dst, float treshold, float min_in_kernel) {
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if ((guide_var.at<float>(i, j) < treshold) && (kernel_pixels.at<float>(i, j) < min_in_kernel)) dst.at<float>(i, j) = 0;
            else dst.at<float>(i, j) = src.at<float>(i, j);
        }
    }
}

void cumulative_sum(Mat &srcMat, Mat &dstMat, char dim) {
    if (dim == 'x') {
        for (int i = 0; i < srcMat.rows; i++) {
            float row_cum_sum = 0.0;
            float c = 0.0;
            for (int j = 0; j < srcMat.cols; j++) {
                if (srcMat.at<float>(i, j) == 0.0) {
                    dstMat.at<float>(i, j) = row_cum_sum;
                } else {
                    float y = srcMat.at<float>(i, j) - c;
                    float t = row_cum_sum + y;
                    c = (t - row_cum_sum) - y;
                    row_cum_sum = t;
                    dstMat.at<float>(i, j) = t;
                }
            }
        }
    } else if (dim == 'y') {
        for (int i = 0; i < srcMat.cols; i++) {
            float col_cum_sum = 0.0;
            float c = 0.0;
            for (int j = 0; j < srcMat.rows; j++) {
                if (srcMat.at<float>(j, i) == 0.0) {
                    dstMat.at<float>(j, i) = col_cum_sum;
                } else {
                    float y = srcMat.at<float>(j, i) - c;
                    float t = col_cum_sum + y;
                    c = (t - col_cum_sum) - y;
                    col_cum_sum = t;
                    dstMat.at<float>(j, i) = t;
                }
            }
        }
    } else {
        std::cout << "Invalid dim parameter: You can choose from x or y dimension parameter.";
    }
}

float samples_delta(Mat &srcMat, int i, int j, int k, int l, char dim) {
    if (dim == 'x') {
        if (j < 0) return srcMat.at<float>(k, l);
        else if (l >= srcMat.cols) l = srcMat.cols - 1;
        
        if (srcMat.at<float>(k, l) == 0) return 0;
        else if (srcMat.at<float>(i, j) == 0) return srcMat.at<float>(k, l);
        return (srcMat.at<float>(k, l) - srcMat.at<float>(i, j));
    } else if (dim == 'y') {
        if (i < 0) return srcMat.at<float>(k, l);
        else if (k >= srcMat.rows) k = srcMat.rows - 1;
    
        if (srcMat.at<float>(k, l) == 0) return 0;
        else if (srcMat.at<float>(i, j) == 0) return srcMat.at<float>(k, l);
        return (srcMat.at<float>(k, l) - srcMat.at<float>(i, j));
    } else {
        std::cout << "Invalid dim parameter: You can choose from x or y dimension parameter.";
        return -1;
    }
}

void box_run(Mat &srcMat, Mat &dstMat, char dim, int r) {
    if (dim == 'x') {
        for (int i = 0; i < srcMat.rows; i++) {
            for (int j = 0; j < srcMat.cols; j++) {
                dstMat.at<float>(i, j) = samples_delta(srcMat, i, j - r - 1, i, j + r, 'x');
            }
        }
    } else if (dim == 'y') {
        for (int i = 0; i < srcMat.cols; i++) {
            for (int j = 0; j < srcMat.rows; j++) {
                dstMat.at<float>(j, i) = samples_delta(srcMat, j - r - 1, i, j + r, i, 'y');
            }
        }
    } else {
        std::cout << "Invalid dim parameter: You can choose from x or y dimension parameter.";
    }
}

Mat box_filter(Mat &srcMat, int r) {
    Mat tmpMat = Mat::zeros(srcMat.rows, srcMat.cols, MAT_TYPE);
    
    Mat iMat = Mat::zeros(srcMat.rows, srcMat.cols, MAT_TYPE);
    Mat boxMatX = Mat::zeros(srcMat.rows, srcMat.cols, MAT_TYPE);
    Mat resMat = Mat::zeros(srcMat.rows, srcMat.cols, MAT_TYPE);
    
    cumulative_sum(srcMat, tmpMat, 'x');
    cumulative_sum(tmpMat, iMat, 'y');
    
    box_run(iMat, boxMatX, 'x', r);
    box_run(boxMatX, resMat, 'y', r);
    
    copy_zeros(srcMat, resMat);
    
    return resMat;
}

Mat new_box_filter(Mat &srcMat, int r) {
    Mat cumMat = Mat::zeros(srcMat.rows, srcMat.cols, MAT_TYPE);
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


// Convert grayscale depthmap to repeated rainbow
cv::Mat ConvertToRainbow(cv::Mat grayscale) {
    const double rainbowMultiplier = 32; // how many times to repeat rainbow in whole intervat
    const float distanceMin = 100;
    const float distanceMax = 1500;
    const float distanceToLightnessCoefficient = 256.0 / (distanceMax - distanceMin);
    cv::Mat shiftedGrayscale;
    grayscale.convertTo(shiftedGrayscale, CV_32F);
    shiftedGrayscale = (shiftedGrayscale - distanceMin) * distanceToLightnessCoefficient;
    for(int i=0; i<shiftedGrayscale.rows; i++)
        for(int j=0; j<shiftedGrayscale.cols; j++) 
            shiftedGrayscale.at<float>(i, j) = static_cast<float>(static_cast<int32_t>(shiftedGrayscale.at<float>(i, j) * rainbowMultiplier) % 176);
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

Mat show (Mat &mat, String name) {
    imshow("Display " + name, ConvertToRainbow(mat));
    return mat;
}
 
void remove_negatives(Mat &mat) {
    for (int i = 0; i < mat.rows; i++) {
       for (int j = 0; j < mat.cols; j++) {
           if (mat.at<float>(i, j) < 0) mat.at<float>(i, j) = 0;
       }
   }
}

Mat show_grayscale (Mat &mat, String name) {
    Mat res = Mat(mat.rows, mat.cols, CV_8U);

    mat.convertTo(res, CV_8U, 255 / max_val(mat));

    imshow("Display " + name, res);
    return res;
}

