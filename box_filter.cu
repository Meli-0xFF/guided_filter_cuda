#include "box_filter.cuh"

__host__ void box_filter_gpu(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, int r) { 
    const int m = 256;
    int numRows = src.rows, numCols = src.cols;
	
    const dim3 gridSizeX(ceil((float)numCols / m), numRows, 1);
    const dim3 blockSizeX(m, 1, 1);
	
    const dim3 gridSizeY(numCols, ceil((float)numRows / m), 1);
    const dim3 blockSizeY(1, m, 1);
	
    GpuMat block_sums_x(numRows, (int)ceil((float)numCols / m), CV_32FC1);
    GpuMat block_sums_scan_x(numRows, (int)ceil((float)numCols / m), CV_32FC1);
	
    GpuMat tmp(numRows, numCols, CV_32FC1);
	
    GpuMat block_sums_y((int)ceil((float)numRows / m), numCols, CV_32FC1);
    GpuMat block_sums_scan_y((int)ceil((float)numRows / m), numCols, CV_32FC1);

    int a = 1, b = 1;
    while (a < (int)ceil((float)numCols / m)) a <<= 1;
    while (b < (int)ceil((float)numRows / m)) b <<= 1;

    inclusive_scan_x<<<gridSizeX, blockSizeX, sizeof(float)*blockSizeX.x*NUM_BANKS + sizeof(float)*blockSizeX.x>>>(src, tmp, block_sums_x, numCols);
    exclusive_scan_x<<<numRows, 32, sizeof(float)*a*NUM_BANKS + sizeof(float)*a>>>(block_sums_x, block_sums_scan_x, a);  
    add_scans_x<<<gridSizeX, blockSizeX, sizeof(float)>>>(tmp, block_sums_scan_x, numCols);
    
    inclusive_scan_y<<<gridSizeY, blockSizeY, sizeof(float)*blockSizeY.y*NUM_BANKS + sizeof(float)*blockSizeY.y>>>(tmp, dst, block_sums_y, numRows);   
    exclusive_scan_y<<<numCols, 32, sizeof(float)*b*NUM_BANKS + sizeof(float)*b>>>(block_sums_y, block_sums_scan_y, b);    
    add_scans_y<<<gridSizeY, blockSizeY, sizeof(float)>>>(dst, block_sums_scan_y, numRows);

    kernel_sum_x<<<gridSizeX, blockSizeX>>>(dst, tmp, r, numCols);
    kernel_sum_y<<<gridSizeY, blockSizeY>>>(tmp, dst, r, numRows);

    copy_zeros_gpu(src, dst);
}