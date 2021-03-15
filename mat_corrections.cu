#include "mat_corrections.cuh"

__global__ void copy_zeros(const cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((x < cols) && (src(blockIdx.y, x) == 0)) dst(blockIdx.y, x) = 0;
}

__global__ void remove_negatives(cv::cuda::PtrStepSzf src, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((x < cols) && (src(blockIdx.y, x) < 0)) src(blockIdx.y, x) = 0;
}

__host__ void copy_zeros_gpu(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst) {
	const int m = 256;
	int numRows = src.rows, numCols = src.cols;
	
	const dim3 gridSizeX(ceil((float)numCols / m), numRows, 1);
	const dim3 blockSizeX(m, 1, 1);

    copy_zeros<<<gridSizeX, blockSizeX>>>(src, dst, numCols);
}

__host__ void remove_negatives_gpu(cv::cuda::GpuMat &src) {
	const int m = 256;
	int numRows = src.rows, numCols = src.cols;
	
	const dim3 gridSizeX(ceil((float)numCols / m), numRows, 1);
	const dim3 blockSizeX(m, 1, 1);

    remove_negatives<<<gridSizeX, blockSizeX>>>(src, numCols);
}