#include "kernel_sum.cuh"

__global__ void kernel_sum_x(const cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst, int r, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (x < (cols - r)) dst(blockIdx.y, x) = src(blockIdx.y, x + r);
	else if (x < cols) dst(blockIdx.y, x) = src(blockIdx.y, cols - 1);
	
	__syncthreads();
	
	if ((x > r) && x < cols) {
        	float a = src(blockIdx.y, x - r - 1);
        	if(a != 0) atomicAdd(&dst(blockIdx.y, x), -a);
    }
}

__global__ void kernel_sum_y(const cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst, int r, int rows) {
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (y < (rows - r)) dst(y, blockIdx.x) = src(y + r, blockIdx.x);
	else if (y < rows) dst(y, blockIdx.x) = src(rows - 1, blockIdx.x);
	
	__syncthreads();
	
	if ((y > r) && y < rows) {
        	float a = src(y - r - 1, blockIdx.x);
        	if(a != 0) atomicAdd(&dst(y, blockIdx.x), -a);
    }
}

__host__ void kernel_sum_gpu(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, int r) {
	
	const int m = 256;
	int numRows = src.rows, numCols = src.cols;
	
	const dim3 gridSizeX(ceil((float)numCols / m), numRows, 1);
	const dim3 blockSizeX(m, 1, 1);
	
	const dim3 gridSizeY(numCols, ceil((float)numRows / m), 1);
	const dim3 blockSizeY(1, m, 1);
	
	GpuMat mid_dst(numRows, numCols, CV_32FC1);

    kernel_sum_x<<<gridSizeX, blockSizeX>>>(src, mid_dst, r, numCols);
    kernel_sum_y<<<gridSizeY, blockSizeY>>>(mid_dst, dst, r, numRows);
}