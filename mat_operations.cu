#include "mat_operations.cuh"

__global__ void add(cv::cuda::PtrStepSzf A, cv::cuda::PtrStepSzf B, cv::cuda::PtrStepSzf dst, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (x < cols) {
        	float a = A(blockIdx.y, x);
        	float b = B(blockIdx.y, x);
        	(a == 0 && b == 0) ? dst(blockIdx.y, x) = 0 : dst(blockIdx.y, x) = a + b;
    }
}

__global__ void add(cv::cuda::PtrStepSzf A, cv::cuda::PtrStepSzf dst, float b, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (x < cols) dst(blockIdx.y, x) = A(blockIdx.y, x) + b;
}

__global__ void subtract(cv::cuda::PtrStepSzf A, cv::cuda::PtrStepSzf B, cv::cuda::PtrStepSzf dst, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (x < cols) {
        	float a = A(blockIdx.y, x);
        	float b = B(blockIdx.y, x);
        	dst(blockIdx.y, x) = a - b;
    }
}

__global__ void subtract(cv::cuda::PtrStepSzf A, cv::cuda::PtrStepSzf dst, float b, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (x < cols) {
         float a = A(blockIdx.y, x);   
        	//(a == 0) ? dst(blockIdx.y, x) = 0 : 
        	dst(blockIdx.y, x) = a - b;
    }
}

__global__ void divide(cv::cuda::PtrStepSzf A, cv::cuda::PtrStepSzf B, cv::cuda::PtrStepSzf dst, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (x < cols) {
        	float a = A(blockIdx.y, x);
        	float b = B(blockIdx.y, x);
        	(b == 0) ? dst(blockIdx.y, x) = 0 : dst(blockIdx.y, x) = a / b;
    }
}

__global__ void divide(cv::cuda::PtrStepSzf A, cv::cuda::PtrStepSzf dst, float b, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (x < cols) {
         float a = A(blockIdx.y, x);   
        	(a == 0) ? dst(blockIdx.y, x) = 0 : dst(blockIdx.y, x) = a / b;
    }
}

__global__ void multiply(cv::cuda::PtrStepSzf A, cv::cuda::PtrStepSzf B, cv::cuda::PtrStepSzf dst, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (x < cols) {
        	float a = A(blockIdx.y, x);
        	float b = B(blockIdx.y, x);
        	(a == 0 || b == 0) ? dst(blockIdx.y, x) = 0 : dst(blockIdx.y, x) = a * b;
    }
}

__global__ void multiply(cv::cuda::PtrStepSzf A, cv::cuda::PtrStepSzf dst, float b, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (x < cols) {
         float a = A(blockIdx.y, x);   
        	(a == 0) ? dst(blockIdx.y, x) = 0 : dst(blockIdx.y, x) = a * b;
    }
}

__host__ void add_gpu(cv::cuda::GpuMat &A, cv::cuda::GpuMat &B, cv::cuda::GpuMat &dst) {
	
	const int m = 256;
	int numRows = A.rows, numCols = A.cols;
	
	const dim3 gridSizeX(ceil((float)numCols / m), numRows, 1);
	const dim3 blockSizeX(m, 1, 1);
    
    add<<<gridSizeX, blockSizeX>>>(A, B, dst, numCols);
}

__host__ void add_gpu(cv::cuda::GpuMat &A, float b, cv::cuda::GpuMat &dst) {
	
	const int m = 256;
	int numRows = A.rows, numCols = A.cols;
	
	const dim3 gridSizeX(ceil((float)numCols / m), numRows, 1);
	const dim3 blockSizeX(m, 1, 1);
    
    add<<<gridSizeX, blockSizeX>>>(A, dst, b, numCols);
}

__host__ void subtract_gpu(cv::cuda::GpuMat &A, cv::cuda::GpuMat &B, cv::cuda::GpuMat &dst) {
	
	const int m = 256;
	int numRows = A.rows, numCols = A.cols;
	
	const dim3 gridSizeX(ceil((float)numCols / m), numRows, 1);
	const dim3 blockSizeX(m, 1, 1);
    
    subtract<<<gridSizeX, blockSizeX>>>(A, B, dst, numCols);
}

__host__ void subtract_gpu(cv::cuda::GpuMat &A, float b, cv::cuda::GpuMat &dst) {
	
	const int m = 256;
	int numRows = A.rows, numCols = A.cols;
	
	const dim3 gridSizeX(ceil((float)numCols / m), numRows, 1);
	const dim3 blockSizeX(m, 1, 1);
    
    subtract<<<gridSizeX, blockSizeX>>>(A, dst, b, numCols);
}

__host__ void divide_gpu(cv::cuda::GpuMat &A, cv::cuda::GpuMat &B, cv::cuda::GpuMat &dst) {
	
	const int m = 256;
	int numRows = A.rows, numCols = A.cols;
	
	const dim3 gridSizeX(ceil((float)numCols / m), numRows, 1);
	const dim3 blockSizeX(m, 1, 1);
    
    divide<<<gridSizeX, blockSizeX>>>(A, B, dst, numCols);
}

__host__ void divide_gpu(cv::cuda::GpuMat &A, float b, cv::cuda::GpuMat &dst) {
	
	const int m = 256;
	int numRows = A.rows, numCols = A.cols;
	
	const dim3 gridSizeX(ceil((float)numCols / m), numRows, 1);
	const dim3 blockSizeX(m, 1, 1);
    
    divide<<<gridSizeX, blockSizeX>>>(A, dst, b, numCols);
}

__host__ void multiply_gpu(cv::cuda::GpuMat &A, cv::cuda::GpuMat &B, cv::cuda::GpuMat &dst) {
	
	const int m = 256;
	int numRows = A.rows, numCols = A.cols;
	
	const dim3 gridSizeX(ceil((float)numCols / m), numRows, 1);
	const dim3 blockSizeX(m, 1, 1);
    
    multiply<<<gridSizeX, blockSizeX>>>(A, B, dst, numCols);
}

__host__ void multiply_gpu(cv::cuda::GpuMat &A, float b, cv::cuda::GpuMat &dst) {
	
	const int m = 256;
	int numRows = A.rows, numCols = A.cols;
	
	const dim3 gridSizeX(ceil((float)numCols / m), numRows, 1);
	const dim3 blockSizeX(m, 1, 1);
    
    multiply<<<gridSizeX, blockSizeX>>>(A, dst, b, numCols);
}