#include "integral_image.cuh"

__global__ void inclusive_scan_x(const cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst,  cv::cuda::PtrStepSzf block_sums, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ float scan_array[];
	
    if (threadIdx.x >= (blockDim.x/2)) return;
    int first = threadIdx.x;
    int second = threadIdx.x + (blockDim.x/2);
    int bankOffsetFirst = CONFLICT_FREE_OFFSET(first);
    int bankOffsetSecond = CONFLICT_FREE_OFFSET(second);
    
    scan_array[first + bankOffsetFirst] = (x < cols) ? src(blockIdx.y, x) : 0;
    scan_array[second + bankOffsetSecond] = ((x + (blockDim.x/2)) < cols) ? src(blockIdx.y, x + (blockDim.x/2)) : 0;
	
	__syncthreads();

    int stride = 1;
    while(stride <= blockDim.x) {
        int index = (threadIdx.x + 1) * stride * 2 - 1;

        if(index < 2*blockDim.x) 
            atomicAdd(&scan_array[index + CONFLICT_FREE_OFFSET(index)], scan_array[index-stride + CONFLICT_FREE_OFFSET(index-stride)]);

        stride <<= 1;

        __syncthreads();
    } 

    stride = blockDim.x / 2;
    while(stride > 0) {
        int index = (threadIdx.x + 1) * stride * 2 - 1;

        if((index + stride) < 2 * blockDim.x) 
            atomicAdd(&scan_array[index+stride + CONFLICT_FREE_OFFSET(index+stride)], scan_array[index + CONFLICT_FREE_OFFSET(index)]);

        stride >>= 1;

        __syncthreads();
    }

    if (x < cols) {
        dst(blockIdx.y, x) = scan_array[first + bankOffsetFirst];
        dst(blockIdx.y, x + (blockDim.x/2)) = scan_array[second + bankOffsetSecond];
    }
    
    if ((threadIdx.x == (blockDim.x/2 - 1) && x < cols)) {
        block_sums(blockIdx.y, blockIdx.x) = scan_array[threadIdx.x + blockDim.x/2 + CONFLICT_FREE_OFFSET(threadIdx.x + blockDim.x/2)];
    }
}

__global__ void exclusive_scan_x(const cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst, int n) {
	extern __shared__ float scan_array[];
	
    if (threadIdx.x >= (n/2)) return;
    int first = threadIdx.x;
    int second = threadIdx.x + (n/2);
    int bankOffsetFirst = CONFLICT_FREE_OFFSET(first);
    int bankOffsetSecond = CONFLICT_FREE_OFFSET(second);
    
    scan_array[first + bankOffsetFirst] = src(blockIdx.x, threadIdx.x);
    scan_array[second + bankOffsetSecond] = src(blockIdx.x, threadIdx.x + (n/2));

    int offset = 1;
    for (int d = n>>1; d > 0; d >>= 1) {
        
        __syncthreads();
 
        if (threadIdx.x < d) {
            int ai = offset*(2*threadIdx.x+1)-1;
            int bi = offset*(2*threadIdx.x+2)-1;
            
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi); 
            
            atomicAdd(&scan_array[bi], scan_array[ai]);
        }
        offset <<= 1;
    }
    
    if (threadIdx.x == 0) scan_array[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;

    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        
        __syncthreads();
        
        if (threadIdx.x < d) {
            int ai = offset*(2*threadIdx.x+1)-1;
            int bi = offset*(2*threadIdx.x+2)-1;
            
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            
            float t = scan_array[ai];
            scan_array[ai] = scan_array[bi];
            atomicAdd(&scan_array[bi], t);
        }
    }
    
    __syncthreads(); 

    dst(blockIdx.x, threadIdx.x) = scan_array[first + bankOffsetFirst];
    dst(blockIdx.x, threadIdx.x + (n/2)) = scan_array[second + bankOffsetSecond];  
}

__global__ void add_scans_x(cv::cuda::PtrStepSzf dst, cv::cuda::PtrStepSzf block_sums_scan, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	extern __shared__ float block_offset;
	
	if (threadIdx.x == 0) block_offset = block_sums_scan(blockIdx.y, blockIdx.x);
	
	__syncthreads();
	
	if(x < cols) atomicAdd(&dst(blockIdx.y,x), block_offset);
}

__global__ void inclusive_scan_y(const cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst,  cv::cuda::PtrStepSzf block_sums, int rows) {
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	extern __shared__ float scan_array[];
	
    if (threadIdx.y >= (blockDim.y/2)) return;
    int first = threadIdx.y;
    int second = threadIdx.y + (blockDim.y/2);
    int bankOffsetFirst = CONFLICT_FREE_OFFSET(first);
    int bankOffsetSecond = CONFLICT_FREE_OFFSET(second);
    
    scan_array[first + bankOffsetFirst] = (y < rows) ? src(y, blockIdx.x) : 0;
    scan_array[second + bankOffsetSecond] = ((y + (blockDim.y/2)) < rows) ? src(y + (blockDim.y/2), blockIdx.x) : 0;
	
	__syncthreads();

    int stride = 1;
    while(stride <= blockDim.y) {
        int index = (threadIdx.y + 1) * stride * 2 - 1;

        if(index < 2*blockDim.y) 
            atomicAdd(&scan_array[index + CONFLICT_FREE_OFFSET(index)], scan_array[index-stride + CONFLICT_FREE_OFFSET(index-stride)]);

        stride <<= 1;

        __syncthreads();
    } 

    stride = blockDim.y / 2;
    while(stride > 0) {
        int index = (threadIdx.y + 1) * stride * 2 - 1;

        if((index + stride) < 2 * blockDim.y) 
            atomicAdd(&scan_array[index+stride + CONFLICT_FREE_OFFSET(index+stride)], scan_array[index + CONFLICT_FREE_OFFSET(index)]);

        stride >>= 1;

        __syncthreads();
    }

    if (y < rows) {
        dst(y, blockIdx.x) = scan_array[first + bankOffsetFirst];
        dst(y + (blockDim.y/2), blockIdx.x) = scan_array[second + bankOffsetSecond];
    }
    
    if ((threadIdx.y == (blockDim.y/2 - 1) && y < rows)) {
        block_sums(blockIdx.y, blockIdx.x) = scan_array[threadIdx.y + blockDim.y/2 + CONFLICT_FREE_OFFSET(threadIdx.y + blockDim.y/2)];
    }
}

__global__ void exclusive_scan_y(const cv::cuda::PtrStepSzf src, cv::cuda::PtrStepSzf dst, int n) {
	extern __shared__ float scan_array[];
	
    if (threadIdx.x >= (n/2)) return;
    int first = threadIdx.x;
    int second = threadIdx.x + (n/2);
    int bankOffsetFirst = CONFLICT_FREE_OFFSET(first);
    int bankOffsetSecond = CONFLICT_FREE_OFFSET(second);
    
    scan_array[first + bankOffsetFirst] = src(threadIdx.x, blockIdx.x);
    scan_array[second + bankOffsetSecond] = src(threadIdx.x + (n/2), blockIdx.x);

    int offset = 1;
    for (int d = n>>1; d > 0; d >>= 1) {
        
        __syncthreads();
 
        if (threadIdx.x < d) {
            int ai = offset*(2*threadIdx.x+1)-1;
            int bi = offset*(2*threadIdx.x+2)-1;
            
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi); 
            
            atomicAdd(&scan_array[bi], scan_array[ai]);
        }
        offset <<= 1;
    }
    
    if (threadIdx.x == 0) scan_array[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;

    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        
        __syncthreads();
        
        if (threadIdx.x < d) {
            int ai = offset*(2*threadIdx.x+1)-1;
            int bi = offset*(2*threadIdx.x+2)-1;
            
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            
            float t = scan_array[ai];
            scan_array[ai] = scan_array[bi];
            atomicAdd(&scan_array[bi], t);
        }
    }
    
    __syncthreads(); 

    dst(threadIdx.x, blockIdx.x) = scan_array[first + bankOffsetFirst];
    dst(threadIdx.x + (n/2), blockIdx.x) = scan_array[second + bankOffsetSecond];  
}

__global__ void add_scans_y(cv::cuda::PtrStepSzf dst, cv::cuda::PtrStepSzf block_sums_scan, int rows) {
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	extern __shared__ float block_offset;
	
	if (threadIdx.y == 0) block_offset = block_sums_scan(blockIdx.y, blockIdx.x);
	
	__syncthreads();
	
	if(y < rows) atomicAdd(&dst(y, blockIdx.x), block_offset);
}

__host__ void integral_image_gpu(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst) {
	
	const int m = 256;
	int numRows = src.rows, numCols = src.cols;
	
	const dim3 gridSizeX(ceil((float)numCols / m), numRows, 1);
	const dim3 blockSizeX(m, 1, 1);
	
	const dim3 gridSizeY(numCols, ceil((float)numRows / m), 1);
	const dim3 blockSizeY(1, m, 1);
	
	GpuMat block_sums_x(numRows, (int)ceil((float)numCols / m), CV_32FC1);
	GpuMat block_sums_scan_x(numRows, (int)ceil((float)numCols / m), CV_32FC1);
	
	GpuMat mid_dst(numRows, numCols, CV_32FC1);
	
	GpuMat block_sums_y((int)ceil((float)numRows / m), numCols, CV_32FC1);
	GpuMat block_sums_scan_y((int)ceil((float)numRows / m), numCols, CV_32FC1);

    int a = 1, b = 1;
    while (a < (int)ceil((float)numCols / m)) a <<= 1;
    while (b < (int)ceil((float)numRows / m)) b <<= 1;

    inclusive_scan_x<<<gridSizeX, blockSizeX, sizeof(float)*blockSizeX.x*NUM_BANKS + sizeof(float)*blockSizeX.x>>>(src, mid_dst, block_sums_x, numCols);
    
    exclusive_scan_x<<<numRows, 32, sizeof(float)*a*NUM_BANKS + sizeof(float)*a>>>(block_sums_x, block_sums_scan_x, a);
    
    add_scans_x<<<gridSizeX, blockSizeX, sizeof(float)>>>(mid_dst, block_sums_scan_x, numCols);
    
    
    
    inclusive_scan_y<<<gridSizeY, blockSizeY, sizeof(float)*blockSizeY.y*NUM_BANKS + sizeof(float)*blockSizeY.y>>>(mid_dst, dst, block_sums_y, numRows);
    
    exclusive_scan_y<<<numCols, 32, sizeof(float)*b*NUM_BANKS + sizeof(float)*b>>>(block_sums_y, block_sums_scan_y, b);
    
    add_scans_y<<<gridSizeY, blockSizeY, sizeof(float)>>>(dst, block_sums_scan_y, numRows); 
}