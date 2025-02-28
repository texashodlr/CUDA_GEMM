#include "cudaLib.cuh"
#define TILE_SIZE 6

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	std::cout << "Lazy, you are!\n";
	std::cout << "Write code, you must\n";

	return 0;
}

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 3.14159f;

	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}




int runGpuMedianFilter (std::string imgPath, std::string outPath, MedianFilterArgs args) {
	
	std::cout << "Lazy, you are! ... ";
	std::cout << "Filter pixels, you must! ... ";

	return 0;
}

int medianFilter_gpu (uint8_t inPixels, ImageDim imgDim, 
	uint8_t outPixels, MedianFilterArgs args) {

	return 0;
}


int runGpuConv (int argc, char ** argv) {

	TensorShape iShape = AlexL1_InShape;
	TensorShape fShape = AlexL1_FilterShape;
	ConvLayerArgs convArgs = AlexL1_ConvArgs;

	std::cout << "Evaluate convolution : \n";
	std::cout << "Input : " << iShape << " \n";
	std::cout << "Filter : " << fShape << " \n";

	TensorShape oShape;

	executeGpuConv(iShape, fShape, oShape, convArgs);
	executeCpuConv(iShape, fShape, oShape, convArgs);
	//uint64_t errorCount = evaluateGpuConv(iShape, fShape, oShape, convArgs);
	//std::cout << "Found " << errorCount << " / " << tensorSize(oShape) << " errors \n";
	return 0;
}

int executeGpuConv(TensorShape iShape, TensorShape fShape,
	TensorShape& oShape, ConvLayerArgs args) {

	oShape.height = (iShape.height + 2 * args.padH - fShape.height) / args.strideH + 1;
	oShape.width = (iShape.width + 2 * args.padW - fShape.width) / args.strideW + 1;
	oShape.channels = (fShape.count);
	oShape.count = 1;

	float* h_in = nullptr;
	float* h_filter = nullptr;
	float* h_bias = nullptr;
	float* h_out = nullptr;

	int retVal;
	retVal = makeTensor(&h_in, iShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n";
		return -1;
	}
	retVal = makeTensor(&h_filter, fShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n";
		return -1;
	}
	retVal = makeVector(&h_bias, oShape.channels);
	if (retVal != 0) {
		std::cout << "Unable to make vector \n";
		return -1;
	}

	std::cout << "OutShape : " << oShape << " \n";
	h_out = (float*)malloc(tensorSize(oShape) * sizeof(float));

	/*CUDA Malloc for in, out, filter and bias*/

	float* d_in, * d_filter, * d_bias, * d_out;
	cudaMalloc(&d_in, tensorSize(iShape) * sizeof(float));
	cudaMalloc(&d_filter, tensorSize(fShape) * sizeof(float));
	cudaMalloc(&d_bias, (oShape.channels) * sizeof(float));
	cudaMalloc(&d_out, tensorSize(oShape) * sizeof(float));

	/*CUDA Memcpy for in, filter and bias*/
	cudaMemcpy(d_in, h_in, tensorSize(iShape) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_filter, h_filter, tensorSize(fShape) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bias, h_bias, (oShape.channels) * sizeof(float), cudaMemcpyHostToDevice);

	/*Block and Grid Dims*/
	dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
	dim3 gridDim((oShape.width + TILE_SIZE-1) / TILE_SIZE, (oShape.height + TILE_SIZE-1) / TILE_SIZE, oShape.channels);

	int windowSize = iShape.channels*(TILE_SIZE + fShape.height - 1) * (TILE_SIZE + fShape.width - 1);
	size_t sharedMemSize = windowSize * sizeof(float); //676*4 = 3*2704 bytes -> 8.1KB 

	/*ConvLayer Kernel Call*/
	//convLayer_gpu << <gridDim, blockDim >> > (d_in, iShape, d_filter, fShape, d_bias, d_out, oShape, args);

	for (uint32_t i = 100; i < 111 ;++i) {
		std::cout << "Input at " << i << ": " << h_in[i] << "\n";
	}
	std::cout << "\n\n GPU Starting!\n\n\n";
	std::cout << "Bias[0] = " << h_bias[0] << "\n";
	convLayer_gpu_SM_DM_v3 << <gridDim, blockDim, sharedMemSize >> > (d_in, iShape, d_filter, fShape, d_bias, d_out, oShape, args);
	cudaDeviceSynchronize();

	/*CUDA memcpy for d_out to d_in*/
	cudaMemcpy(h_out, d_out, tensorSize(oShape) * sizeof(float), cudaMemcpyDeviceToHost);


	for (uint32_t i = 0; i < 10;++i) {
		std::cout << "Output at " << i << ": " << h_out[i] << "\n";
	}
	std::cout << "CPU NEXT \n";

	/* cudaFree() functions */
	cudaFree(d_in);
	cudaFree(d_filter);
	cudaFree(d_bias);
	cudaFree(d_out);

	/*CPU Free*/
	free(h_in);
	free(h_filter);
	free(h_bias);
	free(h_out);
	return 0;
}

uint64_t evaluateGpuConv (TensorShape iShape, TensorShape fShape, 
	TensorShape & oShape, ConvLayerArgs args) {

	uint64_t errorCount = 0;

	//	STUDENT: Add code here

	#ifndef CONV_CHECK_DISABLE
		//	STUDENT: Verify number of errors in ouput matrix generated by convLayer_gpu
		//	STUDENT: Compare results with CPU output
		//	STUDENT: Return error count


	#endif

	return errorCount;
}

__global__ void convLayer_gpu(float* input, TensorShape iShape, float* filter,
	TensorShape fShape, float* bias, float* output, TensorShape oShape, ConvLayerArgs args) {

	/*Coordinates*/
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int channels = blockIdx.z;

	if (col < oShape.width && row < oShape.height) {
		uint32_t out_idx = ((channels * oShape.height + row) * oShape.width + col);
		output[out_idx] = bias[channels];
		//CPU Code For-loop repeat for the filter window//
		for (uint32_t i = 0; i < fShape.height; ++i) {
			for (uint32_t j = 0; j < fShape.width; ++j) {
				for (uint32_t k = 0; k < fShape.channels; ++k) {
					uint32_t in_h = args.strideH * row + i;
					uint32_t in_w = args.strideW * col + j;

					if (in_h < iShape.height && in_w < iShape.width) {
						uint32_t in_idx = (k * iShape.height + in_h) * iShape.width + in_w;
						uint32_t filter_idx = ((channels * fShape.channels + k) * fShape.height + i) * fShape.width + j;

						output[out_idx] += input[in_idx] * filter[filter_idx];
					}
				}
			}
		}
		if (args.activation) {
			output[out_idx] = fmaxf(0.0f, output[out_idx]);
		}
	}
}

__global__ void convLayer_gpu_SM_DM(float* input, TensorShape iShape, float* filter,
	TensorShape fShape, float* bias, float* output, TensorShape oShape, ConvLayerArgs args) {

	extern __shared__ float tile[]; // 26x26 for 2.7KB

	/*Local TB Coords*/
	int shared_x = threadIdx.x;
	int shared_y = threadIdx.y;

	/*Sizing*/
	int shared_dim = TILE_SIZE + fShape.height - 1;

	/*Using my local TB Coords I'm orienting myself globally to then load into shMem*/

	/*Input Coords to load into tile[]*/
	int input_x = blockIdx.x * blockDim.x * args.strideW + shared_x - (fShape.width / 2);
	int input_y = blockIdx.y * blockDim.y * args.strideH + shared_y - (fShape.height / 2);

	//printf("Input_x: %d | Input_y: %d\n", input_x, input_y);

	if ((input_x >= 0 && input_x < iShape.width) && (input_y >= 0 && input_y < iShape.height)) {
		for (uint32_t k = 0; k < iShape.channels;++k) {
			tile[shared_y * shared_dim + shared_x] = input[((k*iShape.height+input_y) * iShape.width) + input_x];
		}		
	}
	else {
		tile[shared_y * shared_dim + shared_x] = 0.0f;
		//tile[shared_y * shared_dim + shared_x] = input[(0 * iShape.height + input_y) * iShape.width + input_x];

		//printf("OOB Baby!");
	}

	__syncthreads();

	/*Coordinates*/
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int channels = blockIdx.z;

	if (col < oShape.width && row < oShape.height) {
		/*Output Matrix Index*/
		uint32_t out_idx = ((channels * oShape.height + row) * oShape.width + col);
		float shared_sum = bias[channels];

		//CPU Code For-loop repeat for the filter window//
		for (uint32_t i = 0; i < fShape.height; ++i) {
			for (uint32_t j = 0; j < fShape.width; ++j) {
				for (uint32_t k = 0; k < fShape.channels; ++k) {

					/*Index+offset of the shMem location*/
					uint32_t sharedMem_x = shared_x + j;
					uint32_t sharedMem_y = shared_y + i;

					/*Filter Index*/
					uint32_t filter_idx = ((channels * fShape.channels + k) * fShape.height + i) * fShape.width + j;
					
					if (sharedMem_x < shared_dim && sharedMem_y < shared_dim) {
						shared_sum += tile[sharedMem_y * shared_dim + sharedMem_x] * filter[filter_idx];
					}
				}
			}
		}
		if (args.activation) {
			output[out_idx] = fmaxf(0.0f, shared_sum);
		}
		else {
			output[out_idx] = shared_sum;
		}
	}
}

__global__ void convLayer_gpu_SM_DM_v2(float* input, TensorShape iShape, float* filter,
	TensorShape fShape, float* bias, float* output, TensorShape oShape, ConvLayerArgs args) {

	extern __shared__ float tile[]; // 26x26 for 2.7KB
	const int tile_height = TILE_SIZE + fShape.height - 1; //26
	const int tile_width = TILE_SIZE + fShape.width - 1; //26
	
	/*Local TB Coords*/
	int shared_x = threadIdx.x;
	int shared_y = threadIdx.y;

	/*Output Coods*/
	int out_x = blockIdx.x * TILE_SIZE + shared_x;
	int out_y = blockIdx.y * TILE_SIZE + shared_y;
	int out_z = blockIdx.z; //0-95 for AlexNet (96)

	for (uint32_t k = 0; k < iShape.channels;++k) {
		int input_x = blockIdx.x * TILE_SIZE * args.strideW + shared_x;
		int input_y = blockIdx.y * TILE_SIZE * args.strideH + shared_y;
		int shared_idx = k * tile_height * tile_width + shared_y * tile_width + shared_x;
	
		if ((input_x >= 0 && input_x < iShape.width) && (input_y >= 0 && input_y < iShape.height)) {
			int in_idx = (k * iShape.height + input_y) * iShape.width + input_x;
			tile[shared_idx] = input[in_idx];
		}
		else {
			tile[shared_idx] = 0.0f;
		}
	}

	__syncthreads();

	/*Coordinates*/
	if (out_x < oShape.width && out_y < oShape.height) {
		/*Output Matrix Index*/
		uint32_t out_idx = ((out_z * oShape.height + out_y) * oShape.width + out_x);
		float shared_sum = bias[out_z];

		//CPU Code For-loop repeat for the filter window//
		for (uint32_t k = 0; k < fShape.channels; ++k) {
			float channel_sum = 0.0f;
			for (uint32_t i = 0; i < fShape.height; ++i) {
				for (uint32_t j = 0; j < fShape.width; ++j) {
					int in_x = out_x * args.strideW + j;  // Absolute input coord
					int in_y = out_y * args.strideH + i;
					int tile_x = in_x - (blockIdx.x * TILE_SIZE * args.strideW);  // Offset within tile
					int tile_y = in_y - (blockIdx.y * TILE_SIZE * args.strideH);

					if (in_x < iShape.width && in_y < iShape.height && tile_x>= 0 && tile_x < tile_width && tile_y >= 0 && tile_y < tile_height) {
						uint32_t shared_idx = k * tile_height * tile_width + tile_y * tile_width + tile_x;
						uint32_t filter_idx = (out_z * fShape.channels + k) * fShape.height * fShape.width + i * fShape.width + j;
						channel_sum += tile[shared_idx] * filter[filter_idx];
					}
					if (out_x == 2 && out_y == 0 && out_z == 0 && k == 0 && i == 0 && j == 0) {
						printf("tile_x = %d, tile_y = %d\n", tile_x, tile_y);
					}
				}
			}
			shared_sum += channel_sum;
			if (out_x == 2 && out_y == 0 && out_z == 0) {
				printf("Channel %d contribution: %f, Running sum: %f\n", k, channel_sum, shared_sum);
			}
		}
		if (args.activation) {
			output[out_idx] = fmaxf(0.0f, shared_sum);
		}
		else {
			output[out_idx] = shared_sum;
		}
	}
}

__global__ void convLayer_gpu_SM_DM_v3(float* input, TensorShape iShape, float* filter,
	TensorShape fShape, float* bias, float* output, TensorShape oShape, ConvLayerArgs args) {

	extern __shared__ float tile[];
	const int tile_height = TILE_SIZE + fShape.height - 1; // 26
	const int tile_width = TILE_SIZE + fShape.width - 1;   // 26

	int shared_x = threadIdx.x;
	int shared_y = threadIdx.y;
	int out_x = blockIdx.x * TILE_SIZE + shared_x;
	int out_y = blockIdx.y * TILE_SIZE + shared_y;
	int out_z = blockIdx.z;
	int base_x = blockIdx.x * TILE_SIZE * args.strideW;
	int base_y = blockIdx.y * TILE_SIZE * args.strideH;

	// Load full 26x26 tile cooperatively
	int thread_id = shared_y * TILE_SIZE + shared_x;  // 0 to 255
	int elements_per_thread = (tile_height * tile_width + TILE_SIZE * TILE_SIZE - 1) / (TILE_SIZE * TILE_SIZE);  // 3
	for (uint32_t k = 0; k < iShape.channels; ++k) {
		for (int n = 0; n < elements_per_thread; ++n) {
			int flat_idx = thread_id + n * (TILE_SIZE * TILE_SIZE);  // Step by 256
			if (flat_idx < tile_height * tile_width) {
				int y = flat_idx / tile_width;  // 0 to 25
				int x = flat_idx % tile_width;  // 0 to 25
				int input_x = base_x + x;
				int input_y = base_y + y;
				int shared_idx = k * tile_height * tile_width + y * tile_width + x;

				if (input_x >= 0 && input_x < iShape.width && input_y >= 0 && input_y < iShape.height) {
					int in_idx = (k * iShape.height + input_y) * iShape.width + input_x;
					tile[shared_idx] = input[in_idx];
				}
				else {
					tile[shared_idx] = 0.0f;
				}
			}
		}
	}
	__syncthreads();

	if (out_x == 0 && out_y == 0 && out_z == 0) {
		for (int z = 0; z < (3 * tile_height * tile_width); ++z) {
			//printf("Index: %d | Value: %f\n", z, tile[z]);
		}
	}
	// Compute output
	if (out_x < oShape.width && out_y < oShape.height) {
		uint32_t out_idx = (out_z * oShape.height + out_y) * oShape.width + out_x;
		float shared_sum = bias[out_z];

		for (uint32_t k = 0; k < fShape.channels; ++k) {
			float channel_sum = 0.0f;
			for (uint32_t i = 0; i < fShape.height; ++i) {
				for (uint32_t j = 0; j < fShape.width; ++j) {
					int in_x = (out_x)*args.strideW + j;
					int in_y = out_y * args.strideH + i;
					int tile_x = in_x - base_x;
					int tile_y = in_y - base_y;
					/*
					if ((out_x == 2) && out_y == 0 && out_z == 0) {
						printf("Bounds: in_x = %d | base_x = %d | in_y = %d | base_y = %d  && iS.w = %u | iS.h = %u ### tile_x = %d | tile_y = %d && tile_width = %d | tile_height = %d\n Filter_H: %u | Filter_W: %u\n ",
							in_x, base_x, in_y, base_y, iShape.width, iShape.height, tile_x, tile_y, (in_x + tile_width), (in_y + tile_height), i, j);
					}
					*/

					if (in_x < iShape.width && in_y < iShape.height) {
						uint32_t shared_idx = k * tile_height * tile_width + tile_y * tile_width + tile_x;
						uint32_t filter_idx = (out_z * fShape.channels + k) * fShape.height * fShape.width + i * fShape.width + j;
						channel_sum += tile[shared_idx] * filter[filter_idx];
						if (out_x == 3 && out_y == 0 && out_z == 0) {
							printf("Channel %d contribution: %f, Running sum: %f | Filter_H: %u | Filter_W: %u | Shared_IDX: %u | Filter_idx: %u\n",
								k, channel_sum, shared_sum, i, j, shared_idx, filter_idx);
							printf("\tBounds: in_x = %d | base_x = %d | in_y = %d | base_y = %d  && iS.w = %u | iS.h = %u ### tile_x = %d | tile_y = %d && tile_width = %d | tile_height = %d\n Filter_H: %u | Filter_W: %u\n ",
								in_x, base_x, in_y, base_y, iShape.width, iShape.height, tile_x, tile_y, (in_x + tile_width), (in_y + tile_height), i, j);
						}
					}
				}
			}
			shared_sum += channel_sum;
		}
		output[out_idx] = shared_sum;
	}
}

int runGpuGemm (int argc, char ** argv) {

	evaluateGpuGemm();
	return 0;
}

int evaluateGpuGemm () {

	return 0;
}

//	STUDENT: Add functions here
