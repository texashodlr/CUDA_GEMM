#include "cudaLib.cuh"

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
	dim3 blockDim(16, 16, 1);
	dim3 gridDim((oShape.width + 15) / 16, (oShape.height + 15) / 16, oShape.channels);

	/*ConvLayer Kernel Call*/
	convLayer_gpu << <gridDim, blockDim >> > (d_in, iShape, d_filter, fShape, d_bias, d_out, oShape, args);
	cudaDeviceSynchronize();

	/*CUDA memcpy for d_out to d_in*/
	cudaMemcpy(h_out, d_out, tensorSize(oShape) * sizeof(float), cudaMemcpyDeviceToHost);


	for (uint32_t i = 0; i < 10;++i) {
		std::cout << "Output at " << i << ": " << h_out[i] << "\n";
	}

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

int runGpuGemm (int argc, char ** argv) {

	evaluateGpuGemm();
	return 0;
}

int evaluateGpuGemm () {

	return 0;
}

//	STUDENT: Add functions here

