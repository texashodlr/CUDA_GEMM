#include "cpuLib.h"

void dbprintf(const char* fmt...) {
	#ifndef DEBUG_PRINT_DISABLE
		va_list args;

		va_start(args, fmt);
		int result = vprintf(fmt, args);
		// printf(fmt, ...);
		va_end(args);
	#endif
	return;
}

void vectorInit(float* v, int size) {
	for (int idx = 0; idx < size; ++idx) {
		v[idx] = (float)(rand() % 100);
	}
}

int verifyVector(float* a, float* b, float* c, float scale, int size) {
	int errorCount = 0;
	for (int idx = 0; idx < size; ++idx) {
		if (c[idx] != scale * a[idx] + b[idx]) {
			++errorCount;
			#ifndef DEBUG_PRINT_DISABLE
				std::cout << "Idx " << idx << " expected " << scale * a[idx] + b[idx] 
					<< " found " << c[idx] << " = " << a[idx] << " + " << b[idx] << "\n";
			#endif
		}
	}
	return errorCount;
}

int verifyVector_convLayer(float* a, float* b, int size) {
	float tolerance = 0.01f;
	int errorCount = 0;
	for (int idx = 0; idx < size; ++idx) {
		float delta = a[idx] - b[idx];
		if (abs(delta) > tolerance) {
			++errorCount;
			std::cout << "Idx " << idx << " expected (CPU):  " << a[idx] << " found (GPU):" << b[idx] << " Delta: " << abs(delta) << "\n";
		}
		if (idx >= 0 && idx < 20) {
			/*Range of IDX Checking*/
			//std::cout << "Idx " << idx << " expected (CPU):  " << a[idx] << " found (GPU):" << b[idx] << "\n";
		}
	}
	return errorCount;
}

int verifyVector_gemm(float* a, float* b, float* c, int size) {
	float tolerance = 0.01f;
	int errorCount = 0;
	for (int idx = 0; idx < size; ++idx) {
		float delta = a[idx] - b[idx];
		float delta2 = a[idx] - c[idx];
		float delta3 = b[idx] - c[idx];
		if (abs(delta) > tolerance || abs(delta2) > tolerance || abs(delta3) > tolerance) {
			++errorCount;
			std::cout << "Idx " << idx << " expected (CPU):  " << a[idx] << " found (GPU):" << b[idx] << " Delta: " << abs(delta) << "\n";
		}
	}
	return errorCount;
}

void printVector(float* v, int size) {
	int MAX_PRINT_ELEMS = 5;
	std::cout << "Printing Vector : \n"; 
	for (int idx = 0; idx < std::min(size, MAX_PRINT_ELEMS); ++idx){
		std::cout << "v[" << idx << "] : " << v[idx] << "\n";
	}
	std::cout << "\n";
}

/**
 * @brief CPU code for SAXPY accumulation Y += A * X
 * 
 * @param x 	vector x
 * @param y 	vector y - will get overwritten with accumulated results
 * @param scale scale factor (A)
 * @param size 
 */
void saxpy_cpu(float* x, float* y, float scale, uint64_t size) {
	for (uint64_t idx = 0; idx < size; ++idx) {
		y[idx] = scale * x[idx] + y[idx];
	}
}

int runCpuSaxpy(uint64_t vectorSize) {
	uint64_t vectorBytes = vectorSize * sizeof(float);

	printf("Hello Saxpy!\n");

	float * a, * b, * c;

	a = (float *) malloc(vectorSize * sizeof(float));
	b = (float *) malloc(vectorSize * sizeof(float));
	c = (float *) malloc(vectorSize * sizeof(float));

	if (a == NULL || b == NULL || c == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}

	vectorInit(a, vectorSize);
	vectorInit(b, vectorSize);
	//	C = B
	std::memcpy(c, b, vectorSize * sizeof(float));
	float scale = 2.0f;

	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" a = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", a[i]);
		}
		printf(" ... }\n");
		printf(" b = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", b[i]);
		}
		printf(" ... }\n");
	#endif

	//	C = A + B
	saxpy_cpu(a, c, scale, vectorSize);

	#ifndef DEBUG_PRINT_DISABLE 
		printf(" c = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", c[i]);
		}
		printf(" ... }\n");
	#endif

	int errorCount = verifyVector(a, b, c, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	return 0;
}

/**
 * @brief CPU-based Monte-Carlo estimation of value of pi
 * 
 * @param iterationCount 	number of iterations of MC evaluation
 * @param sampleSize 		number of random points evaluated in each iteration
 * @return int 
 */
int runCpuMCPi(uint64_t iterationCount, uint64_t sampleSize) {

	std::random_device random_device;
	std::uniform_real_distribution<float> dist(0.0, 1.0);

	float x, y;
	uint64_t hitCount = 0;
	uint64_t totalHitCount = 0;
	std::string str;

	auto tStart = std::chrono::high_resolution_clock::now();

	#ifndef DEBUG_PRINT_DISABLE
		std::cout << "Iteration: ";
	#endif

	for (int iter = 0; iter < iterationCount; ++ iter) {
		hitCount = 0;

		#ifndef DEBUG_PRINT_DISABLE
			str = std::to_string(iter);
			std::cout << str << std::flush;
		#endif

		//	Main CPU Monte-Carlo Code
		for (uint64_t idx = 0; idx < sampleSize; ++idx) {
			x = dist(random_device);
			y = dist(random_device);
			
			if ( int(x * x + y * y) == 0 ) {
				++ hitCount;
			}
		}

		#ifndef DEBUG_PRINT_DISABLE
			std::cout << std::string(str.length(),'\b') << std::flush;
		#endif
		totalHitCount += hitCount;
	}
	#ifndef DEBUG_PRINT_DISABLE
		std::cout << str << std::flush << "\n\n";
	#endif

	//	Calculate Pi
	float approxPi = ((double)totalHitCount / sampleSize) / iterationCount;
	approxPi = approxPi * 4.0f;
		
	std::cout << std::setprecision(10);
	std::cout << "Estimated Pi = " << approxPi << "\n";

		
	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}


std::ostream& operator<< (std::ostream &o,ImageDim imgDim) {
	return (
		o << "Image : " << imgDim.height  << " x " << imgDim.channels << " x "
			<< imgDim.channels << " x " << imgDim.pixelSize << " " 
	);
}

int loadBytesImage(std::string bytesFilePath, ImageDim &imgDim, uint8_t ** imgData ) {
	#ifndef DEBUG_PRINT_DISABLE
		std::cout << "Opening File @ \'" << bytesFilePath << "\' \n";
	#endif

	std::ifstream bytesFile;

	bytesFile.open(bytesFilePath.c_str(), std::ios::in | std::ios::binary);

	if (! bytesFile.is_open()) {
		std::cout << "Unable to open \'" << bytesFilePath << "\' \n";
		return -1;
	}

	ImageDim_t fileDim;
	bytesFile.read((char *) &fileDim, sizeof(fileDim));

	std::cout << "Found " << fileDim.height << " x " << fileDim.width
		<< " x " << fileDim.channels << " x " << fileDim.pixelSize << " \n";
	
	uint64_t numBytes = fileDim.height * fileDim.width * fileDim.channels;
	*imgData = (uint8_t *) malloc(numBytes * sizeof(uint8_t));
	if (imgData == nullptr) {
		std::cout << "Unable to allocate memory for image data \n";
		return -2;
	}

	bytesFile.read((char *) *imgData, numBytes * sizeof(uint8_t));

	std::cout << "Read " << bytesFile.gcount() << " bytes \n" ;

	imgDim.height		= fileDim.height;
	imgDim.width		= fileDim.width;
	imgDim.channels		= fileDim.channels;
	imgDim.pixelSize	= fileDim.pixelSize;

	bytesFile.close();
	
	return bytesFile.gcount();

}

int writeBytesImage (std::string outPath, ImageDim &imgDim, uint8_t * outData) {
	std::ofstream bytesFile;

	bytesFile.open(outPath.c_str(), std::ios::out | std::ios::binary);

	if (! bytesFile.is_open()) {
		std::cout << "Unable to open \'" << outPath << "\' \n";
		return -1;
	}

	uint64_t numBytes = imgDim.height * imgDim.width * imgDim.channels;
	bytesFile.write((char*) &imgDim, sizeof(imgDim));
	bytesFile.write((char *) outData, numBytes * sizeof(uint8_t));

	bytesFile.close();
	return 0;
}

int medianFilter_cpu (uint8_t * inPixels, ImageDim imgDim, uint8_t * outPixels, MedianFilterArgs args) {

	uint32_t startRow = (args.filterH - 1) / 2;
	uint32_t endRow = imgDim.height - ((args.filterH - 1) / 2);
	uint32_t startCol = (args.filterW - 1) / 2;
	uint32_t endCol = imgDim.width - ((args.filterW - 1) / 2);
	uint32_t inRow, inCol;

	std::vector <uint8_t> window;
	window.resize(args.filterH * args.filterW);

	for (uint32_t channel = 0; channel < imgDim.channels; ++ channel) {
		for (uint32_t outRow = startRow; outRow < endRow; ++ outRow) {
			for (uint32_t outCol = startCol; outCol < endCol; ++ outCol) {
				for (uint32_t filRow = 0; filRow < args.filterH; ++ filRow) {
					for (uint32_t filCol = 0; filCol < args.filterW; ++ filCol) {
						inRow = outRow - (args.filterH - 1) / 2 + filRow;
						inCol = outCol - (args.filterW - 1) / 2 + filCol;
						window[filRow * args.filterW + filCol] = inPixels[(inRow * imgDim.width + inCol) * imgDim.channels + channel];
					}
				}

				std::sort(window.begin(), window.end());

				*(outPixels + (outRow * imgDim.width + outCol) * imgDim.channels + channel) = 
					window[(args.filterH * args.filterW) / 2];
			}
		}
		std::cout << "Channel " << channel << " \n";
	}
	return 0;
}

int runCpuMedianFilter (std::string imgPath, std::string outPath, MedianFilterArgs args) {
	ImageDim imgDim;
	uint8_t * imgData;
	
	int bytesRead = loadBytesImage(imgPath, imgDim, &imgData);
	int imgSize = imgDim.height * imgDim.width * imgDim.channels * imgDim.pixelSize;

	std::cout << "Size = " << imgSize << "\n";
	uint8_t * outData = (uint8_t *) malloc(imgSize * sizeof(uint8_t));

	medianFilter_cpu(imgData, imgDim, outData, args);

	writeBytesImage(outPath, imgDim, outData);
	return 0;
}


std::ostream& operator<< (std::ostream &o,PoolOp op) { 
	switch(op) {
	case PoolOp::MaxPool : return o << "MaxPool";
	case PoolOp::AvgPool : return o << "AvgPool";
	case PoolOp::MinPool : return o << "MinPool";
	default: return o<<"(invalid pool op)";
	}
}

int poolLayer_cpu (float * input, TensorShape inShape, 
	float * output, TensorShape outShape, PoolLayerArgs args) {
	float poolPick;
	
	uint32_t poolH = args.poolH;
	uint32_t poolW = args.poolW;
	//	STUDENT: Calculate or unpack TensorShapes
	std::cout << "Lazy, you are! ... ";
	uint32_t outputH = 1;		
	uint32_t outputW = 1;
	uint32_t row, col;

	std::cout << args.opType << " : " << inShape.height << " x " << inShape.width 
		<< " with a " << poolH << " x " << poolW << " window -> " 
		<< outputH << " x " << outputW << "\n";

	for (uint32_t outRow = 0; outRow < outputH; ++ outRow) {
		for (uint32_t outCol = 0; outCol < outputW; ++ outCol) {
			//	STUDENT: Assign to first value of pool area
			// poolPick = 0; 

			for (uint32_t poolRow = 0; poolRow < args.poolH; ++ poolRow) {
				for (uint32_t poolCol = 0; poolCol < args.poolW; ++ poolCol) {
					//	STUDENT: Calculate row and col of element here
					switch (args.opType)
					{
					//	STUDENT: Add cases and complete pooling code for all 3 options
					case PoolOp::MaxPool:
					
					default:
						std::cout << "Pick max from pool, you must!\n";
						return 0;	//	STUDENT: Remove this as reqd.
						break;
					}
				}
			}
			std::cout << poolPick << " @ (" << outRow << ", " << outCol << ")\n";
		}
	}
	return 0;
}

int runCpuPool (TensorShape inShape, PoolLayerArgs poolArgs) {
	
	srand(time(NULL));

	//	STUDENT: Initialize required memories));
	std::cout << "Set Tensors to stun !!";

	//	STUDENT: call pool function
	//	poolLayer_cpu(inMatrix, inShape, outMatrix, outShape, poolArgs);

	return 0;
}



std::ostream& operator << (std::ostream &o, const TensorShape & t) {
	return (
		o << "Tensor : " 
		<< t.count << " x " << t.channels << " x "
		<< t.height << " x " << t.width << " "
	);

}

uint64_t tensorSize (const TensorShape & t) {
	uint64_t size =  ( (uint64_t)t.count * t.channels * t.height * t.width );
	if (size == 0) {
		std::cout << "Invalid shape parameters \n";
	}
	return size;
}


int makeTensor (float ** t, TensorShape & shape) {
	if (*t != nullptr) {
		std::cout << "Pointer already points to memory ! \n";
		return -1;
	}

	if (shape.count == 0) {
		std::cout << " Shape has invalid count (4th dim) - setting to 1 \n";
		shape.count = 1;
	}

	uint64_t tensorSize = shape.height * shape.width * shape.channels * shape.count;
	*t = (float *) malloc (tensorSize * sizeof(float));

	if (*t == nullptr) {
		std::cout << "Malloc failed ! \n";
		return -2;
	}

	float * m = * t;
	uint64_t offset;

	//std::random_device random_device;
	//std::uniform_real_distribution<float> dist(0.0, 1.0);
	std::mt19937 rng(200);
	std::uniform_real_distribution<float> dist(0.0, 1.0);

	//	Implement NCHW layout
	for (uint32_t count = 0; count < shape.count; ++ count) {
		for (uint32_t chIdx = 0; chIdx < shape.channels; ++ chIdx ) {
			for (uint32_t rowIdx = 0; rowIdx < shape.height; ++ rowIdx) {
				for (uint32_t colIdx = 0; colIdx < shape.width; ++ colIdx) {
					offset = count * shape.channels * shape.height * shape.width + chIdx * shape.height * shape.width + rowIdx * shape.width + colIdx;
					m[offset] = dist(rng);
				}
			}
		}
	}
	return 0;
}


int makeTensor_uvm(float** t, TensorShape& shape) {

	float* m = *t;
	uint64_t offset;

	//std::random_device random_device;
	//std::uniform_real_distribution<float> dist(0.0, 1.0);
	std::mt19937 rng(200);
	std::uniform_real_distribution<float> dist(0.0, 1.0);

	//	Implement NCHW layout
	for (uint32_t count = 0; count < shape.count; ++count) {
		for (uint32_t chIdx = 0; chIdx < shape.channels; ++chIdx) {
			for (uint32_t rowIdx = 0; rowIdx < shape.height; ++rowIdx) {
				for (uint32_t colIdx = 0; colIdx < shape.width; ++colIdx) {
					offset = count * shape.channels * shape.height * shape.width + chIdx * shape.height * shape.width + rowIdx * shape.width + colIdx;
					m[offset] = dist(rng);
				}
			}
		}
	}
	return 0;
}


int makeVector (float ** v, uint64_t size) {
	if (*v != nullptr) {
		std::cout << "Pointer already points to memory ! \n";
		return -1;
	}

	*v = (float *) malloc (size * sizeof(float));

	float * m = * v;

	//std::random_device random_device;
	//std::uniform_real_distribution<float> dist(0.0, 1.0);
	std::mt19937 rng(200);
	std::uniform_real_distribution<float> dist(0.0, 1.0);

	//	Implement NCHW layout
	for (uint64_t idx = 0; idx < size; ++ idx) {
		m[idx] = dist(rng);
	}
	return 0;
}


int runCpuConv (int argc, char ** argv) {

	TensorShape iShape = AlexL1_InShape;
	TensorShape fShape = AlexL1_FilterShape;
	TensorShape oShape;
	ConvLayerArgs convArgs = AlexL1_ConvArgs;
	executeCpuConv (iShape, fShape, oShape, convArgs);
	return 0;
}

int executeCpuConv (TensorShape iShape, TensorShape fShape, 
	TensorShape & oShape, ConvLayerArgs args) {

	oShape.height 	= (iShape.height + 2 * args.padH - fShape.height) / args.strideH + 1;
	oShape.width	= (iShape.width  + 2 * args.padW - fShape.width)  / args.strideW + 1;
	oShape.channels	= (fShape.count);
	oShape.count 	= 1;				//	Might scale to batch size

	float * in = nullptr;
	float * filter = nullptr;
	float * bias = nullptr; 
	float * out = nullptr;

	int retVal;
	retVal = makeTensor(&in, iShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n" ;
		return -1;
	}
	retVal = makeTensor(&filter, fShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n" ;
		return -1;
	}
	retVal = makeVector(&bias, oShape.channels);
	if (retVal != 0) {
		std::cout << "Unable to make vector \n" ;
		return -1;
	}

	//std::cout << "OutShape : " << oShape << " \n";
	out = (float *) malloc (tensorSize(oShape) * sizeof(float));
	//std::cout << "Bias[0] = " << bias[0] << "\n";
	
	auto start = std::chrono::high_resolution_clock::now();
	convLayer_cpu(in, iShape, filter, fShape, bias, out, oShape, args);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "CPU execution time: " << elapsed.count() << " seconds\n";


	free(in);
	free(filter);
	free(bias);
	free(out);
	return 0;
}


float* executeCpuConv2(TensorShape iShape, TensorShape fShape,
	TensorShape& oShape, ConvLayerArgs args) {

	oShape.height = (iShape.height + 2 * args.padH - fShape.height) / args.strideH + 1;
	oShape.width = (iShape.width + 2 * args.padW - fShape.width) / args.strideW + 1;
	oShape.channels = (fShape.count);
	oShape.count = 1;				//	Might scale to batch size

	float* in = nullptr;
	float* filter = nullptr;
	float* bias = nullptr;
	float* out = nullptr;

	int retVal;
	retVal = makeTensor(&in, iShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n";
		//return -1;
		exit;
	}
	retVal = makeTensor(&filter, fShape);
	if (retVal != 0) {
		std::cout << "Unable to make tensor \n";
		//return -1;
		exit;
	}
	retVal = makeVector(&bias, oShape.channels);
	if (retVal != 0) {
		std::cout << "Unable to make vector \n";
		exit;
	}

	//std::cout << "OutShape : " << oShape << " \n";
	out = (float*)malloc(tensorSize(oShape) * sizeof(float));

	auto start = std::chrono::high_resolution_clock::now();
	convLayer_cpu(in, iShape, filter, fShape, bias, out, oShape, args);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "CPU execution time: " << elapsed.count() << " seconds\n";

	free(in);
	free(filter);
	free(bias);
	//free(out);
	return out;
}


int convLayer_cpu( float * input, TensorShape iShape, 
	float * filter, TensorShape fShape, 
	float * bias, float * output, TensorShape oShape, 
	ConvLayerArgs & args) {

	//	Convolution in CPU
    // oShape.count denotes the batchsize
	for (uint32_t n = 0; n < oShape.count; ++ n) {
		for (uint32_t m = 0; m < oShape.channels; ++ m) {
			for (uint32_t x = 0; x < oShape.height; ++ x ) {
				for (uint32_t y = 0; y < oShape.width; ++ y) {
					//	For each output fmap value
					//	STUDENT: Set output fmap to bias
					//output[n][m][x][y] = bias[m];
					uint32_t out_idx = ((n * oShape.channels + m) * oShape.height + x) * oShape.width + y;

					output[out_idx] = bias[m];

					for (uint32_t k = 0; k < fShape.channels; ++k) {
						for (uint32_t i = 0; i < fShape.height; ++i) {
							for (uint32_t j = 0; j < fShape.width; ++j) {
								//	STUDENT: Calculate
								//	O[n][m][x][y] += 
								//		I[n][k][args.strideH * x][args.strideW * y] *
								//		W[m][k][i][j];
								//output[n][m][x][y] += input[n][k][args.strideH * x][args.strideW * y] * filter[m][k][i][k];
								uint32_t in_h = args.strideH * x + i;
								uint32_t in_w = args.strideW * y + j;

								if (in_h < iShape.height && in_w < iShape.width) {
									uint32_t in_idx = ((n * iShape.channels + k) * iShape.height + in_h) * iShape.width + in_w;
									uint32_t filter_idx = ((m * fShape.channels + k) * fShape.height + i) * fShape.width + j;
									//uint32_t filter_idx = ((m * fShape.channels + k) * fShape.height + i) * fShape.width + j;
									output[out_idx] += input[in_idx] * filter[filter_idx];
								}
							}
						}
						
					}
					//	STUDENT: Check by disabling activation
					//	STUDENT: Apply Activation here
					if (args.activation) {
						//	O[n][m][x][y] = Activation( O[n][m][x][y] );
						output[out_idx] = fmaxf(0.0f, output[out_idx]);
					}
					//	
				}
			}
		}
	}
	return 0;
}

void matMulNaiveCpu (int N, int *a, int *b, int *c) {
    int i, j, k;
    for (j = 0; j < N; ++ j) {
        for (i=0; i<N; i++) {
            c[i*N + j] = 0;
        }
    }
    
    for (k=0; k<N; k++) {
        for (j=0; j<N; j++) {
            for (i=0; i<N; i++) {
                c[i*N + j] += a[i*N + k] * b[k*N + j];
            }
        }
    }
}



int runCpuGemm (int argc, char ** argv) {

	uint32_t BatchSize = 3;
	TensorShape aShape = { BatchSize, 1, 1, 4096 };
	TensorShape bShape = { 1, 1, 4096, 4096 };
	TensorShape cShape;
	GemmLayerArgs args = { 2, 2, 1 };
	std::cout << "Executing GEMM with BatchSize: " << BatchSize << "\n";
	executeCpuGemm_v2(aShape, bShape, cShape, args);
	return 0;
}
int runCpuGemm1(int argc, char** argv) {

	TensorShape aShape = { 1, 1, 6, 4 };
	TensorShape bShape = { 1, 1, 4, 8 };
	TensorShape cShape;
	GemmLayerArgs args = { 2, 2, 1 };

	executeCpuGemm(aShape, bShape, cShape, args);
	return 0;
}


int executeCpuGemm_v1(TensorShape aShape, TensorShape bShape, 
	TensorShape & cShape, GemmLayerArgs args, uint32_t BatchSize) {

	if (aShape.width != bShape.height || aShape.channels != bShape.channels 
		|| aShape.count != bShape.count) {
		//std::cout << "Dimensions dont match : " << aShape << " x " << bShape << " \n";
		//return -1;
	}

	cShape.height = aShape.height;
	cShape.width = bShape.width;
	cShape.channels = aShape.channels;
	cShape.count = BatchSize;

	float * a = nullptr;
	float * b = nullptr;

	makeTensor(& a, aShape);
	makeTensor(& b, bShape);

	float * c = (float *) malloc(tensorSize(cShape) * sizeof(float));

	auto start = std::chrono::high_resolution_clock::now();
	gemmLayer_cpu(a, aShape, b, bShape, c, cShape, args);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "CPU execution time: " << elapsed.count() << " seconds\n";

	return 0;
}


int executeCpuGemm(TensorShape aShape, TensorShape bShape,
	TensorShape& cShape, GemmLayerArgs args) {

	if (aShape.width != bShape.height || aShape.channels != bShape.channels
		|| aShape.count != bShape.count) {
		std::cout << "Dimensions dont match : " << aShape << " x " << bShape << " \n";
		return -1;
	}

	cShape.height = aShape.height;
	cShape.width = bShape.width;
	cShape.channels = aShape.channels;
	cShape.count = aShape.count;

	float* a = nullptr;
	float* b = nullptr;

	makeTensor(&a, aShape);
	makeTensor(&b, bShape);

	float* c = (float*)malloc(tensorSize(cShape) * sizeof(float));

	auto start = std::chrono::high_resolution_clock::now();
	gemmLayer_cpu(a, aShape, b, bShape, c, cShape, args);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "CPU execution time: " << elapsed.count() << " seconds\n";


	return 0;
}


int gemmLayer_cpu (float * a, TensorShape aShape,
	float * b, TensorShape bShape,
	float * c, TensorShape & cShape,
	GemmLayerArgs & args) {
	

    int tilesAlongW = (cShape.width + args.tileW - 1) / args.tileW;
    int tilesAlongH = (cShape.height + args.tileH - 1) / args.tileH;
    int subTilesAlongK = (aShape.width + args.tileH - 1) / args.tileH;

    int tileId = 0;


	while (tileId < tilesAlongW * tilesAlongH) {
        int offsetH = (tileId / tilesAlongW) * args.tileH;
        int offsetW = (tileId % tilesAlongW) * args.tileW;
        int rowIdx, colIdx;
        int row, col, subTile, subTileK, k;

        #ifdef PRINT_DEBUG
        printf("%d thread @ tile # %d\n", threadIdx, tileId);
        #endif

        for (subTile = 0; subTile < subTilesAlongK; ++ subTile) {
            for (row = 0; row < args.tileH; ++ row) {
                for (col = 0; col < args.tileW; ++ col) {
                    #ifdef PRINT_DEBUG
                    printf("%d @ (%03d, %03d)  = %d\n", threadIdx, 
                        row + offsetH, col + offsetW, IDX2R(row + offsetH, col + offsetW, TILE_W));
                    #endif
                    rowIdx = row + offsetH;
                    colIdx = col + offsetW;

                    //  Check bounds of actual output matrix
                    if (rowIdx < cShape.height && colIdx < cShape.width) {
						if (subTile == 0)
                        	c[IDX2R(rowIdx, colIdx, cShape.width)] = 0;
                        for (subTileK = 0; subTileK < args.tileH; ++ subTileK) {
                            k = subTile * args.tileH + subTileK;
                            if (k < aShape.width) {
                                c[IDX2R(rowIdx, colIdx, cShape.width)] += a[IDX2R(rowIdx, k, aShape.width)] * b[IDX2R(k, colIdx, bShape.width)];
                            }
                        }
                    }
                }
            }
		}
		++ tileId;
    }
	return 0;
}

float* executeCpuGemm_v3(TensorShape aShape, TensorShape bShape,
	TensorShape& cShape, GemmLayerArgs args) {

	cShape.height = aShape.height;
	cShape.width = bShape.width;
	cShape.channels = aShape.channels;
	cShape.count = aShape.count;

	float* a = nullptr;
	float* b = nullptr;

	makeTensor(&a, aShape);
	makeTensor(&b, bShape);

	float* c = (float*)malloc(tensorSize(cShape) * sizeof(float));

	auto start = std::chrono::high_resolution_clock::now();
	gemmLayer_cpu_batchsize(a, aShape, b, bShape, c, cShape, args);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "CPU execution time: " << elapsed.count() << " seconds\n";

	free(a);
	free(b);

	return c;


}

int executeCpuGemm_v2(TensorShape aShape, TensorShape bShape,
	TensorShape& cShape, GemmLayerArgs args) {

	cShape.height = aShape.height;
	cShape.width = bShape.width;
	cShape.channels = aShape.channels;
	cShape.count = aShape.count;

	float* a = nullptr;
	float* b = nullptr;

	makeTensor(&a, aShape);
	makeTensor(&b, bShape);

	float* c = (float*)malloc(tensorSize(cShape) * sizeof(float));

	auto start = std::chrono::high_resolution_clock::now();
	gemmLayer_cpu_batchsize(a, aShape, b, bShape, c, cShape, args);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "CPU execution time: " << elapsed.count() << " seconds\n";

	free(a);
	free(b);
	free(c);

	return 0;

	
}

int gemmLayer_cpu_batchsize(float* a, TensorShape aShape,
	float* b, TensorShape bShape,
	float* c, TensorShape& cShape,
	GemmLayerArgs& args) {

	int tilesAlongW = (cShape.width + args.tileW - 1) / args.tileW;
	int tilesAlongH = (cShape.height + args.tileH - 1) / args.tileH;
	int subTilesAlongK = (aShape.width + args.tileH - 1) / args.tileH;

	// Loop over batches
	for (uint32_t batch = 0; batch < cShape.count; ++batch) {
		int batchOffsetA = batch * aShape.channels * aShape.height * aShape.width;
		int batchOffsetB = (bShape.count == 1) ? 0 : batch * bShape.channels * bShape.height * bShape.width;
		int batchOffsetC = batch * cShape.channels * cShape.height * cShape.width;

		int tileId = 0;
		while (tileId < tilesAlongW * tilesAlongH) {
			int offsetH = (tileId / tilesAlongW) * args.tileH;
			int offsetW = (tileId % tilesAlongW) * args.tileW;
			int rowIdx, colIdx;

			for (int subTile = 0; subTile < subTilesAlongK; ++subTile) {
				for (int row = 0; row < args.tileH; ++row) {
					for (int col = 0; col < args.tileW; ++col) {
						rowIdx = row + offsetH;
						colIdx = col + offsetW;

						if (rowIdx < cShape.height && colIdx < cShape.width) {
							if (subTile == 0)
								c[batchOffsetC + IDX2R(rowIdx, colIdx, cShape.width)] = 0;

							for (int subTileK = 0; subTileK < args.tileH; ++subTileK) {
								int k = subTile * args.tileH + subTileK;
								if (k < aShape.width) {
									c[batchOffsetC + IDX2R(rowIdx, colIdx, cShape.width)] +=
										a[batchOffsetA + IDX2R(rowIdx, k, aShape.width)] *
										b[batchOffsetB + IDX2R(k, colIdx, bShape.width)];
								}
							}
						}
					}
				}
			}
			++tileId;
		}
	}
	return 0;
}