#include "CriticalPolyphaseFilterbank.h"


/*
* VS does complain about not knowing __ldg as a function, since older architectures do not support that.
* However, even when setting the compute capability high VS does complain so this is a workaround.
*/
template<typename T>
__device__ __forceinline__ T ldg(const T* ptr) {
#if __CUDA_ARCH__ >= 350
	return __ldg(ptr);
#else
	return *ptr;
#endif
}


static const size_t COEFF_SIZE = 512;
static const size_t DATA_SIZE = 1792;
static const size_t THREADS_PER_BLOCK = 672;
static const size_t SUBBLOCK_SIZE = 3;
static const size_t THXPERWARP = 32;



/* The GPU method performs a FIR Filter.
It is taken from https://github.com/AstroAccelerateOrg/astro-accelerate/blob/master/lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu
Which is part of the publication https://dl.acm.org/citation.cfm?id=2286986
It was adjusted to process floats and not float2 and some of the index calculations were outsourced.
ldg is still the same function as __ldg however since __ldg will not work with old compute capabilities there is a check.
*/
__global__ void CPF_Fir_shared_32bit(const float * __restrict__ d_data, float* d_spectra, const float * __restrict__ d_coeff, unsigned int fft_size, unsigned int nTaps, unsigned int nSpectra) {

	float ftemp;
	int memblock, localId, s_mempos, g_mempos, num_spectra, start_column, warpId, itemp;
	int tx = threadIdx.x;

	__shared__ float s_data[DATA_SIZE];
	__shared__ float s_coeff[COEFF_SIZE];

	warpId = ((int)tx / THXPERWARP);
	memblock = warpId * SUBBLOCK_SIZE;
	localId = tx - ((int)tx / THXPERWARP) * THXPERWARP;		// Calculates threads Id within a WARP
	num_spectra = (DATA_SIZE / THXPERWARP - nTaps + 1);

	// read input data from global memory a store them into shared memory
	int constantnumber = blockIdx.x * THXPERWARP + blockIdx.y * num_spectra * fft_size + localId;
	for (int i = 0; i < SUBBLOCK_SIZE; i++) {
		start_column = memblock + i;
		if (start_column < DATA_SIZE / THXPERWARP) {
			s_mempos = start_column * THXPERWARP + localId;
			g_mempos = start_column * fft_size + constantnumber;
			// TODO: we need ldg? NVProf NVVP
			s_data[s_mempos] = ldg(&d_data[g_mempos]);
		}
	}
	itemp = (int)(nTaps / (THREADS_PER_BLOCK / THXPERWARP)) + 1;
	for (int f = 0; f < itemp; f++) {
		start_column = warpId + f * (THREADS_PER_BLOCK / THXPERWARP);
		if (start_column < nTaps) {
			s_coeff[start_column*THXPERWARP + localId] = ldg(&d_coeff[start_column*fft_size + blockIdx.x*THXPERWARP + localId]);
		}
	}

	__syncthreads();

	// Calculation of the FIR part
	for (int i = 0; i < SUBBLOCK_SIZE; i++) { // WARP loops through columns in it's sub-block
		start_column = memblock + i;
		if (start_column < num_spectra) {
			s_mempos = start_column * THXPERWARP + localId;
			ftemp = 0.0f;
			for (int j = 0; j < nTaps ; j++) {
				ftemp += s_coeff[j*THXPERWARP + localId] * (s_data[s_mempos + j * THXPERWARP]);
			}
			// TODO: Check NVVP Bank conflicts in SM.
			if (start_column * fft_size + constantnumber < fft_size * nSpectra) {
				d_spectra[start_column * fft_size + constantnumber] = ftemp;
			}
		}
	}
}

/**
* Constructor of the Critical Polyphase Filterbank class.
* @param nchans					Number of Channels to be processed
* @param ntaps					Number of Taps to be processed
* @param filter_coefficients		Vector of filter coefficients used for the FIR-Filter
* @param cudaStream				Stream where the Program is supposed to run on
*
*/
CriticalPolyphaseFilterbank::CriticalPolyphaseFilterbank(std::size_t fftSize, std::size_t nTaps, std::size_t nSpectra, FilterCoefficientsType const & filterCoefficients, cudaStream_t stream) : fftSize(fftSize), nTaps(nTaps), nSpectra(nSpectra), filterCoefficients(filterCoefficients), stream(stream)
{
	// Create the cufft Plan.
  std::cerr << "FFTSize: " << fftSize << std::endl;
  std::cerr << "nSpectra: " << nSpectra<< std::endl;
	cufftResult error = cufftPlan1d(&plan, fftSize, CUFFT_R2C, nSpectra);
	cufftSetStream(plan, stream);

	if (CUFFT_SUCCESS != error) {
		printf("CUFFT error: %d\n", error);
	}

	firOutput.resize(nSpectra * fftSize);
}

/**
* Destructor of Critical Polyphase Filterbank.
*/
CriticalPolyphaseFilterbank::~CriticalPolyphaseFilterbank()
{
	cufftDestroy(plan);
}

/**
* @brief Apply the polyphase filter to a block of timeseries data
*
* @param input			The timeseries to process
* @param output			The channelised filtered output data
*
* @tparam InputType		The input data type
* @tparam OutputType		The output data type
*/
void CriticalPolyphaseFilterbank::process(const thrust::device_vector<float> &input, thrust::device_vector <cufftComplex> &output)
{
	// This should be in the default settings but just to make sure.
	//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

	//---------> CUDA block and CUDA grid parameters
  static const size_t SM_Columns = (DATA_SIZE / THXPERWARP - nTaps + 1);
	int nCUDAblocks_y = (int) ceil((float)nSpectra / SM_Columns);
	int nCUDAblocks_x = (int)(fftSize / THXPERWARP);

	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);	//nCUDAblocks_y goes through spectra
	dim3 blockSize(THREADS_PER_BLOCK, 1, 1); 		    //nCUDAblocks_x goes through fftSize

	//---------> FIR filter part
	const float* inputptr = thrust::raw_pointer_cast(&input[0]);
	float* outputptr = thrust::raw_pointer_cast(&firOutput[0]);
	const float* coeffptr = thrust::raw_pointer_cast(&filterCoefficients[0]);

	/* You might be confused about the kernel call, because it uses 0 bytes shared Memory.
	* But actually it uses 0 bytes dynamically allocated shared memory, the SM is statically allocated in the kernel.
	* We list it to set the stream.*/
	CPF_Fir_shared_32bit << <gridSize, blockSize, 0, stream >>> (inputptr, outputptr, coeffptr, fftSize, nTaps, nSpectra);

	cufftResult error = cufftExecR2C(plan, (cufftReal *) thrust::raw_pointer_cast(&firOutput[0]), (cufftComplex *)thrust::raw_pointer_cast(&output[0]) );
	if (CUFFT_SUCCESS != error) {
		printf("CUFFT error: %d\n", error);
	}

  cudaDeviceSynchronize();
}
