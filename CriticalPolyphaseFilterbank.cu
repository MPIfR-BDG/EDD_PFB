#include "CriticalPolyphaseFilterbank.h"
#include <vector>
#include <stddef.h>
#include <time.h>
#include "debug.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "timer.cuh"
#include "params.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "utils_cuda.h"
#include <iostream>
#include <fstream>
#include <iomanip> 
#include <string>
#include <typeinfo>
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


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

/* The GPU method performs a FIR Filter. 
It is taken from https://github.com/AstroAccelerateOrg/astro-accelerate/blob/master/lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu 
Which is part of the publication https://dl.acm.org/citation.cfm?id=2286986
It was adjusted to process floats and not float2 and some of the index calculations were outsourced.
ldg is still the same function as __ldg however since __ldg will not work with old compute capabilities there is a check.
*/
__global__ void CPF_Fir_shared_32bit(float const* __restrict__ d_data, float* d_spectra, float const* __restrict__ d_coeff, int nChannels) {

	float ftemp;
	int memblock, localId, s_mempos, g_mempos, num_spectra, start_column, warpId, itemp;
	int tx = threadIdx.x;

	__shared__ float s_data[DATA_SIZE*2];
	__shared__ float s_coeff[COEFF_SIZE];

	warpId = ((int)tx / THXPERWARP);
	memblock = warpId * SUBBLOCK_SIZE;
	localId = tx - ((int)tx / THXPERWARP) * THXPERWARP;		// Calculates threads Id within a WARP
	num_spectra = (DATA_SIZE / THXPERWARP - TAPS + 1);

	// read input data from global memory a store them into shared memory
	int constantnumber = blockIdx.x * THXPERWARP + blockIdx.y * num_spectra * nChannels + localId;
	for (int i = 0; i < SUBBLOCK_SIZE; i++) {
		start_column = memblock + i;
		if (start_column < DATA_SIZE / THXPERWARP) {
			s_mempos = start_column * THXPERWARP + localId;
			g_mempos = start_column * nChannels + constantnumber;
			// TODO: we need ldg? NVProf NVVP
			s_data[s_mempos] = ldg(&d_data[g_mempos]);
		}
	}
	itemp = (int)(TAPS / (THREADS_PER_BLOCK / THXPERWARP)) + 1;
	for (int f = 0; f < itemp; f++) {
		start_column = warpId + f * (THREADS_PER_BLOCK / THXPERWARP);
		if (start_column < TAPS) {
			s_coeff[start_column*THXPERWARP + localId] = ldg(&d_coeff[start_column*nChannels + blockIdx.x*THXPERWARP + localId]);
		}
	}

	__syncthreads();

	// Calculation of the FIR part
	for (int i = 0; i < SUBBLOCK_SIZE; i++) { // WARP loops through columns in it's sub-block
		start_column = memblock + i;
		if (start_column < num_spectra) {
			s_mempos = start_column * THXPERWARP + localId;
			ftemp = 0.0f;
			for (int j = 0; j < TAPS; j++) {
				ftemp += s_coeff[j*THXPERWARP + localId] * (s_data[s_mempos + j * THXPERWARP]);
			}
			// TODO: Check NVVP Bank conflicts in SM.
			d_spectra[start_column * nChannels + constantnumber] = ftemp;
		}
	}
}

/*
* Executes all calculations for the Polyphase Filterbank. 
* First, the FIR-Filter is applied to the input_data (one Kernel call of CPF_Fir_shared_32bit).
* Second, the FFT is applied (cufftExecR2C).
*/
void CriticalPolyphaseFilterbank::Polyphase_Filterbank(thrust::device_vector<float> & d_input, thrust::device_vector <cufftComplex>& d_output) {
	// This should be in the default settings but just to make sure.
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

	GpuTimer timer;
	double fir_time = 0.0, fft_time = 0.0, fir_init = 0.0, cufft_ini = 0.0, cufft_plan = 0.0;
	//---------> CUDA block and CUDA grid parameters
	if (TIMER) timer.Start();
	int nCUDAblocks_y = (int) ceil((float)nSpectra / SM_Columns);
	int nCUDAblocks_x = (int)(nChans / THXPERWARP); 
	if (DEBUG) printf("Starting (%d,%d,%d) Blocks and (%d,%d,%d) Threads with SM_Columns %d... \n\n", nCUDAblocks_x, nCUDAblocks_y, 1, THREADS_PER_BLOCK, 1, 1, SM_Columns);

	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);	//nCUDAblocks_y goes through spectra
	dim3 blockSize(THREADS_PER_BLOCK, 1, 1); 		//nCUDAblocks_x goes through channels

	//---------> FIR filter part
	thrust::device_vector<float> fir_output((NSPECTRA + nTaps - 1) * nChans, (float)0.0);
	float* inputptr = thrust::raw_pointer_cast(&d_input[0]);
	float* outputptr = thrust::raw_pointer_cast(&fir_output[0]);
	float* coeffptr = thrust::raw_pointer_cast(&FilterCoefficients[0]);
	if (TIMER) timer.Stop(); fir_init += timer.Elapsed();
	/* You might be confused about the kernel call, because it uses 0 bytes shared Memory. 
	* But actually it uses 0 bytes dynamically allocated shared memory, the SM is statically allocated in the kernel. 
	* We list it to set the stream.*/
	if (TIMER) timer.Start();
	CPF_Fir_shared_32bit << <gridSize, blockSize, 0, stream >>> (inputptr, outputptr, coeffptr, nChans);

	if (TIMER) {
		timer.Stop();
		fir_time += timer.Elapsed();
		timer.Start();
	}
	outputptr = thrust::raw_pointer_cast(&fir_output[0]);
	if (TIMER) {
		timer.Stop();
		cufft_ini += timer.Elapsed();
		timer.Start();
	}
	cufftComplex* ptr_d_output = thrust::raw_pointer_cast(&d_output[0]);
	cufftExecR2C(plan, (cufftReal *)outputptr, (cufftComplex *)ptr_d_output);

	if (TIMER) {
		timer.Stop();
		fft_time += timer.Elapsed();
		printf("\n%f; %f; %f; %f", fir_time, fft_time, fir_init, cufft_ini);
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
CriticalPolyphaseFilterbank::CriticalPolyphaseFilterbank(std::size_t nchans, std::size_t ntaps, FilterCoefficientsType const & filter_coefficients, cudaStream_t cudaStream)
{
	// Set params.
	nTaps = ntaps;
	nChans = nchans;
	FilterCoefficients = filter_coefficients;
	stream = cudaStream;

	GpuTimer timer;

	// Create the cufft Plan.
	double cufft_plan = 0.0;
	cufftResult error;
	int nchans2 = (int)nChans;
	if (TIMER) timer.Start();
	error = cufftPlanMany(&plan, 1, &nchans2, &nchans2, 1, nChans, &nchans2, 1, nChans, CUFFT_R2C, NSPECTRA);
	cufftSetStream(plan, stream);
	if (TIMER) timer.Stop(); cufft_plan += timer.Elapsed();
	
	if (CUFFT_SUCCESS != error) {
		printf("CUFFT error: %d\n", error);
	}
	if (DEBUG && TIMER) printf("Cufft Plan time : %f", cufft_plan);
}

/**
* Destructor of Critical Polyphase Filterbank.
*/
CriticalPolyphaseFilterbank::~CriticalPolyphaseFilterbank()
{
	cufftDestroy(plan);
	FilterCoefficients.shrink_to_fit();
	FilterCoefficients.clear();
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
void CriticalPolyphaseFilterbank::process(thrust::device_vector<float> & input, thrust::device_vector <cufftComplex>& d_output)
{
	double polyphasefilterbanktime = 0;
	GpuTimer timer;
	if (DEBUG) printf("Start Process ...\n");
	if (TIMER) timer.Start();
	// We are assuming that we do not have more than 2.147.483.647 Columnstocalculate since we use an integer.
	// TODO floor or ceil ewan!
	int columnstocalculate = ceil((input.size() - ((nTaps-1) * nChans)) / nChans);
	nSpectra = columnstocalculate;
	Polyphase_Filterbank(input, d_output);
	if (DEBUG) printf("\n\t\t----- One Call of the PFB process method happily finished! :) -----\n\n");
	if (TIMER) {
		timer.Stop();
		polyphasefilterbanktime += timer.Elapsed();
		printf(";%f", polyphasefilterbanktime);
	}
}
