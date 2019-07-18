#pragma once
#include <cstdint>
#include <cufft.h>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class CriticalPolyphaseFilterbank
{
public:
	typedef	thrust::device_vector<float> FilterCoefficientsType;
private:
	FilterCoefficientsType FilterCoefficients;
	std::size_t nTaps;
	std::size_t nChans;
	std::size_t nSpectra;
	std::size_t maxColumn;
	cudaStream_t stream;
	cufftHandle plan;
public:
	/**
	* @brief Construct a new critically sampled polyphase filterbank
	*
	* @param[in] nchans The number of spectral channels to produce
	* @param[in] ntaps The number of filter taps to use
	* @param filter_coefficients The filter coefficients
	*
	* @detail The number of filter coefficients should be equal to ntaps x nchans.
	*/
	explicit CriticalPolyphaseFilterbank(std::size_t nchans, std::size_t ntaps,
		FilterCoefficientsType const & filter_coefficients, cudaStream_t cudaStream);
	~CriticalPolyphaseFilterbank();
	CriticalPolyphaseFilterbank(CriticalPolyphaseFilterbank const &) = delete;
	/**
	* @brief Apply the polyphase filter to a block of timeseries data
	* This function will call the other process function with the stream argument of the class.
	*
	* @param input The timeseries to process
	* @param output The channelised filtered output data
	*
	* @tparam InputType The input data type
	* @tparam OutputType The output data type

	We have overloaded the function, for different application context.
	First, there is a simple process function incase we are just processing input and output.
	Second, we might want to change the stream in this case the stream can be set as the last parameter.
	Thrid, we might want to change the number of taps and channels. --> we don't really want to do this to often since the creation of the cuFFTPlan requires roughly 200 ms, which can be considered a major
	slowdown since without the creation of the cuFFTPlan the overall execution of the FIR Filter and the FFT transform merely requires 14 ms.
	*/
	// Make const input -- thrust::device_vector<float2/CufftComplex> for output.
	// Check cuffTComplex in vector with different compiler.
	void process(thrust::device_vector<float> & input, thrust::device_vector <cufftComplex>& d_output);

	/* When all parameter of the class are set correctly we actually start the Polyphase Filterbank consisting of a FIR Filter and a FFT.*/
	private: void Polyphase_Filterbank(thrust::device_vector<float>& d_input, thrust::device_vector <cufftComplex>& d_output);

};
