#pragma once
#include <cstdint>
#include <cufft.h>
#include <cuda.h>
#include <thrust/device_vector.h>

#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/double_device_buffer.cuh"
#include "psrdada_cpp/double_host_buffer.cuh"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/Unpacker.cuh"

/* FIR Filter. */
void FIRFilter(
    const float *input,
    cufftComplex *output,
    thrust::device_vector<float> &filterCoefficients, size_t fftSize,
    size_t nTaps, size_t nSpectra, cudaStream_t stream = NULL);


typedef thrust::device_vector<float> FilterCoefficientsType;


template <class HandlerType>
class CriticalPolyphaseFilterbank {
private:
  FilterCoefficientsType filterCoefficients;
  std::size_t nTaps;
  std::size_t fftSize;
  std::size_t nSpectra;
  std::size_t _call_count;
  std::size_t inputBitDepth;
  std::size_t outputBitDepth;
  float minV, maxV;

  HandlerType &_handler;

  cufftHandle plan;
  cudaStream_t _h2d_stream;
  cudaStream_t _proc_stream;
  cudaStream_t _d2h_stream;

  std::unique_ptr<psrdada_cpp::Unpacker> _unpacker;

  // io double buffer
  psrdada_cpp::DoubleDeviceBuffer<uint64_t> inputData;
  psrdada_cpp::DoubleDeviceBuffer<uint32_t> outputData_d;
  psrdada_cpp::DoubleHostBuffer<uint32_t> outputData_h;

  // scratch data
	thrust::device_vector<float> firOutput;
	thrust::device_vector<cufftComplex> ppfData;
	thrust::device_vector<float> unpackedData;
public:
  /**
  * @brief Construct a new critically sampled polyphase filterbank
  *
  * @param[in] fftSize The number fo samples used for the fft
  * @param[in] nTaps The number of filter taps to use
  * @param[in] nSpectra The number of output spectra to generate
  * @param filter_coefficients The filter coefficients
  *
  * @detail The number of filter coefficients should be equal to ntaps x nchans.
  */
  explicit CriticalPolyphaseFilterbank(
      std::size_t fftSize, std::size_t nTaps, std::size_t nSpectra, std::size_t inputBitDepth, std::size_t outputBitDepth,  float minV, float maxV,
      FilterCoefficientsType const &filterCoefficients,
      HandlerType &handler);

  ~CriticalPolyphaseFilterbank();
  CriticalPolyphaseFilterbank(CriticalPolyphaseFilterbank const &) = delete;


  /**
  * @brief Apply the polyphase filter to a block of timeseries data
  * This function will call the other process function with the stream argument
  * of the class.
  *
  * @param input The timeseries to process
  * @param output The channelised filtered output data
  *
  * @tparam InputType The input data type
  * @tparam OutputType The output data type
  */
  void process(const thrust::device_vector<float> &input,
               thrust::device_vector<cufftComplex> &output);

  void init(psrdada_cpp::RawBytes &block);
  bool operator()(psrdada_cpp::RawBytes &block);

};


/* Fills the filterCoefficients vector with valus according to a FIR filter
 * with Kiser Window with parameters pialpha and critical frequency fc*/
void calculateKaiserCoefficients(FilterCoefficientsType &filterCoefficients, float pialpha, float fc);

#include "CriticalPolyphaseFilterbank.cu"
