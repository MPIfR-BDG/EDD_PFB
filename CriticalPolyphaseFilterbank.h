#pragma once
#include <cstdint>
#include <cufft.h>
#include <cuda.h>
#include <thrust/device_vector.h>


/* FIR Filter. */
void FIRFilter(const thrust::device_vector<float> &input,
    thrust::device_vector<float> &output, const
    thrust::device_vector<float> &filterCoefficients, size_t fftSize,
    size_t nTaps, size_t nSpectra, cudaStream_t stream = NULL);


class CriticalPolyphaseFilterbank {
public:
  typedef thrust::device_vector<float> FilterCoefficientsType;

private:
  FilterCoefficientsType filterCoefficients;
  std::size_t nTaps;
  std::size_t fftSize;
  std::size_t nSpectra;

  cudaStream_t stream;
  cufftHandle plan;

	thrust::device_vector<float> firOutput;
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
      std::size_t fftSize, std::size_t nTaps, std::size_t nSpectra,
      FilterCoefficientsType const &filterCoefficients,
      cudaStream_t stream);
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
};


/* Fills the filterCoefficients vector with valus according to a FIR filter
 * with Kiser Window with parameters pialpha and critical frequency fc*/
void calculateKaiserCoefficients(CriticalPolyphaseFilterbank::FilterCoefficientsType &filterCoefficients, float pialpha, float fc);
