// Test programm for polyphase filterbank implementation
// 18 Jul 2019, Tobias Winchen

#include <iostream>
#include <iomanip>

#include "CriticalPolyphaseFilterbank.h"

// FIR filter with Kaiser Window  - on GPU to avoid additional dependency, not
// for performance reasons as calculated only once anyway.
__global__ void calculateKaiserCoeff(float* coeff, size_t N, float pialpha, float fc)
{
  float norm = cyl_bessel_i0f(pialpha);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; (i < N);
       i += blockDim.x * gridDim.x)
  {
    const float t = (2. * i) / N - 1;
    const float wn = cyl_bessel_i0f(pialpha * sqrt(1. - t * t)) / norm;

    // sin(x) / x at x=0 is not defined. To avoid branching we use small
    // offset of 1E-128 everywhere. ToDo: Check normalization for missing factors 2 or pi.
    const float hn = 1. / (float(i) - float(N/2.) + 1E-128) * sin(2. * fc * (float(i) - float(N/2.) + 1E-128));
    coeff[i] = hn * wn;
    //coeff[i] = 1. ;
  }
}



int main(int argc, char** argv)
{
  float f = 1./3;
  if (argc == 2)
    f = atof(argv[1]);

  size_t fft_length = 128;
  size_t nTap = 4;

  size_t nSpectra = 16384 * 16;

	cudaStream_t stream;
  cudaStreamCreate( &stream );

  CriticalPolyphaseFilterbank::FilterCoefficientsType filterCoefficients(fft_length * nTap);

  // Window with a critical frequency at the number of channels. pialhpa = 8 is
  // a non-optimized choice.
  calculateKaiserCoeff<<<4, 1024>>>(thrust::raw_pointer_cast(filterCoefficients.data()), filterCoefficients.size(), 8, 1./(fft_length / 2 + 1));

  cudaDeviceSynchronize();

  CriticalPolyphaseFilterbank ppf(fft_length, nTap, nSpectra, filterCoefficients, stream);

  // generate input
  thrust::device_vector<float> data_input((nSpectra + nTap- 1) * fft_length);\

  //for (size_t i =0; i <data_input.size(); i++)
  //{
  //  data_input[i] = 100. * sin( M_PI * f  *  float(i));
  //  //data_input[i] = 1.;
  //}

  thrust::device_vector<cufftComplex> data_output(nSpectra * (fft_length / 2 +1));

  ppf.process(data_input, data_output);

  cudaDeviceSynchronize();

  // Output result to stdout
 //for (size_t i =0; i < data_output.size(); i++)
 //{
 //  cufftComplex v = data_output[i];
 //  std::cout << std::scientific << v.x << "\t" << v.y << std::endl;
 //}

}
