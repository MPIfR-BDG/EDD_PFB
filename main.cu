// Test programm for polyphase filterbank implementation
// 18 Jul 2019, Tobias Winchen

#include <iostream>
#include <iomanip>

#include "CriticalPolyphaseFilterbank.h"



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
  calculateKaiserCoefficients(filterCoefficients, 8,  1./(fft_length / 2 + 1));

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
