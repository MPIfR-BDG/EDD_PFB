#include "gtest/gtest.h"
#include "CriticalPolyphaseFilterbank.h"

#include <thrust/extrema.h>


TEST(FIRFilterKernel, TestSum)
{
  // Test that the FIR filter kernel does the correct summations. For input
  // data values 1 and filter values 1 the output is the numebr of taps

  std::size_t nSpectra = 16384;

  // test for multiple ntap and fft_length configurations
  for (const size_t &nTaps : {4, 16})
  {
    for (const size_t &fftSize: {64, 128})
    {
      CriticalPolyphaseFilterbank::FilterCoefficientsType filterCoefficients(fftSize * nTaps, 1.f);
      thrust::device_vector<float> input((nSpectra + nTaps - 1) * fftSize, 1.f);
      thrust::device_vector<float> output;

      FIRFilter(input, output, filterCoefficients, fftSize, nTaps, nSpectra, NULL);

      // FIRFilter shoould make shure that the output is large enough
      EXPECT_EQ(output.size(), nSpectra * fftSize) << "FFTSize: " << fftSize << ", " << "NTaps: " << nTaps ;

      thrust::pair<thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator> minmax;
      minmax = thrust::minmax_element(output.begin(), output.end());

      EXPECT_EQ(output[*minmax.first], nTaps) << "FFTSize: " << fftSize << ", " << "NTaps: " << nTaps ;
      EXPECT_EQ(output[*minmax.second], nTaps) << "FFTSize: " << fftSize << ", " << "NTaps: " << nTaps ;


    }
  }
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

