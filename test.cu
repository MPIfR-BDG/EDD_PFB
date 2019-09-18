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
      FilterCoefficientsType filterCoefficients(fftSize * nTaps, 1.f);
      thrust::device_vector<float> input((nSpectra + nTaps - 1) * fftSize, 1.f);
      thrust::device_vector<float> output(nSpectra * fftSize);

      FIRFilter(thrust::raw_pointer_cast(&input[0]), thrust::raw_pointer_cast(&output[0]),
          filterCoefficients, fftSize, nTaps, nSpectra, NULL);

      // FIRFilter shoould make shure that the output is large enough
      EXPECT_EQ(output.size(), nSpectra * fftSize) << "FFTSize: " << fftSize << ", " << "NTaps: " << nTaps ;

      thrust::pair<thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator> minmax;
      minmax = thrust::minmax_element(output.begin(), output.end());

      EXPECT_EQ(output[*minmax.first], nTaps) << "FFTSize: " << fftSize << ", " << "NTaps: " << nTaps ;
      EXPECT_EQ(output[*minmax.second], nTaps) << "FFTSize: " << fftSize << ", " << "NTaps: " << nTaps ;
    }
  }
}

TEST(OutputPackAndStrip, 8bit)
{
  thrust::device_vector<cufftComplex> ppfData;
	thrust::device_vector<uint8_t> packed;
  size_t nSpectra = 128;
  size_t fftSize = 64;
  ppfData.resize(nSpectra * (fftSize / 2 + 1));
  packed.resize(nSpectra * (fftSize));

  for (size_t i =0; i < nSpectra; i++)
  {
    for (size_t j = 0; j < fftSize / 2 + 1; j++)
    {
      float v = 2 * i * fftSize / 2 + 2 * (j - 1);
      ppfData[i * (fftSize / 2 + 1) + j] = make_cuFloatComplex(v, v + 1);
    }
    ppfData[i * (fftSize / 2 + 1)] = make_cuFloatComplex(-1., -2);;
  }

  packNbitAndStripDC<8><<<128, 1024>>>((float*) thrust::raw_pointer_cast(ppfData.data()),
              (uint32_t*) thrust::raw_pointer_cast(packed.data()), fftSize, nSpectra, -128, 127);

  for (size_t i =0; i < nSpectra * fftSize; i++)
  {
    if (i < 127)
      EXPECT_EQ(i, packed[i] - 128);
    else
      EXPECT_EQ(packed[i], 255);
  }


}

TEST(OutputPackAndStrip, 32bit)
{
	thrust::device_vector<cufftComplex> ppfData;
	thrust::device_vector<uint32_t> packed;
  size_t nSpectra = 128;
  size_t fftSize = 64;
  ppfData.resize(nSpectra * (fftSize / 2 + 1));
  packed.resize(nSpectra * (fftSize));

  for (size_t i =0; i < nSpectra; i++)
  {
    for (size_t j = 0; j < fftSize / 2 + 1; j++)
    {
      float v = i * fftSize / 2 + j - 1;
      ppfData[i * (fftSize / 2 + 1) + j] = make_cuFloatComplex(v, v + .5);
    }
    ppfData[i * (fftSize / 2 + 1)] = make_cuFloatComplex(-1., -2);;
  }
  stripDCChannel<<<128, 1024>>>((float*) thrust::raw_pointer_cast(ppfData.data()),
              (float*) thrust::raw_pointer_cast(packed.data()), fftSize, nSpectra);

  for (size_t i = 0; i < nSpectra * fftSize; i += 2)
  {
    uint32_t data_p_r = packed[i];
    uint32_t data_p_i = packed[i+1];

    EXPECT_FLOAT_EQ(float (i / 2), *(float*)(&data_p_r));
    EXPECT_FLOAT_EQ(float (i / 2) + .5, *(float*)(&data_p_i));
  }
}


int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

