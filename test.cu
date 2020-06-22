#include "gtest/gtest.h"
#include "CriticalPolyphaseFilterbank.h"

#include <thrust/extrema.h>


TEST(FIRFilterKernel, TestSum)
{
  // Test that the FIR filter kernel does the correct summations. For input
  // data values 1 and filter values 1 the output is the numebr of taps


  // test for multiple ntap and fft_length configurations
  for (const size_t &fftSize: {64, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072})
  {
  for (const size_t &nTaps : {4, 16, 32})
    {
				// std::cerr << "Starting: " << fftSize << std::endl;
				CUDA_ERROR_CHECK(cudaDeviceSynchronize());
				std::size_t nSpectra = 1024 * 131072 / fftSize;
				FilterCoefficientsType filterCoefficients(fftSize * nTaps, 1.f);
				thrust::device_vector<float> input((nSpectra + nTaps - 1) * fftSize, 1.f);
				thrust::device_vector<float> output(nSpectra * fftSize);
				CUDA_ERROR_CHECK(cudaDeviceSynchronize());
				//std::cerr << " - Allocated memory" << std::endl;

				FIRFilter(thrust::raw_pointer_cast(&input[0]),
						thrust::raw_pointer_cast(&output[0]),
						filterCoefficients,
						fftSize, nTaps, nSpectra, NULL);

				CUDA_ERROR_CHECK(cudaDeviceSynchronize());
				//std::cerr << " - Executed kernel" << std::endl;

				// FIRFilter shoould make shure that the output is large enough
				EXPECT_EQ(output.size(), nSpectra * fftSize) << "FFTSize: " << fftSize << ", " << "NTaps: " << nTaps ;
				CUDA_ERROR_CHECK(cudaDeviceSynchronize());
				//std::cerr << " - Checked output size" << std::endl;

				thrust::pair<thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator> minmax;
				minmax = thrust::minmax_element(output.begin(), output.end());
				cudaDeviceSynchronize();
				//std::cerr << " - Got minmax " << std::endl;

				EXPECT_EQ(*minmax.first, nTaps) << "FFTSize: " << fftSize << ", " << "NTaps: " << nTaps ;
				EXPECT_EQ(*minmax.second, nTaps) << "FFTSize: " << fftSize << ", " << "NTaps: " << nTaps ;
				cudaDeviceSynchronize();
				//std::cerr << " - Checked minmax " << std::endl;
		}
  }
}


TEST(Integrate, Integrate)
{

  for (const size_t &fftSize: {64, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072})
  {
  
	// std::cerr << "Starting: " << fftSize << std::endl;
	size_t nSpectra = 2048 * 131072 / fftSize;
	thrust::device_vector<float> output(fftSize, 0.);
	thrust::device_vector<float2> input(nSpectra  * (fftSize / 2 + 1), make_float2(1.0f, 1.0f));  // avoid thrust bug for float2
	//thrust::device_vector<float> input(nSpectra  * (fftSize / 2 + 1) *  2, 1.0f);  // avoid thrust bug for float2

	for (int i =0; (i < nSpectra) && (i < 64); i++)
	{
		input[i * (fftSize / 2 + 1)] = make_float2(23.f, 42.f); 
	}

	CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  	integrateStripDCChannel<<<128, 1024>>>((float*) thrust::raw_pointer_cast(input.data()),
              (float*) thrust::raw_pointer_cast(output.data()), fftSize, nSpectra);
	CUDA_ERROR_CHECK(cudaDeviceSynchronize());

	thrust::pair<thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator> minmax;
	minmax = thrust::minmax_element(output.begin(), output.end());
	CUDA_ERROR_CHECK(cudaDeviceSynchronize());

	EXPECT_EQ(*minmax.first, nSpectra) << "FFTSize: " << fftSize;
	EXPECT_EQ(*minmax.second, nSpectra) << "FFTSize: " << fftSize;
	CUDA_ERROR_CHECK(cudaDeviceSynchronize());
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

