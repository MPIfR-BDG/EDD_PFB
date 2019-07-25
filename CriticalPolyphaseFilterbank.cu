#include "CriticalPolyphaseFilterbank.h"

/*
* Compatibility helper
*/
template <typename T> __device__ __forceinline__ T ldg(const T *ptr) {
#if __CUDA_ARCH__ >= 350
  return __ldg(ptr);
#else
  return *ptr;
#endif
}


static const size_t THREADS_PER_BLOCK = 512; //

static const size_t COEFF_SIZE =
    512; // large enough to contain the filter coefficients
static const size_t DATA_SIZE =
    4096 - COEFF_SIZE; // optimize based on available shared memory of card
static const size_t SUBBLOCK_SIZE =
    (DATA_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK + 1;
static const size_t THXPERWARP = 32;


/* GPU kernel for a FIR Filter.
Implementation is based on
https://github.com/AstroAccelerateOrg/astro-accelerate/blob/master/lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu
Which is basd on
  K. Adámek, J. Novotný, W. Armour,
  A polyphase filter for many-core architectures,
  Astronomy and Computing 16 (2016), 1-16,
  https://doi.org/10.1016/j.ascom.2016.03.003.
*/
__global__ void CPF_Fir_shared_32bit(const float *__restrict__ d_data,
                                     float *__restrict__ d_spectra,
                                     const float *__restrict__ d_coeff,
                                     unsigned int fft_size, unsigned int nTaps,
                                     unsigned int nSpectra) {
  float ftemp;
  int memblock, localId, s_mempos, g_mempos, num_spectra, start_column, warpId,
      itemp;
  int tx = threadIdx.x;

  __shared__ float s_data[DATA_SIZE];
  __shared__ float s_coeff[COEFF_SIZE];

  warpId = ((int)tx / THXPERWARP);
  memblock = warpId * SUBBLOCK_SIZE;
  localId = tx -
            ((int)tx / THXPERWARP) *
                THXPERWARP; // Calculates threads Id within a WARP
  num_spectra = (DATA_SIZE / THXPERWARP - nTaps + 1);

  // read input data from global memory a store them into shared memory
  int constantnumber =
      blockIdx.x * THXPERWARP + blockIdx.y * num_spectra * fft_size + localId;
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
      s_coeff[start_column * THXPERWARP + localId] =
          ldg(&d_coeff[start_column * fft_size + blockIdx.x * THXPERWARP +
                       localId]);
    }
  }

  __syncthreads();
  // Calculation of the FIR part
  for (int i = 0; i < SUBBLOCK_SIZE;
       i++) { // WARP loops through columns in it's sub-block
    start_column = memblock + i;
    if (start_column < num_spectra) {
      s_mempos = start_column * THXPERWARP + localId;
      ftemp = 0.0f;
      for (int j = 0; j < nTaps; j++) {
        ftemp += s_coeff[j * THXPERWARP + localId] *
                 (s_data[s_mempos + j * THXPERWARP]);
      }
      // TODO: Check NVVP Bank conflicts in SM.
      if (start_column * fft_size + constantnumber < fft_size * nSpectra) {
        d_spectra[start_column * fft_size + constantnumber] = ftemp;
      }
    }
  }
}



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
  }
}


void calculateKaiserCoefficients(CriticalPolyphaseFilterbank::FilterCoefficientsType &filterCoefficients, float pialpha, float fc)
{
  calculateKaiserCoeff<<<4, 1024>>>(thrust::raw_pointer_cast(filterCoefficients.data()), filterCoefficients.size(), pialpha, fc);
}




void FIRFilter(const thrust::device_vector<float> &input,
    thrust::device_vector<float> &output, const
    thrust::device_vector<float> &filterCoefficients, size_t fftSize,
    size_t nTaps, size_t nSpectra, cudaStream_t stream)
{
  output.resize(fftSize * nSpectra);
  const size_t SM_Columns = (DATA_SIZE / THXPERWARP - nTaps + 1);
  const size_t nCUDAblocks_y = (size_t)ceil((float)nSpectra / SM_Columns);
  const size_t nCUDAblocks_x = (size_t)(fftSize / THXPERWARP);

  dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y,
                1);                        // nCUDAblocks_y goes through spectra
  dim3 blockSize(THREADS_PER_BLOCK, 1, 1); // nCUDAblocks_x goes through fftSize

  CPF_Fir_shared_32bit<<<gridSize, blockSize, 0, stream>>>(
      thrust::raw_pointer_cast(&input[0]),
      thrust::raw_pointer_cast(&output[0]),
      thrust::raw_pointer_cast(&filterCoefficients[0]), fftSize, nTaps,
      nSpectra);
}



CriticalPolyphaseFilterbank::CriticalPolyphaseFilterbank(
    std::size_t fftSize, std::size_t nTaps, std::size_t nSpectra,
    FilterCoefficientsType const &filterCoefficients, cudaStream_t stream)
    : fftSize(fftSize), nTaps(nTaps), nSpectra(nSpectra),
      filterCoefficients(filterCoefficients), stream(stream) {
  cufftResult error = cufftPlan1d(&plan, fftSize, CUFFT_R2C, nSpectra);
  cufftSetStream(plan, stream);

  if (CUFFT_SUCCESS != error) {
    printf("CUFFT error: %d\n", error);
  }

  firOutput.resize(nSpectra * fftSize);
}


CriticalPolyphaseFilterbank::~CriticalPolyphaseFilterbank() {
  cufftDestroy(plan);
}


void CriticalPolyphaseFilterbank::process(
    const thrust::device_vector<float> &input,
    thrust::device_vector<cufftComplex> &output) {

  FIRFilter(input, firOutput, filterCoefficients, fftSize, nTaps, nSpectra, stream);

  cufftResult error =
      cufftExecR2C(plan, (cufftReal *)thrust::raw_pointer_cast(&firOutput[0]),
                   (cufftComplex *)thrust::raw_pointer_cast(&output[0]));

  if (CUFFT_SUCCESS != error) {
    printf("CUFFT error: %d\n", error);
  }

  cudaDeviceSynchronize();
}
