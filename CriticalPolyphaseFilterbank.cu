#include "CriticalPolyphaseFilterbank.h"
#include <thrust/execution_policy.h>
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


// constants for filter kenerl execution
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


// copy overlapping parts of block for FIR filter
__global__ void copy_overlap(float* __restrict__ unpackedData, size_t
    sizeOfData, size_t offset){
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < offset);
       i += blockDim.x * gridDim.x) {
    unpackedData[i] = unpackedData[sizeOfData - offset + i];
  }
}


// FIR filter with Kaiser Window  - on GPU to avoid additional dependency, not
// for performance reasons as calculated only once anyway.
__global__ void calculateKaiserCoeff(float* coeff, size_t N, float pialpha,
    float fc)
{
  float norm = cyl_bessel_i0f(pialpha);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; (i < N);
       i += blockDim.x * gridDim.x)
  {
    const float t = (2. * i) / N - 1;
    const float wn = cyl_bessel_i0f(pialpha * sqrt(1. - t * t)) / norm;

    // sin(x) / x at x=0 is not defined. To avoid branching we use small
    // offset of 1E-128 everywhere. ToDo: Check normalization for missing factors 2 or pi.
    const float hn = 1. / (float(i) - float(N/2.) + 1E-128)
      * sin(2. * fc * (float(i) - float(N/2.) + 1E-128));
    coeff[i] = hn * wn;
  }
}


void calculateKaiserCoefficients(FilterCoefficientsType &filterCoefficients,
    float pialpha, float fc)
{
  calculateKaiserCoeff<<<4, 1024>>>
    (thrust::raw_pointer_cast(filterCoefficients.data()),
     filterCoefficients.size(), pialpha, fc);
}


void FIRFilter(const float *input,
    float *output, const thrust::device_vector<float> &filterCoefficients,
    size_t fftSize, size_t nTaps, size_t nSpectra, cudaStream_t stream)
{
  const size_t SM_Columns = (DATA_SIZE / THXPERWARP - nTaps + 1);
  const size_t nCUDAblocks_y = (size_t)ceil((float)nSpectra / SM_Columns);
  const size_t nCUDAblocks_x = (size_t)(fftSize / THXPERWARP);

  dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y,
                1);                        // nCUDAblocks_y goes through spectra
  dim3 blockSize(THREADS_PER_BLOCK, 1, 1); // nCUDAblocks_x goes through fftSize

  CPF_Fir_shared_32bit<<<gridSize, blockSize, 0, stream>>>(
      input, output, thrust::raw_pointer_cast(&filterCoefficients[0]), fftSize,
      nTaps, nSpectra);
}



template <class HandlerType>
CriticalPolyphaseFilterbank<HandlerType>::CriticalPolyphaseFilterbank(
    std::size_t fftSize, std::size_t nTaps, std::size_t nSpectra,std::size_t nBits,
    FilterCoefficientsType const &filterCoefficients,  HandlerType &handler) : fftSize(fftSize), nTaps(nTaps), nSpectra(nSpectra), _nBits(nBits),
      filterCoefficients(filterCoefficients),  _handler(handler), _call_count(0)
{
  BOOST_LOG_TRIVIAL(info)
      << "Creating new CriticalPolyphaseFilterbank instance with parameters: \n"
      << "  fftSize              " << fftSize << "\n"
      << "  nTaps                " << nTaps << "\n"
      << "  nSpectra             " << nSpectra << "\n"
      << "  nBits                " << nBits;

  if (filterCoefficients.size() != (nTaps * fftSize) )
  {
    BOOST_LOG_TRIVIAL(error) << "nTaps = " << nTaps
      << ", fftSize = " << fftSize << "\n"
      << "length of filter coefficients: " << filterCoefficients.size();
    throw std::runtime_error("Bad length of filter coefficients!");
  }

  CUDA_ERROR_CHECK(cudaStreamCreate(&_h2d_stream));
  CUDA_ERROR_CHECK(cudaStreamCreate(&_proc_stream));
  CUDA_ERROR_CHECK(cudaStreamCreate(&_d2h_stream));

  cufftResult error = cufftPlan1d(&plan, fftSize, CUFFT_R2C, nSpectra);
  CUFFT_ERROR_CHECK(cufftSetStream(plan, _proc_stream));

  inputData.resize( nSpectra * fftSize * nBits / 64 );
  unpackedData.resize((nSpectra + (nTaps-1)) * fftSize);
  firOutput.resize(nSpectra * fftSize);

  BOOST_LOG_TRIVIAL(debug) << "FIR Output size: " <<  firOutput.size();
  thrust::fill(thrust::device, unpackedData.begin(), unpackedData.end(), 0);

  outputData_d.resize(nSpectra * (fftSize / 2 + 1));
  outputData_h.resize(nSpectra * fftSize / 2); // we drop the DC channel during device to host copy
  BOOST_LOG_TRIVIAL(debug) << "Output size: " <<  outputData_h.size();

  _unpacker.reset(new psrdada_cpp::Unpacker( _proc_stream ));
}


template <class HandlerType>
CriticalPolyphaseFilterbank<HandlerType>::~CriticalPolyphaseFilterbank() {
  cufftDestroy(plan);
}


template <class HandlerType>
void CriticalPolyphaseFilterbank<HandlerType>::process(
    const thrust::device_vector<float> &input,
    thrust::device_vector<cufftComplex> &output)
{

  FIRFilter( thrust::raw_pointer_cast(&input[0]),
      thrust::raw_pointer_cast(&firOutput[0]), filterCoefficients, fftSize,
      nTaps, nSpectra, _proc_stream);
  CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));

  CUFFT_ERROR_CHECK(cufftExecR2C(plan, (cufftReal *)thrust::raw_pointer_cast(&firOutput[0]),
                   (cufftComplex *)thrust::raw_pointer_cast(&output[0])));

}



template <class HandlerType>
void CriticalPolyphaseFilterbank<HandlerType>::init(psrdada_cpp::RawBytes &block)
{
  BOOST_LOG_TRIVIAL(debug) << "CriticalPolyphaseFilterbank init called";
  _handler.init(block);
}


template <class HandlerType>
bool CriticalPolyphaseFilterbank<HandlerType>::operator()(psrdada_cpp::RawBytes &block)
{
  _call_count++;
  BOOST_LOG_TRIVIAL(debug)
    << "CriticalPolyphaseFilterbank operator() called (count = "
    << _call_count << ")";

  CUDA_ERROR_CHECK(cudaStreamSynchronize(_h2d_stream));

  inputData.swap();
  BOOST_LOG_TRIVIAL(debug) << "  - Copy data to device (" 
    << inputData.size() * sizeof(inputData.a()[0]) << " bytes)";

  CUDA_ERROR_CHECK(cudaMemcpyAsync(static_cast<void *>(inputData.a_ptr()),
                                 static_cast<void *>(block.ptr()),
                                  inputData.size() * sizeof(inputData.a()[0]),
                                  cudaMemcpyHostToDevice,
                                 _h2d_stream));
  ////////////////////////////////////////////////////////////////////////
  if (_call_count == 1) {
    return false;
  }

  BOOST_LOG_TRIVIAL(debug) << "  - Unpacking raw voltages";
  size_t offset = (nTaps - 1) * fftSize;
  switch (_nBits) {
    case 8:
      _unpacker->unpack<8>(thrust::raw_pointer_cast(inputData.a_ptr()),
          thrust::raw_pointer_cast(&unpackedData.data()[offset]),
          inputData.size() );
      break;
    case 12:
      _unpacker->unpack<8>(thrust::raw_pointer_cast(inputData.a_ptr()),
          thrust::raw_pointer_cast(&unpackedData.data()[offset]),
          inputData.size() );
      break;
    default:
      throw std::runtime_error("Unsupported number of bits");
  }

  BOOST_LOG_TRIVIAL(debug) << "  - Processing ...";
  process(unpackedData, outputData_d.b());

  // copy overlap to beginning of new block
  copy_overlap<<<4, 1024, 0 ,
    _proc_stream>>>(thrust::raw_pointer_cast(unpackedData.data()),
        unpackedData.size(), offset);

  ////////////////////////////////////////////////////////////////////////
  if (_call_count == 2){
    return false;
  }
  CUDA_ERROR_CHECK(cudaStreamSynchronize(_d2h_stream));
  outputData_d.swap();

  // Drop DC channel during copy
  BOOST_LOG_TRIVIAL(debug) << "  - Copy data to host ("
    << outputData_h.size() * sizeof(outputData_h.b()[0]) << " bytes)";

  const size_t dpitch = (fftSize / 2);
  const size_t spitch = (fftSize / 2 + 1);
  const size_t width = fftSize / 2;

  CUDA_ERROR_CHECK(
       cudaMemcpy2DAsync((void *)(outputData_h.b_ptr()),
         sizeof(cufftComplex) * dpitch,
        static_cast<void *>(outputData_d.b_ptr() + 1),
         sizeof(cufftComplex) * spitch,
         sizeof(cufftComplex) * width,
         nSpectra,
         cudaMemcpyDeviceToHost, _d2h_stream));

  ////////////////////////////////////////////////////////////////////////
  if (_call_count == 3) {
    return false;
  }

  BOOST_LOG_TRIVIAL(debug) << "  - Calling handler";
  psrdada_cpp::RawBytes bytes(reinterpret_cast<char *>(outputData_h.b_ptr()),
                 outputData_h.size(),
                 outputData_h.size());

  // The handler can't do anything asynchronously without a copy here
  // as it would be unsafe (given that it does not own the memory it
  // is being passed).
  _handler(bytes);
  return false;
}


