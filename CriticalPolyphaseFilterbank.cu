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
static const size_t THREADS_PER_BLOCK = 256; //


// COEFF_SIZE=CHANNELS_PER_BLOCK * TAPS

static const size_t COEFF_SIZE = 1024;
//    512; // large enough to contain the filter coefficients
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
  const size_t nCUDAblocks_y = (size_t)floor((float)nSpectra / SM_Columns);
  //const size_t nCUDAblocks_y = (size_t)ceil((float)nSpectra / SM_Columns);
  if (nCUDAblocks_y > 65535)
  {
    BOOST_LOG_TRIVIAL(error) << "Requested " << nCUDAblocks_y
      << " nCUDAblocks_y - maximum is 65536! Try reducing the size"
         " of the input buffer (or increasing the number of channels).";
  }
  const size_t nCUDAblocks_x = (size_t)(fftSize / THXPERWARP);

  dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y,
                1);                        // nCUDAblocks_y goes through spectra
  dim3 blockSize(THREADS_PER_BLOCK, 1, 1); // nCUDAblocks_x goes through fftSize

  CPF_Fir_shared_32bit<<<gridSize, blockSize, 0, stream>>>(
      input, output, thrust::raw_pointer_cast(&filterCoefficients[0]), fftSize,
      nTaps, nSpectra);
}




// convert a float to an int32 clipped to minv, maxv and with a maxium
// bit_depth. For an input_bit_depth of 2 and 4 the loop is faster than fmin,
// fmax
template <unsigned int input_bit_depth>
__device__ __forceinline__ uint32_t convert32(float inp, float maxV, float minV, float level)
{
  uint32_t p = 0;
  #pragma unroll
  for (int k = 1; k < (1 << input_bit_depth); k++) {
    p += (inp > ((k * level) + minV));
  } // this is more efficient than fmin, fmax for clamp and cast.
  return p;
}

template <>
__device__ __forceinline__ uint32_t convert32<8>(float inp, float maxV, float minV, float level)
{
  inp -= minV;
  inp /= level;
  inp = fminf(inp, ((1 << 8)- 1));
  inp = fmaxf(inp, 0);
  uint32_t p = uint32_t (inp);
  return p;
}

template <>
__device__ __forceinline__ uint32_t convert32<16>(float inp, float maxV, float minV, float level)
{
  inp -= minV;
  inp /= level;
  inp = fminf(inp, ((1 << 16)- 1));
  inp = fmaxf(inp, 0);
  uint32_t p = uint32_t (inp);
  return p;
}



// pack float to 2,4,8,16 bit integers with linear scaling. Striop the DC
// component
template <unsigned int input_bit_depth>
__global__ void packNbitAndStripDC(const float *__restrict__ input,
                         uint32_t *__restrict__ output, size_t fftSize, size_t nspectra,
                         float minV, float maxV) {
  // number of values to pack into one output element, use 32 bit here to
  // maximize number of threads
  const uint8_t NPACK = 32 / input_bit_depth;

  const float l = (maxV - minV) / ((1 << input_bit_depth) - 1);
  __shared__ uint32_t tmp[1024];

  for (uint32_t i = NPACK * blockIdx.x * blockDim.x + threadIdx.x;
       (i < fftSize * nspectra); i += blockDim.x * gridDim.x * NPACK)
        // factor 2 as two float for complex
  {
    tmp[threadIdx.x] = 0;

    #pragma unroll
    for (uint8_t j = 0; j < NPACK; j++) {
      // Load new input value, clip and convert to Nbit integer
      size_t linearIndex = i + j * blockDim.x;
      size_t offset = (1 + linearIndex / fftSize) * 2; // factor 2 as two float for complex
      const float inp = input[linearIndex + offset];

      uint32_t p = convert32<input_bit_depth>(inp, maxV, minV, l);
      // store in shared memory with linear access
      tmp[threadIdx.x] += p << (input_bit_depth * j);
    }
    __syncthreads();

    // load value from shared memory and rearange to output - the read value is
    // reused per warp
    uint32_t out = 0;

    // bit mask: Thread 0 always first input_bit_depth bits, thread 1 always
    // second input_bit_depth bits, ...
    const uint32_t mask = ((1 << input_bit_depth) - 1) << (input_bit_depth * (threadIdx.x % NPACK));
    #pragma unroll
    for (uint32_t j = 0; j < NPACK; j++) {
      uint32_t v = tmp[(threadIdx.x / NPACK) * NPACK + j] & mask;
      // retrieve correct bits
      v = v >> (input_bit_depth * (threadIdx.x % NPACK));
      v = v << (input_bit_depth * j);
      out += v;
    }

    size_t oidx = threadIdx.x / NPACK + (threadIdx.x % NPACK) * (blockDim.x / NPACK) + (i - threadIdx.x) / NPACK;
    output[oidx] = out;
    __syncthreads();
  }
}



__global__ void stripDCChannel(const float *__restrict__ input,
                         float *__restrict__ output, size_t fftSize, size_t nspectra) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       (i < fftSize * nspectra); i += blockDim.x * gridDim.x)
        // factor 2 as two float for complex
  {
      size_t offset = (1 + i / fftSize) * 2; // factor 2 as two float for complex
      output[i] = input[i + offset];
  }
}




template <class HandlerType>
CriticalPolyphaseFilterbank<HandlerType>::CriticalPolyphaseFilterbank(
    std::size_t fftSize, std::size_t nTaps, std::size_t nSpectra,std::size_t inputBitDepth, size_t outputBitDepth, float minV, float maxV,
    FilterCoefficientsType const &filterCoefficients,  HandlerType &handler) :
  fftSize(fftSize), nTaps(nTaps), nSpectra(nSpectra),
  inputBitDepth(inputBitDepth), outputBitDepth(outputBitDepth), minV(minV), maxV(maxV),
  filterCoefficients(filterCoefficients),  _handler(handler), _call_count(0)
{
  BOOST_LOG_TRIVIAL(info)
      << "Creating new CriticalPolyphaseFilterbank instance with parameters: \n"
      << "  fftSize              " << fftSize << "\n"
      << "  nTaps                " << nTaps << "\n"
      << "  nSpectra             " << nSpectra << "\n"
      << "  inputBitDepth        " << inputBitDepth << "\n"
      << "  outputBitDepth       " << outputBitDepth;

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

  inputData.resize( nSpectra * fftSize * inputBitDepth/ 64 );
  unpackedData.resize((nSpectra + (nTaps-1)) * fftSize);
  firOutput.resize(nSpectra * fftSize);

  BOOST_LOG_TRIVIAL(debug) << "FIR Output size: " <<  firOutput.size();
  thrust::fill(thrust::device, unpackedData.begin(), unpackedData.end(), 0);

  if(((nSpectra * (fftSize / 2) * outputBitDepth) % 32) != 0)
  {
    BOOST_LOG_TRIVIAL(error) << "Output size dose not fully utilze array of 32 bit. Adapt fftSize to multiple of 32!";
    throw std::runtime_error("Bad size of buffer");
  }
  outputData_d.resize(nSpectra * (fftSize) * outputBitDepth / 32);
  outputData_h.resize(outputData_d.size());


  // we drop the DC channel during device to host copy
  ppfData.resize(nSpectra * (fftSize / 2 + 1));
  BOOST_LOG_TRIVIAL(debug) << "Output size: " <<  outputData_h.size() * 4<< " (byte)";

  _unpacker.reset(new psrdada_cpp::effelsberg::edd::Unpacker( _proc_stream ));
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
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
  std::stringstream headerInfo;
  headerInfo << "\n"
      << "# CriticalPFB parameters: \n"
      << "fftSize                  " << fftSize << "\n"
      << "nTaps                    " << nTaps << "\n"
      << "outputBitDepth           " << outputBitDepth << "\n"
      << "minV                     " << minV << "\n"
      << "maxV                     " << maxV << "\n";
  BOOST_LOG_TRIVIAL(debug) << "CriticalPolyphaseFilterbank init header info:\n" << headerInfo.str() ;

  size_t bEnd = std::strlen(block.ptr());
  if (bEnd + headerInfo.str().size() < block.total_bytes())
  {
    std::strcpy(block.ptr() + bEnd, headerInfo.str().c_str());
  }
  else
  {
    BOOST_LOG_TRIVIAL(warning) << "Header of size " << block.total_bytes()
      << " bytes already contains " << bEnd
      << "bytes. Cannot add PFB info of size "
      << headerInfo.str().size() << " bytes.";
  }
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
  outputData_d.swap();
  outputData_h.swap();

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
  switch (inputBitDepth) {
    case 8:
      _unpacker->unpack<8>(thrust::raw_pointer_cast(inputData.b_ptr()),
          thrust::raw_pointer_cast(&unpackedData.data()[offset]),
          inputData.size() );
      break;
    case 12:
      _unpacker->unpack<12>(thrust::raw_pointer_cast(inputData.b_ptr()),
          thrust::raw_pointer_cast(&unpackedData.data()[offset]),
          inputData.size() );
      break;
    default:
      throw std::runtime_error("Unsupported number of bits");
  }

  BOOST_LOG_TRIVIAL(debug) << "  - Processing ...";
  process(unpackedData, ppfData);
  //pack FFt output to outputbitdepth and drop DC channel
  switch (outputBitDepth)
  {
    case 2:
      packNbitAndStripDC<2><<<128, 1024, 0,_proc_stream>>>((float*) thrust::raw_pointer_cast(ppfData.data()),
              thrust::raw_pointer_cast(outputData_d.a().data()), fftSize, nSpectra, minV, maxV);
      break;
    case 4:
      packNbitAndStripDC<4><<<128, 1024, 0,_proc_stream>>>((float*) thrust::raw_pointer_cast(ppfData.data()),
              thrust::raw_pointer_cast(outputData_d.a().data()), fftSize, nSpectra, minV, maxV);
      break;
    case 8:
      packNbitAndStripDC<8><<<128, 1024, 0,_proc_stream>>>((float*) thrust::raw_pointer_cast(ppfData.data()),
              thrust::raw_pointer_cast(outputData_d.a().data()), fftSize, nSpectra, minV, maxV);
      break;
    case 16:
      packNbitAndStripDC<16><<<128, 1024, 0,_proc_stream>>>((float*) thrust::raw_pointer_cast(ppfData.data()),
              thrust::raw_pointer_cast(outputData_d.a().data()), fftSize, nSpectra, minV, maxV);
      break;
    case 32:
      // for 32 bit it would be more efficient to do this during copy as
      // initially implemented, see git revs.
      stripDCChannel<<<128, 1024, 0,_proc_stream>>>((float*) thrust::raw_pointer_cast(ppfData.data()),
              (float*) thrust::raw_pointer_cast(outputData_d.a().data()), fftSize, nSpectra);
      break;
    default:
      throw std::runtime_error("Unsupported number of bits");
  }

  // copy overlap to beginning of new block
  copy_overlap<<<4, 1024, 0 ,
    _proc_stream>>>(thrust::raw_pointer_cast(unpackedData.data()),
        unpackedData.size(), offset);

  CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));
  ////////////////////////////////////////////////////////////////////////
  if (_call_count == 2){
    return false;
  }
  CUDA_ERROR_CHECK(cudaStreamSynchronize(_d2h_stream));

  BOOST_LOG_TRIVIAL(debug) << "  - Copy data to host (" << outputData_h.size() * sizeof(outputData_h.a()[0]) << " bytes)";

  CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(outputData_h.a_ptr()),
                        static_cast<void *>(outputData_d.b_ptr()),
                        outputData_h.size() * sizeof(outputData_h.a_ptr()[0]),
                        cudaMemcpyDeviceToHost, _d2h_stream));

  ////////////////////////////////////////////////////////////////////////
  if (_call_count == 3) {
    return false;
  }

  BOOST_LOG_TRIVIAL(debug) << "  - Calling handler";
  psrdada_cpp::RawBytes bytes(reinterpret_cast<char *>(outputData_h.b_ptr()),
                 outputData_h.size() * sizeof(outputData_h.b_ptr()[0]),
                 outputData_h.size() * sizeof(outputData_h.b_ptr()[0]));

  // The handler can't do anything asynchronously without a copy here
  // as it would be unsafe (given that it does not own the memory it
  // is being passed).
  _handler(bytes);
  return false;
}


