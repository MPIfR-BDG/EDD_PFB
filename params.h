#pragma once
#ifndef TAPS
#define TAPS 16
#endif
// Number of Channels:
#ifndef CHANNELS
#define CHANNELS 64
#endif
// Number of Spectra:
#ifndef NSPECTRA
#define NSPECTRA 16384
#endif
// Configuarable Settings which change e.g. memory size and increase or slow down computations dependent on the architecture and computation size:
// From the original github https://github.com/AstroAccelerateOrg/astro-accelerate/tree/master/lib/AstroAccelerate/PPF:
// Warning: Calculation of these parameters is different for each precision case. Sorry.
// This is a dummy parameter, which is not used in the code itself. It says how many thread-blocks we would like to be resident on single SM
#ifndef ACTIVE_BLOCKS
#define ACTIVE_BLOCKS 3
#endif
// This is again dummy parameter. It says how many single precision floating point numbers can fit into shared memory available to single block?
#ifndef TOT_SM_SIZE
#define TOT_SM_SIZE 12288
#endif
// This is again dummy parameter which says how many channels are processed per single thread-block. It is accualy size of a warp=32.
#ifndef CHANNELS_PER_BLOCK
#define CHANNELS_PER_BLOCK 32
#endif
// note: for Maxwell generation the ACTIVE_BLOCKS could be half of blocks present on single SM as this generation has 96kB of shared memory, but shared memory per block is still 48kB
// WARP size
// Remark: In case you wonder why this is so long, Actually, the thrust <thrust/host_vector.h> and <thrust/device_vector.h> libraries also have a global variable WARP making compilation impossible when it is called the same name.
#ifndef THXPERWARP
#define THXPERWARP 32
#endif
// COEFF_SIZE is given by number of taps and channels processed per thread-block COEFF_SIZE=CHANNELS_PER_BLOCK*pfb_ntaps=512
#ifndef COEFF_SIZE
#define COEFF_SIZE 512
#endif
// DATA_SIZE says how many input data elements in floating point numbers we want to store in shared memory per thread-block. DATA_SIZE=TOT_SM_SIZE/ACTIVE_BLOCKS=4096; DATA_SIZE=DATA_SIZE-COEFF_SIZE=3584;
// this is because we need to store coefficients in the shared memory along with the input data.
// Lastly we must divide this by two DATA_SIZE=DATA_SIZE/2=1792; This is because we store real and imaginary input data separately to prevent bank conflicts. 
// ? We do not have imaginary and real so actually more should be available? Check the Method.
#ifndef DATA_SIZE
#define DATA_SIZE 1792
#endif
// THREADS_PER_BLOCK gives number of threads per thread-block. It could be calculated as such THREADS_PER_BLOCK=MAX_THREADS_PER_SM(2048)/ACTIVE_BLOCKS; rounded to nearest lower multiple of 32; In case of Maxwell generation MAX_THREADS_PER_SM=2048, thus THREADS_PER_BLOCK=682.6666, rounding to nearest lower multiple of 32 gives THREADS_PER_BLOCK=672;
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 672
#endif
// SUBBLOCK_SIZE gives size of the sub-block as given in our article. It is calculated as ratio of DATA_SIZE and THREADS_PER_BLOCK rounded up so SUBBLOCK_SIZE=DATA_SIZE/THREADS_PER_BLOCK=2.6666, rounding up gives SUBBLOCK_SIZE=3;
#ifndef SUBBLOCK_SIZE
#define SUBBLOCK_SIZE 3
#endif
#ifndef OUTPUTWRITE
#define OUTPUTWRITE true
#endif
// Number of columns in shared memory.
// TODO check if we can increase that... because we have more SM as expected.
#ifndef SM_Columns
#define SM_Columns (DATA_SIZE / THXPERWARP - TAPS + 1) 
#endif 
