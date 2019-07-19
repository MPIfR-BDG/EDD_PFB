cmake_minimum_required(VERSION 3.1)
project(pfb)

find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_INCLUDE})

list(APPEND CUDA_NVCC_FLAGS -DENABLE_CUDA --std c++11 -Wno-deprecated-gpu-targets --ptxas-options=-v  --generate-line-info)
#list(APPEND CUDA_NVCC_FLAGS_DEBUG --debug; --device-debug; --generate-line-info -Xcompiler "-Wextra" -Xcompiler "-Werror")


#
#add_library(pfb STATIC
#  kernel.cu
#  CriticalPolyphaseFilterbank.cu
#  CriticalPolyphaseFilterbank.h
#  params.h
#  debug.h
#  filehelper.h
#  reference.c
#  resource.h
#  Rng.h
#  timer.cuh
#  utils_cuda.h
#  utils_file.h
#)
#
#target_compile_features(particles PUBLIC cxx_std_11)
#
#                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(pfb main.cu CriticalPolyphaseFilterbank.cu)
target_link_libraries(pfb ${CUDA_CUFFT_LIBRARIES})
#set_property(TARGET pfb
#             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

