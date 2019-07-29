cmake_minimum_required(VERSION 3.1)
project(pfb)

find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_INCLUDE})

list(APPEND CUDA_NVCC_FLAGS -DENABLE_CUDA --std c++11 -Wno-deprecated-gpu-targets --ptxas-options=-v  --generate-line-info)


add_subdirectory(thirdparty/psrdada_cpp)


cuda_add_executable(pfb main.cu CriticalPolyphaseFilterbank.cu)
target_link_libraries(pfb ${CUDA_CUFFT_LIBRARIES} ${PSRDADA_CPP})



add_subdirectory(thirdparty/googletest)
include_directories(
  ${gtest_SOURCE_DIR}/include
  ${gtest_SOURCE_DIR})
link_directories(${gtest_BINARY_DIR}/src)

enable_testing()
cuda_add_executable(pfb_test test.cu CriticalPolyphaseFilterbank.cu)
target_link_libraries(pfb_test gtest gtest_main pthread  ${CUDA_CUFFT_LIBRARIES})
add_test(PolyphaseFilterBank pfb_test)

