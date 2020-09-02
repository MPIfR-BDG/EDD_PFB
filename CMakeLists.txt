cmake_minimum_required(VERSION 3.1)
project(pfb)

find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_INCLUDE})

list(APPEND CUDA_NVCC_FLAGS -DENABLE_CUDA --std c++14 -Wno-deprecated-gpu-targets --ptxas-options=-v  --generate-line-info)

#add_subdirectory(thirdparty/psrdada_cpp)
# ToDo: PSRDADA_CPP should declare directories
#option(${PSRDADA_INCLUDE_DIR} "Location of psrdada include directory" "")
option(${PSRDADA_CPP_DIR} "Location of psrdada directory" "")


########################################################################
# Find PSRDADA + PSRDADA_CPP, should go to FindPSRDADA.cmake in psrdadacpp repo
include(FindPackageHandleStandardArgs)

find_path(
    PSRDADA_INCLUDE_DIR
    NAMES "multilog.h"
    PATHS "${CMAKE_FRAMEWORK_PATH}/include" 
    "${CMAKE_FRAMEWORK_PATH}/include/psrdada" 
    "/usr/local/include/psrdada"
    ${PSRDADA_INCLUDE_DIR}
)
find_library(PSRDADA_LIBRARY NAMES psrdada)
find_package_handle_standard_args(PSRDADA DEFAULT_MSG PSRDADA_LIBRARY PSRDADA_INCLUDE_DIR)
set(PSRDADA_INCLUDE_DIRS ${PSRDADA_INCLUDE_DIR})
set(PSRDADA_LIBRARIES ${PSRDADA_LIBRARY})

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message(STATUS "Found OpenMP" )
    set(DEPENDENCY_LIBRARIES ${DEPENDENCY_LIBRARIES} ${OpenMP_EXE_LINKER_FLAGS})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


find_path(
  PSRDADA_CPP_INCLUDE_DIR
    NAMES "psrdada_cpp/dada_client_base.hpp"
    PATHS "${CMAKE_FRAMEWORK_PATH}/include" ${PSRDADA_CPP_DIR} ${PSRDADA_CPP_DIR}/include
)
find_library(PSRDADA_CPP_LIBRARY NAMES psrdada_cpp HINTS ${PSRDADA_CPP_DIR} ${PSRDADA_CPP_DIR}/lib)
find_library(PSRDADA_CPPEFF_LIBRARY NAMES psrdada_cpp_effelsberg_edd HINTS ${PSRDADA_CPP_DIR} ${PSRDADA_CPP_DIR}/lib)
find_package_handle_standard_args(PSRDADA DEFAULT_MSG PSRDADA_LIBRARY PSRDADA_INCLUDE_DIR)
set(PSRDADA_CPP_INCLUDE_DIRS ${PSRDADA_CPP_INCLUDE_DIR})
set(PSRDADA_CPP_LIBRARIES
	${PSRDADA_CPP_LIBRARY}
  ${PSRDADA_CPPEFF_LIBRARY}
	)
message(STATUS " - PSRDADA_CPP_LIBRARIES "${PSRDADA_CPP_LIBRARIES} )

########################################################################

include_directories(${PSRDADA_INCLUDE_DIRS} ${PSRDADA_CPP_INCLUDE_DIRS})

find_package(Boost COMPONENTS log program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
cuda_add_executable(pfb main.cu)
target_link_libraries(pfb ${CUDA_CUFFT_LIBRARIES}  ${PSRDADA_CPP_LIBRARIES} ${PSRDADA_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS pfb DESTINATION bin)


add_subdirectory(thirdparty/googletest)
include_directories(
  ${gtest_SOURCE_DIR}/include
  ${gtest_SOURCE_DIR})
link_directories(${gtest_BINARY_DIR}/src)

enable_testing()
cuda_add_executable(pfb_test test.cu)
target_link_libraries(pfb_test gtest gtest_main pthread  ${CUDA_CUFFT_LIBRARIES} ${Boost_LIBRARIES}  ${PSRDADA_CPP_LIBRARIES} ${PSRDADA_LIBRARIES})
add_test(PolyphaseFilterBank pfb_test)


message(STATUS " - GTEST include directory" ${gtest_SOURCE_DIR}/include)
message(STATUS " - BOOST include directory"  ${Boost_INCLUDE_DIR})
message(STATUS " - PSRDADA include directory"  ${PSRDADA_INCLUDE_DIRS})
message(STATUS " - PSRDADACPP include directory"  ${PSRDADA_CPP_INCLUDE_DIRS} )
