cmake_minimum_required(VERSION 3.18)
project(HelloWMMA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)
find_package(CUDA REQUIRED)
MESSAGE("CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})
MESSAGE("CUDA_LIBRARIES_DIRS: " ${CUDA_LIBRARIES})
include_directories(${CUDA_INCLUDE_DIRS})
set(LIB_DIR ${CUDA_INCLUDE_DIRS}/../lib/x64)


macro(output_dirs_setup)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endmacro()

if(CMAKE_CONFIGURATION_TYPES)
  message("The default multi-configuration generator: ${CMAKE_CONFIGURATION_TYPES}")
  set(CMAKE_CONFIGURATION_TYPES ${CMAKE_BUILD_TYPE} CACHE STRING "For single config types" FORCE)
  message("Force to single-configuration generator: ${CMAKE_CONFIGURATION_TYPES}")
else()
  message("Single-configuration generator: ${CMAKE_CONFIGURATION_TYPES}")
endif()
# if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
#   string(REPLACE "/O2" "/Od" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
#   string(REPLACE "-O2" "-Od" CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")
#   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  
#   message("CMAKE_CUDA_FLAGS_RELWITHDEBINFO is ${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")
# endif()
output_dirs_setup()
set(CMAKE_CUDA_ARCHITECTURES "80")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
LINK_DIRECTORIES(${LIB_DIR})
LINK_LIBRARIES(${CUDA_LIBRARIES} cublas)

include_directories(${CMAKE_SOURCE_DIR}/src)
set(gemm_src
"${CMAKE_SOURCE_DIR}/src/wmma/gemm_fp16.cu"
"${CMAKE_SOURCE_DIR}/src/wmma/gemm_bf16.cu"
)
add_library(gemm SHARED ${gemm_src})
target_compile_definitions(gemm PRIVATE BUILDING_DLL)
add_executable(test ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cu)
target_link_libraries(test gemm)
