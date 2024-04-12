cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

find_program(CCACHE ccache)
if(${CCACHE} STREQUAL "CCACHE-NOTFOUND")
  message(STATUS "Compile without ccache")
else()
  set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE} CACHE PATH "cache Compiler")
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE} CACHE PATH "cache Compiler")
  message(STATUS "CMAKE_C_COMPILER_LAUNCHER:${CMAKE_C_COMPILER_LAUNCHER}")
  message(STATUS "CMAKE_CXX_COMPILER_LAUNCHER:${CMAKE_CXX_COMPILER_LAUNCHER}")
endif()

project(LibtorchNPUDemo CXX C)

set(LINUX TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define build type
IF(CMAKE_BUILD_TYPE MATCHES Debug)
  message("Debug build.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_DEBUG")
ELSEIF(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  message("RelWithDebInfo build")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
ELSE()
  message("Release build.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
ENDIF()

SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
SET(CMAKE_CXX_FLAGS_RELEASE "-O2")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# set the real path
set(torch_path "path/to/your/libtorch/for/pytorch")
if(NOT EXISTS ${torch_path})
  message(FATAL_ERROR "please set correct torch_path")
endif()
include_directories(${torch_path}/include)
link_directories(${torch_path}/lib)

# set the real path
set(torch_npu_path "path/to/your/libtorch_npu/for/torch_npu")
if(NOT EXISTS ${torch_npu_path})
  message(FATAL_ERROR "please set correct torch_npu_path")
endif()
include_directories(${torch_npu_path}/include)
link_directories(${torch_npu_path}/lib)

add_executable(torch_npu_demo inference_model.cpp)
target_link_libraries(torch_npu_demo -ltorch -ltorch_cpu -lc10 -ltorch_npu)
