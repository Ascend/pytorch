file(GLOB SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

set(CMAKE_CXX_STANDARD 17)

if(DEFINED ENV{BSCPP_ROOT})
  set(CMAKE_C_COMPILER "$ENV{BSCPP_ROOT}/bin/clang")
  set(CMAKE_CXX_COMPILER "$ENV{BSCPP_ROOT}/bin/clang++")
else()
  message(FATAL_ERROR "env BSCPP_ROOT not exists, please export BSCPP_ROOT")
endif()

if(DEFINED ENV{BSCPP_SYCL_TARGET})
  set(CMAKE_CXX_FLAGS "-fsycl -fsycl-targets=$ENV{BSCPP_SYCL_TARGET} -std=c++17")
  set(CMAKE_SHARED_LINKER_FLAGS "-fsycl -fsycl-targets=$ENV{BSCPP_SYCL_TARGET} -std=c++17")
else()
  message(FATAL_ERROR "env BSCPP_SYCL_TARGET not exists, please export BSCPP_SYCL_TARGET")
endif()

add_library(bscpp_ops SHARED ${SOURCE_FILES})
