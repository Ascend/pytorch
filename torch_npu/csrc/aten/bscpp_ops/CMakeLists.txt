file(GLOB SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

set(CMAKE_CXX_STANDARD 17)

if(DEFINED ENV{BISHENG_CPP_HOME})
  set(CMAKE_C_COMPILER "$ENV{BISHENG_CPP_HOME}/bin/clang")
  set(CMAKE_CXX_COMPILER "$ENV{BISHENG_CPP_HOME}/bin/clang++")
else()
  message(FATAL_ERROR "env BISHENG_CPP_HOME not exists, please export BISHENG_CPP_HOME")
endif()

if(DEFINED ENV{BSCPP_SYCL_TARGET})
  set(CMAKE_CXX_FLAGS "-fsycl -fsycl-targets=$ENV{BSCPP_SYCL_TARGET} -std=c++17")
  set(CMAKE_SHARED_LINKER_FLAGS "-fsycl -fsycl-targets=$ENV{BSCPP_SYCL_TARGET} -std=c++17")
else()
  message(FATAL_ERROR "env BSCPP_SYCL_TARGET not exists, please export BSCPP_SYCL_TARGET")
endif()

add_library(bscpp_ops SHARED ${SOURCE_FILES})
