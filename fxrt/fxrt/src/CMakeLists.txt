cmake_minimum_required(VERSION 3.11)

project(fxrt VERSION 1.0)

message("-- CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message("-- PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  message("-- CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
endif()

set(CC gcc)
set(CXX g++)

# Build Debug
if("${DEBUG}" STREQUAL "on")
set(CMAKE_BUILD_TYPE "Debug")
else()
set(CMAKE_BUILD_TYPE "Release")
endif()
message("-- CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# The parent torch_npu build controls warning levels and the CXX standard.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
# Set PIC option
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # add_compile_options(-fPIC)

# Add compiler, ir, runtime and kernels directories
add_subdirectory(common)
add_subdirectory(config)
add_subdirectory(hardware)
add_subdirectory(ir)
add_subdirectory(ops)
add_subdirectory(optimize)
add_subdirectory(profiler)
add_subdirectory(runtime)

set(OBJECTS fxrt_common fxrt_config hardware_abstract execution_engine
    ops_obj pass_obj fxrt_ir_obj ops_utils fxrt_profiler)

# Create shared library. target_compile_options(fxrt PRIVATE -fPIC)
add_library(fxrt SHARED common/common.h)
set_target_properties(fxrt PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(fxrt PRIVATE ${NANOBIND_INCLUDE_DIR})
target_link_libraries(fxrt ${OBJECTS} nanobind-static Python::Module)

set_target_properties(fxrt PROPERTIES
  INSTALL_RPATH "${FXRT_LIB_RPATH}"
  BUILD_WITH_INSTALL_RPATH TRUE
)
