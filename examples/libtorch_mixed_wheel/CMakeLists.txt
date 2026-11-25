cmake_minimum_required(VERSION 3.18)
project(LibtorchNPUMixedWheel LANGUAGES CXX)

set(LIBTORCH_NPU_ROOT "" CACHE PATH "Standalone LibTorch NPU package")
find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Embed)
find_library(NPU_CORE NAMES torch_npu PATHS "${LIBTORCH_NPU_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(mixed_wheel mixed_wheel.cpp)
target_compile_features(mixed_wheel PRIVATE cxx_std_20)
target_include_directories(mixed_wheel PRIVATE "${LIBTORCH_NPU_ROOT}/include")
target_link_libraries(mixed_wheel PRIVATE "${NPU_CORE}" ${TORCH_LIBRARIES} Python3::Python ${CMAKE_DL_LIBS})
