cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(libtorch_resnet)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(torch_npu_path ${CMAKE_CURRENT_SOURCE_DIR}/../../libtorch_npu/)
include_directories(${torch_npu_path}/include)
link_directories(${torch_npu_path}/lib)

add_executable(libtorch_resnet libtorch_resnet.cpp)
target_link_libraries(libtorch_resnet "${TORCH_LIBRARIES}")
target_link_libraries(libtorch_resnet torch_npu)

set_property(TARGET libtorch_resnet PROPERTY CXX_STANDARD 14)
