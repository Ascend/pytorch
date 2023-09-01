cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(libtorch_resnet)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../libtorch_npu/)

find_package(Torch REQUIRED)
find_package(Torch_npu REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(libtorch_resnet libtorch_resnet.cpp)
target_link_libraries(libtorch_resnet "${TORCH_LIBRARIES}")
target_link_libraries(libtorch_resnet "${TORCH_NPU_LIBRARIES}")

set_property(TARGET libtorch_resnet PROPERTY CXX_STANDARD 17)
