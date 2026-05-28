cmake_minimum_required(VERSION 3.18)
project(triton_runtime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYTORCH_PYTHON_PACKAGES "${PYTORCH_PYTHON_PACKAGES}" CACHE PATH "torch Python packages directory")
set(PYTORCH_NPU_PACKAGES "${PYTORCH_NPU_PACKAGES}" CACHE PATH "torch_npu Python packages directory")
set(CANN_PATH "${CANN_PATH}" CACHE PATH "CANN installation root")

set(PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE FILEPATH "Python executable path")

find_package(Python3 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

# ──── Library ────

set(LIB_SOURCES
    src/arg_info.cpp
    src/python_env.cpp
    src/kernel_registry.cpp
    src/kernel_compiler.cpp
    src/kernel_cache.cpp
    src/triton_runtime.cpp
)

add_library(triton_runtime SHARED ${LIB_SOURCES})

target_include_directories(triton_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${PYTORCH_PYTHON_PACKAGES}/include
        ${PYTORCH_NPU_PACKAGES}/include
        ${Python3_INCLUDE_DIRS}
        ${CANN_PATH}/include
        ${CANN_PATH}/include/experiment
        ${CANN_PATH}/include/experiment/msprof
        ${CANN_PATH}/pkg_inc
        ${CANN_PATH}/pkg_inc/profiling
)

target_link_directories(triton_runtime PRIVATE
    ${PYTORCH_PYTHON_PACKAGES}/lib
    ${PYTORCH_PYTHON_PACKAGES}/../torch.libs
    ${CANN_PATH}/lib64
    ${PYTORCH_NPU_PACKAGES}/lib
)

target_link_libraries(triton_runtime
    PUBLIC
        torch
        ${TORCH_LIBRARIES}
    PRIVATE
        pybind11::embed
        torch_cpu
        torch_npu
        c10
        dl
        Python3::Python
        runtime
        ascendcl
        profapi
        ssl
        crypto
)

target_compile_options(triton_runtime PRIVATE -Wall -Wextra -fPIC)
set_target_properties(triton_runtime PROPERTIES
    OUTPUT_NAME triton_runtime
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

# ──── Install ────

install(TARGETS triton_runtime
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/
    DESTINATION include
)

# ──── Tests ────

option(TRITON_RUNTIME_BUILD_TESTS "Build tests" OFF)
if(TRITON_RUNTIME_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
