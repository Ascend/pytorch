cmake_minimum_required(VERSION 3.18)
project(demo LANGUAGES CXX)

# Helper function for demo targets
function(add_demo target_name)
    add_executable(${target_name} ${target_name}.cpp)

    target_include_directories(${target_name} PRIVATE
        ${TRITON_RUNTIME_DIR}/include
        ${PYTORCH_PYTHON_PACKAGES}/include
        ${PYTORCH_NPU_PACKAGES}/include
        ${Python3_INCLUDE_DIRS}
        ${CANN_PATH}/include
        ${CANN_PATH}/include/experiment
        ${CANN_PATH}/include/experiment/msprof
        ${CANN_PATH}/pkg_inc
        ${CANN_PATH}/pkg_inc/profiling
    )

    target_link_directories(${target_name} PRIVATE
        ${TRITON_RUNTIME_DIR}/build
        ${PYTORCH_PYTHON_PACKAGES}/lib
        ${PYTORCH_PYTHON_PACKAGES}/../torch.libs
        ${CANN_PATH}/lib64
        ${PYTORCH_NPU_PACKAGES}/lib
    )

    target_link_libraries(${target_name} PRIVATE
        triton_runtime
        torch
        ${TORCH_LIBRARIES}
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

    set_target_properties(${target_name} PROPERTIES
        BUILD_RPATH "${TRITON_RUNTIME_DIR}/build;${PYTORCH_PYTHON_PACKAGES}/lib;${PYTORCH_NPU_PACKAGES}/lib;${CANN_PATH}/lib64"
        INSTALL_RPATH "${TRITON_RUNTIME_DIR}/build;${PYTORCH_PYTHON_PACKAGES}/lib;${PYTORCH_NPU_PACKAGES}/lib;${CANN_PATH}/lib64"
    )
endfunction()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ──── Paths (override with -D flags if needed) ────

set(PYTORCH_PYTHON_PACKAGES "${PYTORCH_PYTHON_PACKAGES}" CACHE PATH "torch Python packages directory")
set(PYTORCH_NPU_PACKAGES "${PYTORCH_NPU_PACKAGES}" CACHE PATH "torch_npu Python packages directory")
set(CANN_PATH "${CANN_PATH}" CACHE PATH "CANN installation root")

message("CMAKE_CURRENT_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR}")
set(TRITON_RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../triton_runtime" CACHE PATH "Path to triton_runtime project root")

set(PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE FILEPATH "Python executable path")

find_package(Python3 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)

# ──── Demo executables ────

add_demo(demo)
