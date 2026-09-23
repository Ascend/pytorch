cmake_minimum_required(VERSION 3.11)
project(fxrt VERSION 1.0)

message("-- CMAKE_C_COMPILER_VERSION: ${CMAKE_C_COMPILER_VERSION}")
message("-- CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  message("-- CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
endif()

set(CC gcc)
set(CXX g++)

# fxrt is built only as part of torch_npu: the top-level CMakeLists adds this
# directory after the torch_npu target exists when BUILD_FXRT is set, and
# setup.py passes FXRT_INSTALL_DST, the directory the wheel's torch_npu/fxrt is
# packed from.
if(NOT TARGET torch_npu OR NOT DEFINED FXRT_INSTALL_DST)
  message(FATAL_ERROR "fxrt is built only as part of torch_npu (BUILD_FXRT and FXRT_INSTALL_DST)")
endif()

# Runtime search paths for the wheel layout: fxrt libraries sit in
# torch_npu/fxrt/lib, nanobind modules in torch_npu/fxrt, and the torch_npu
# libraries they link (libtorch_npu.so and its PUBLIC dependencies such as
# libtensorpipe.so) in torch_npu/lib. auditwheel resolves every recorded
# dependency through these paths, so torch_npu/lib has to be on them.
set(FXRT_LIB_RPATH "$ORIGIN:$ORIGIN/../../lib")
set(FXRT_MODULE_RPATH "$ORIGIN:$ORIGIN/lib:$ORIGIN/../lib")

if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "/usr/bin/g++")
endif()
message("-- CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "ccache found at: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

if(ENABLE_TORCH_FRONT)
  add_compile_definitions(ENABLE_TORCH_FRONT)
endif()

set(TOP_DIR "${PROJECT_SOURCE_DIR}/..")
include(${TOP_DIR}/cmake/nlohmann_json.cmake)
include(${TOP_DIR}/cmake/dvm.cmake)
include(${TOP_DIR}/cmake/torch_npu.cmake)

# Python and nanobind, resolved once for src/ and python/.
#
# The outer PTA build is the source of truth: its setup.py passes the
# interpreter that drives the build as FXRT_TARGET_PYTHON, the same value
# torchair gets as TORCHAIR_TARGET_PYTHON. Pinning it keeps fxrt on exactly the
# Python torch_npu was built against -- PTA derives its own PYTHON_INCLUDE_DIR
# from that interpreter, and build images carry several Pythons, so letting
# CMake search PATH can resolve a different one.
#
# Only Development.Module is requested. Like PTA, nothing here links libpython:
# these libraries are loaded into a running interpreter, so Python symbols stay
# undefined, and build images such as manylinux ship no libpython at all.
# find_package is still needed because nanobind's CMake config requires the
# Python::Interpreter and Python::Module targets.
if(DEFINED FXRT_TARGET_PYTHON)
  set(Python_EXECUTABLE "${FXRT_TARGET_PYTHON}")
endif()
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "fxrt Python: ${Python_EXECUTABLE} (headers: ${Python_INCLUDE_DIRS})")

execute_process(COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
                OUTPUT_VARIABLE nanobind_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.include_dir())"
                OUTPUT_VARIABLE NANOBIND_INCLUDE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(nanobind CONFIG REQUIRED)

# Include directories
include_directories(include)
include_directories(src)

add_subdirectory(src)
add_subdirectory(python)

# ---------------------------------------------------------------------------
# Stage the built artifacts and Python sources into FXRT_INSTALL_DST. setup.py
# runs ninja without an install step, so this ALL target is what fills the
# wheel's torch_npu/fxrt, mirroring torchair's install_torchair target.
# ---------------------------------------------------------------------------
file(MAKE_DIRECTORY ${FXRT_INSTALL_DST})
# Operator libraries are copied below and deleted from the global lib dir
# afterwards, so the copy has to wait for every one of them to be linked.
set(FXRT_OP_LIBS ops_cpu_aten ops_cpu_symbolic ops_ascend_aclnn ops_ascend_aclnn_common
                 ops_ascend_composite ops_ascend_custom ops_ascend_hccl ops_ascend_mem)
set(FXRT_DVM_V2_LIB "")
if(TARGET ops_ascend_dvm_v2)
  list(APPEND FXRT_OP_LIBS ops_ascend_dvm_v2)
  set(FXRT_DVM_V2_LIB ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libops_ascend_dvm_v2.so)
endif()
# fxrt ships no C++ headers: it offers no C++ interface for now.
add_custom_target(install_fxrt ALL
        # Start from an empty staging dir, so files moved or deleted in the
        # source tree do not linger from a previous build
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${FXRT_INSTALL_DST}
        # Copy Python source tree, without bytecode caches
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/python/fxrt ${FXRT_INSTALL_DST}
        COMMAND find ${FXRT_INSTALL_DST} -name __pycache__ -prune -exec rm -rf {} +
        # Runtime shared libraries -> fxrt/lib/ (loaded by fxrt.ops.load)
        COMMAND ${CMAKE_COMMAND} -E make_directory ${FXRT_INSTALL_DST}/lib
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libfxrt.so
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libfxrt_common.so
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libfxrt_config.so
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libfxrt_profiler.so
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libfxrt_aclgraph_config.so
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libhardware_abstract.so
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libhardware_ascend.so
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libhardware_cpu.so
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libexecution_engine.so
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libkernel.so
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libops_utils.so
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libops_cpu_aten.so
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libops_cpu_symbolic.so
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libops_ascend_aclnn.so
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libops_ascend_aclnn_common.so
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libops_ascend_composite.so
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libops_ascend_custom.so
                ${FXRT_DVM_V2_LIB}
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libops_ascend_hccl.so
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libops_ascend_mem.so
                ${FXRT_INSTALL_DST}/lib
        # nanobind extension modules -> package root (importable as fxrt._fxrt_api)
        COMMAND bash -c "cp ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/_fxrt_api*.so ${FXRT_INSTALL_DST}/"
        COMMAND bash -c "cp ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/_fxrt_ir*.so ${FXRT_INSTALL_DST}/"
        COMMAND bash -c "cp ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/_fxrt_collective*.so ${FXRT_INSTALL_DST}/"
        COMMAND bash -c "cp ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/_fxrt_config*.so ${FXRT_INSTALL_DST}/"
        COMMAND bash -c "cp ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/_fxrt_aclgraph_config*.so ${FXRT_INSTALL_DST}/"
        COMMAND bash -c "cp ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/_fxrt_torch*.so ${FXRT_INSTALL_DST}/"
        # fxrt artifacts were redirected into the global lib dir by
        # CMAKE_LIBRARY_OUTPUT_DIRECTORY; remove them so the wheel does not
        # ship a duplicate copy under torch_npu/lib/.
        COMMAND bash -c "rm -f ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/_fxrt_*.so ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libfxrt*.so ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libhardware_abstract.so ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libhardware_ascend.so ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libhardware_cpu.so ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libexecution_engine.so ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libkernel.so ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libops_*.so ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libfxrt_ir_obj.a"
        DEPENDS fxrt _fxrt_api _fxrt_ir _fxrt_collective _fxrt_config _fxrt_aclgraph_config _fxrt_torch
                hardware_abstract hardware_cpu hardware_ascend execution_engine kernel ops_utils fxrt_profiler fxrt_config fxrt_aclgraph_config fxrt_common
                ${FXRT_OP_LIBS}
        VERBATIM)
