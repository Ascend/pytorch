# FxConverter class-1: generic OPP dual-stage Convert
#   mrt::ir (ABI=0) -> ExecuteGraph -> ModelV2
#
# Headers are prepared from pinned source revisions under third_party. CANN
# link stubs are built locally; installed CANN is required only when executing.

cmake_minimum_required(VERSION 3.12)
project(gert_backend LANGUAGES CXX)
# This directory used to configure in a separate CMake build. Keep FXRT and
# torch-npu directory properties from leaking into this ABI-isolated project.
foreach(_property INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS
                  LINK_DIRECTORIES LINK_OPTIONS)
  set_property(DIRECTORY PROPERTY "${_property}" "")
endforeach()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include cstdint")
endif()
# The converter is a separate CMake project nested in the FXRT build, so it must
# declare the language level required by the FxRT IR headers itself. In particular,
# intrusive_ptr.h uses C++17 variable templates such as std::is_base_of_v.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "fx_converter CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")

# All header and link inputs live in the checkout; configuration must not probe
# an installed CANN package or import torch_npu (which loads CANN at import time).
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/converter_deps.cmake)
add_subdirectory(${FXC_THIRD_PARTY_DIR}/cann_stub ${CMAKE_CURRENT_BINARY_DIR}/cann_stub)

# fxrt/src and fxrt/python/pybind each set CMAKE_BUILD_TYPE with a plain set(), which does not
# reach this sibling directory; with it empty CMake applies no per-config flags and every
# fx_converter target builds at -O0.
if("${DEBUG}" STREQUAL "on")
set(CMAKE_BUILD_TYPE "Debug")
else()
set(CMAKE_BUILD_TYPE "Release")
endif()
message("-- fx_converter CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# GE host libs are ABI=0; Torch/FxRT main remain ABI=1.
add_compile_options(-U_GLIBCXX_USE_CXX11_ABI -D_GLIBCXX_USE_CXX11_ABI=0)
string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
add_compile_definitions(
  nlohmann=ascend_nlohmann
)

if(NOT TARGET nanobind-static AND NOT COMMAND nanobind_add_module)
  # This is a Python extension, so only module headers/targets are needed.
  # Wheel build images may export the C API from Python without shipping libpython.
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
                  OUTPUT_VARIABLE nanobind_DIR
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(nanobind CONFIG REQUIRED)
endif()

# Resolves the sibling FXRT core and checks the
# IR export schema version. Must come before mrt_ir_abi0, which compiles those sources.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/torch_npu_fxrt_core.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/mrt_ir_abi0.cmake)

# The outer build supplies its source tree and torch_npu target. Do not import
# torch_npu or inspect an installed torch-npu wheel during packaging.
foreach(_input FXC_TORCH_ROOT FXC_TORCH_NPU_SOURCE_DIR)
  if(NOT DEFINED ${_input} OR NOT EXISTS "${${_input}}")
    message(FATAL_ERROR "In-tree converter requires ${_input}: '${${_input}}'")
  endif()
endforeach()
if(NOT TARGET torch_npu)
  message(FATAL_ERROR "In-tree converter requires the torch_npu target")
endif()
set(_FXC_TORCH_ROOT "${FXC_TORCH_ROOT}")
set(_FXC_TORCH_NPU_ROOT "${FXC_TORCH_NPU_SOURCE_DIR}")

# One entry per directory. CONFIGURE_DEPENDS picks up a new .cc on the next configure,
# which is what a hand-maintained list missed (concat_dual_stage_impl.cc).
set(_FXC_SOURCE_DIRS
  src/api
  src/common
  src/ir_import
  src/mrt_ge
  src/register
  src/lowering
  src/lowering/common
  src/lowering/dual_stage
  src/lowering/torch_fallback
  src/lowering/aicore
  src/opp
  src/shape
  src/exe_graph
  src/model_v2
)
set(FX_CONVERTER_SOURCES)
foreach(_fxc_dir IN LISTS _FXC_SOURCE_DIRS)
  file(GLOB _fxc_dir_sources CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/${_fxc_dir}/*.cc")
  list(APPEND FX_CONVERTER_SOURCES ${_fxc_dir_sources})
endforeach()
list(SORT FX_CONVERTER_SOURCES)

add_library(mrt_fx_converter SHARED ${FX_CONVERTER_SOURCES})
# _FXRT_PKG_ROOT comes from torch_npu_fxrt_core.cmake (the torch_npu source tree).
# Include order: local modules first
# (api/ir_import/mrt_ge/lowering/opp/register/shape/exe_graph/model_v2),
# then FxRT IR sources, then local public headers and GE_HEADER fallback.
target_include_directories(mrt_fx_converter PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${_FXRT_PKG_ROOT}/src
  ${_FXRT_PKG_ROOT}/include
  # Public/package headers assembled from pinned source repositories.
  ${FXC_CANN_HEADER_DIR}/include
  ${FXC_CANN_HEADER_DIR}/include/ge
  ${FXC_VENDORED_INCLUDE_DIRS}
  ${FXC_CANN_HEADER_DIR}/pkg_inc
  ${FXC_CANN_HEADER_DIR}/pkg_inc/runtime
  ${FXC_CANN_HEADER_DIR}/pkg_inc/base
  ${FXC_CANN_HEADER_DIR}/pkg_inc/common
  ${FXC_CANN_HEADER_DIR}/pkg_inc/common/ge_common
  ${FXC_CANN_HEADER_DIR}/pkg_inc/profiling
  # ---- GE_HEADER fallback (headers missing from CANN public package) ----
  ${FXC_GE_INCLUDE_DIR}/base
  ${FXC_GE_INCLUDE_DIR}
  ${FXC_GE_INCLUDE_DIR}/graph_metadef
  ${FXC_GE_INCLUDE_DIR}/graph_metadef/graph
  ${FXC_GE_INCLUDE_DIR}/runtime/v2
  ${FXC_GE_INCLUDE_DIR}/inc
  ${FXC_GE_INCLUDE_DIR}/inc/external
  ${FXC_GE_INCLUDE_DIR}/inc/framework
  ${FXC_GE_INCLUDE_DIR}/inc/graph_metadef
  ${FXC_GE_INCLUDE_DIR}/inc/graph_metadef/external
  ${FXC_GE_INCLUDE_DIR}/inc/graph_metadef/exe_graph
  ${FXC_GE_INCLUDE_DIR}/inc/graph_metadef/common
  ${FXC_JSON_INCLUDE_DIR}
)

# The ABI=1 island. Source-level options are appended after the directory-level
# -D_GLIBCXX_USE_CXX11_ABI=0 set at the top of this file, so the later -D wins for this
# one translation unit only. Its includes are also scoped here: nothing else in the
# library may see torch_npu's headers.
set(_FXC_TORCH_POOL_INCS
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${_FXC_TORCH_NPU_ROOT}"
  "${_FXC_TORCH_NPU_ROOT}/third_party/acl/inc"
  "${_FXC_TORCH_NPU_ROOT}/third_party/hccl/inc"
  "${_FXC_TORCH_ROOT}/include"
  "${_FXC_TORCH_ROOT}/include/torch/csrc/api/include"
  ${Python_INCLUDE_DIRS})
set_source_files_properties(src/model_v2/torch_pool_allocator.cc PROPERTIES
  # torch 2.15 ATen headers require C++20. Keep that standard on this ABI=1 file
  # only; the rest of the library stays on the C++17 GE host ABI.
  COMPILE_OPTIONS "-U_GLIBCXX_USE_CXX11_ABI;-D_GLIBCXX_USE_CXX11_ABI=1;-std=c++20"
  INCLUDE_DIRECTORIES "${_FXC_TORCH_POOL_INCS}"
)
target_link_libraries(mrt_fx_converter PUBLIC
  mrt_ir_abi0
  -Wl,--no-as-needed
  # Keep CANN references external, including allocator RTTI. The link stubs
  # provide symbols and real SONAMEs without GE objects or static initializers.
  ${FXC_CANN_STUB_LIBRARIES}
  dl
  -Wl,--as-needed
)
# torch_npu for the ABI=1 island (torch_pool_allocator.cc). --no-as-needed keeps the
# DT_NEEDED entry: nothing in the ABI=0 part of this library references libtorch_npu
# symbols, but the ABI=1 object does, and dropping the entry would leave it unresolved.
# At runtime the library is already loaded -- fx_converter imports torch_npu before it
# touches the converter -- so this never triggers a second copy of the allocator.
target_link_directories(mrt_fx_converter PUBLIC
  "${_FXC_TORCH_ROOT}/lib"
)
set(_FXC_TORCH_POOL_LIBS "-Wl,--no-as-needed;$<TARGET_FILE:torch_npu>;torch;torch_cpu;c10;-Wl,--as-needed")
target_link_libraries(mrt_fx_converter PUBLIC ${_FXC_TORCH_POOL_LIBS})
add_dependencies(mrt_fx_converter torch_npu)
# torch_npu's own CANN dependencies are resolved at runtime. Do not require the
# build machine to supply its indirect DSOs (hccl, acl_op_compiler, etc.).
target_link_options(mrt_fx_converter PRIVATE -Wl,--allow-shlib-undefined)
# The invariant the link line above relies on, checked on the artifact instead of trusted to
# stay true: gert's allocator types must not be defined in this image. Both failure modes
# are silent at build time and fatal at run time (a foreign typeinfo crashes GE's
# SelectL1Allocator in __dynamic_cast; a second set of GE static initializers corrupts the
# heap at interpreter exit), so a build error is the only reliable place to catch them.
add_custom_command(TARGET mrt_fx_converter POST_BUILD
  COMMAND "${CMAKE_COMMAND}"
          "-DTARGET_SO=$<TARGET_FILE:mrt_fx_converter>"
          "-DNM=${CMAKE_NM}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_gert_typeinfo.cmake"
  COMMAND "${CMAKE_COMMAND}"
          "-DTARGET_SO=$<TARGET_FILE:mrt_fx_converter>"
          "-DNM=${CMAKE_NM}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_ge_compat.cmake"
  COMMAND "${CMAKE_COMMAND}"
          "-DTARGET_SO=$<TARGET_FILE:mrt_fx_converter>"
          "-DNM=${CMAKE_NM}" "-DREADELF=${CMAKE_READELF}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_aclnn_dependencies.cmake"
  VERBATIM
)
set_target_properties(mrt_fx_converter PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/../../../lib:$ORIGIN/../../../../torch/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)
install(TARGETS mrt_fx_converter LIBRARY DESTINATION lib RUNTIME DESTINATION bin)

nanobind_add_module(_fxrt_fx_converter api/pybind_fx_converter.cc)
target_link_libraries(_fxrt_fx_converter PRIVATE mrt_fx_converter)
target_compile_definitions(_fxrt_fx_converter PRIVATE
  FXC_EXPECTED_IR_SCHEMA_VERSION=${FXC_EXPECTED_IR_SCHEMA_VERSION})
target_include_directories(_fxrt_fx_converter PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${_FXRT_PKG_ROOT}/src
)
set_target_properties(_fxrt_fx_converter PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)
target_compile_options(_fxrt_fx_converter PRIVATE -U_GLIBCXX_USE_CXX11_ABI -D_GLIBCXX_USE_CXX11_ABI=0)
target_compile_definitions(_fxrt_fx_converter PRIVATE
  nlohmann=ascend_nlohmann
)
install(TARGETS _fxrt_fx_converter LIBRARY DESTINATION . RUNTIME DESTINATION bin)

# Check every strong import against stubs and the actual host/Python/torch export
# tables. This catches a missing stub when a new API is added, without loading
# torch_npu or any CANN library on the build machine.
add_custom_command(TARGET _fxrt_fx_converter POST_BUILD
  COMMAND "${Python_EXECUTABLE}" "${FXC_THIRD_PARTY_DIR}/cann_stub/check_link.py"
          --artifact "$<TARGET_FILE:mrt_fx_converter>"
          --artifact "$<TARGET_FILE:_fxrt_fx_converter>"
          --provider "$<TARGET_FILE:mrt_fx_converter>"
          --provider-dir "${CMAKE_CURRENT_BINARY_DIR}/cann_stub/lib"
          --provider-dir "${_FXC_TORCH_ROOT}/lib"
          --provider "${FXC_TORCH_NPU_LIBRARY}"
          --compiler "${CMAKE_CXX_COMPILER}" --nm "${CMAKE_NM}"
  VERBATIM
)


# Python sources and native artifacts are staged only into the converter package.
install(DIRECTORY python/gert_backend/ DESTINATION .
        FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)

# setup.py builds the package without running `cmake --install`. Stage converter
# files after install_fxrt, which recreates FXRT_INSTALL_DST on every build.
set(_FXC_INSTALL_DST "${FXRT_INSTALL_DST}/gert_backend")
add_custom_target(install_gert_backend ALL
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${_FXC_INSTALL_DST}/lib"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/python/gert_backend" "${_FXC_INSTALL_DST}"
  COMMAND find "${_FXC_INSTALL_DST}" -name __pycache__ -type d -prune -exec rm -rf {} +
  COMMAND "${CMAKE_COMMAND}" -E copy
          "$<TARGET_FILE:_fxrt_fx_converter>" "${_FXC_INSTALL_DST}"
  COMMAND "${CMAKE_COMMAND}" -E copy
          "$<TARGET_FILE:mrt_fx_converter>" "${_FXC_INSTALL_DST}/lib"
  DEPENDS install_fxrt _fxrt_fx_converter mrt_fx_converter
  VERBATIM
)
