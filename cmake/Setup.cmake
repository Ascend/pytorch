# Replaces the -D flags that setup.py's CPPLibBuild used to pass to CMake, plus
# the codegen/stub steps it ran before invoking CMake. Included at the top of
# CMakeLists.txt right after find_package(Python3).

# --- Install layout -----------------------------------------------------------
# The legacy wheel ships compiled libraries under torch_npu/lib. Force "lib"
# (not lib64, which GNUInstallDirs selects on some platforms).
include(GNUInstallDirs)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE PATH "Library directory (torch_npu/lib)" FORCE)

# Compiled .so/.a land in a build-dir lib tree. The legacy setup.py passes
# -DTORCHNPU_INSTALL_LIBDIR=<build>/packages/torch_npu/lib; only default to
# <build>/lib (for the scikit-build-core path) when it is not already defined.
if(NOT DEFINED TORCHNPU_INSTALL_LIBDIR)
  set(TORCHNPU_INSTALL_LIBDIR "${CMAKE_BINARY_DIR}/lib")
endif()

# An installed torch_npu registers a torch.backends entrypoint, so any Python
# subprocess below that does `import torch` would otherwise sideload it (and
# fail if that install is broken). Disable autoload for the whole configure
# step, mirroring the guard that used to sit at the top of setup.py.
set(ENV{TORCH_DEVICE_BACKEND_AUTOLOAD} "0")

# --- Version ------------------------------------------------------------------
# TORCH_VERSION is read by CMakeLists.txt and by op-plugin's version parsing.
execute_process(
  COMMAND "${Python3_EXECUTABLE}"
          "${PROJECT_SOURCE_DIR}/tools/setup_helpers/prebuild.py" --version
  OUTPUT_VARIABLE TORCHNPU_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _torchnpu_version_result
  ERROR_QUIET)
if(NOT _torchnpu_version_result EQUAL 0)
  message(FATAL_ERROR "Failed to resolve torch_npu version")
endif()
set(TORCH_VERSION "${TORCHNPU_VERSION}")

# --- Installed PyTorch directory ---------------------------------------------
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import os, torch; print(os.path.dirname(os.path.realpath(torch.__file__)))"
  OUTPUT_VARIABLE PYTORCH_INSTALL_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pytorch_dir_result
  ERROR_QUIET)
if(NOT _pytorch_dir_result EQUAL 0)
  message(FATAL_ERROR "Cannot find the installed PyTorch directory")
endif()

set(PYTHON_INCLUDE_DIR "${Python3_INCLUDE_DIRS}")

# --- Environment variable forwarding (mirrors setup.py CPPLibBuild) -----------

# DISABLE_INSTALL_TORCHAIR / DISABLE_INSTALL_FXRT / DISABLE_RPC_FRAMEWORK are
# "disable" flags: unset or "FALSE" means build, anything else disables.
set(_disable_torchair FALSE)
if(DEFINED ENV{DISABLE_INSTALL_TORCHAIR} AND NOT "$ENV{DISABLE_INSTALL_TORCHAIR}" STREQUAL "FALSE")
  set(_disable_torchair TRUE)
endif()
set(_disable_fxrt FALSE)
if(DEFINED ENV{DISABLE_INSTALL_FXRT} AND NOT "$ENV{DISABLE_INSTALL_FXRT}" STREQUAL "FALSE")
  set(_disable_fxrt TRUE)
endif()
set(_disable_rpc FALSE)
if(DEFINED ENV{DISABLE_RPC_FRAMEWORK} AND NOT "$ENV{DISABLE_RPC_FRAMEWORK}" STREQUAL "FALSE")
  set(_disable_rpc TRUE)
endif()

if(NOT _disable_torchair)
  file(GLOB _torchair_files "${PROJECT_SOURCE_DIR}/third_party/torchair/torchair/*")
  if(_torchair_files)
    set(BUILD_TORCHAIR ON)
    set(TORCHAIR_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/packages/torch_npu/dynamo/torchair")
    set(TORCHAIR_TARGET_PYTHON "${Python3_EXECUTABLE}")
  endif()
endif()

if(NOT _disable_fxrt)
  file(GLOB _fxrt_files "${PROJECT_SOURCE_DIR}/fxrt/fxrt/*")
  if(_fxrt_files)
    set(BUILD_FXRT ON)
    set(FXRT_INSTALL_DST "${CMAKE_BINARY_DIR}/packages/torch_npu/fxrt")
    set(FXRT_TARGET_PYTHON "${Python3_EXECUTABLE}")
  endif()
endif()

if(NOT _disable_rpc AND EXISTS "${PROJECT_SOURCE_DIR}/third_party/Tensorpipe/tensorpipe")
  set(BUILD_TENSORPIPE ON)
endif()

if(DEFINED ENV{ENABLE_LTO} AND "$ENV{ENABLE_LTO}" STREQUAL "TRUE")
  set(ENABLE_LTO ON)
endif()

if(DEFINED ENV{PGO_MODE} AND NOT "$ENV{PGO_MODE}" STREQUAL "" AND NOT "$ENV{PGO_MODE}" STREQUAL "0")
  set(PGO_MODE "$ENV{PGO_MODE}")
endif()

# Default to the c++11 ABI (setup.py passed -DGLIBCXX_USE_CXX11_ABI=1 by
# default); an explicit -DGLIBCXX_USE_CXX11_ABI=0 or the env var
# _GLIBCXX_USE_CXX11_ABI=0 opts out. Only set the default when neither the
# command line nor CMakeLists.txt has defined the variable.
if(NOT DEFINED GLIBCXX_USE_CXX11_ABI)
  set(GLIBCXX_USE_CXX11_ABI 1)
endif()
if(DEFINED ENV{_GLIBCXX_USE_CXX11_ABI} AND "$ENV{_GLIBCXX_USE_CXX11_ABI}" STREQUAL "0")
  set(GLIBCXX_USE_CXX11_ABI 0)
endif()

if(DEFINED ENV{_ABI_VERSION})
  set(ABI_VERSION "$ENV{_ABI_VERSION}")
endif()

# --- Staging directories for subproject file copies ---------------------------
# torchair/fxrt stage Python sources into CMAKE_BINARY_DIR at build time; the
# install(DIRECTORY ...) rules below copy them into the install tree at install
# time. Create the dirs now so install(DIRECTORY) accepts them at configure time.
file(MAKE_DIRECTORY
  "${CMAKE_BINARY_DIR}/packages/torch_npu/dynamo/torchair"
  "${CMAKE_BINARY_DIR}/packages/torch_npu/fxrt")

# --- Pre-configure codegen + ACL stubs ----------------------------------------
# Runs generate_code.sh (which must complete before the FILE(GLOB ...) calls in
# the aten/ subdirectory collect the generated sources) and builds the ACL stub
# libraries that libtorch_npu links against.
execute_process(
  COMMAND "${Python3_EXECUTABLE}"
          "${PROJECT_SOURCE_DIR}/tools/setup_helpers/prebuild.py"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  RESULT_VARIABLE _prebuild_result)
if(NOT _prebuild_result EQUAL 0)
  message(FATAL_ERROR "Pre-configure steps (codegen/stub) failed")
endif()
