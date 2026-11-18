cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

find_program(CCACHE ccache)
if(${CCACHE} STREQUAL "CCACHE-NOTFOUND")
  message(STATUS "Compile without ccache")
else()
  set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE} CACHE PATH "cache Compiler")
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE} CACHE PATH "cache Compiler")
  message(STATUS "CMAKE_C_COMPILER_LAUNCHER:${CMAKE_C_COMPILER_LAUNCHER}")
  message(STATUS "CMAKE_CXX_COMPILER_LAUNCHER:${CMAKE_CXX_COMPILER_LAUNCHER}")
endif()

project(TORCHNPU CXX C)
add_compile_options(-fmacro-prefix-map=${CMAKE_SOURCE_DIR}/=)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Forward setup.py's -D flags and run the pre-configure codegen/stub steps.
# Only for the scikit-build-core path (SKBUILD is injected by scikit-build-core).
# The legacy setup.py already passes the same -D flags and runs codegen/stub
# itself, so including Setup.cmake there would re-run codegen (and shadow -D).
if (SKBUILD)
  include(${PROJECT_SOURCE_DIR}/cmake/Setup.cmake)
endif()

set(PUBLIC_ENV_CHECK_SCRIPT ${PROJECT_SOURCE_DIR}/tools/check_public_environments.py)
set(PUBLIC_ENV_SCHEMA ${PROJECT_SOURCE_DIR}/test/torch_npu_schema.json)
set(PUBLIC_ENV_SOURCE ${PROJECT_SOURCE_DIR}/torch_npu/csrc/core/npu/register/OptionsManager.cpp)
add_custom_target(check_public_environments
  COMMAND ${Python3_EXECUTABLE} ${PUBLIC_ENV_CHECK_SCRIPT}
    --schema ${PUBLIC_ENV_SCHEMA}
    --source ${PUBLIC_ENV_SOURCE}
  DEPENDS ${PUBLIC_ENV_CHECK_SCRIPT} ${PUBLIC_ENV_SCHEMA} ${PUBLIC_ENV_SOURCE}
  COMMENT "Checking public environment variable modes"
  VERBATIM)

find_program(MOLD_LINKER mold)
if(MOLD_LINKER)
  add_link_options(-fuse-ld=mold)
  message(STATUS "Using mold linker: ${MOLD_LINKER}")
else()
  message(STATUS "mold linker not found, using default linker")
endif()

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(DEFINED TORCH_VERSION)
  add_definitions(-DPYTORCH_NPU_VERSION="${TORCH_VERSION}")
endif()

set(PLUGIN_NAME torch_npu)

set(RPATH_VALUE $ORIGIN)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${RPATH_VALUE}/lib/:${RPATH_VALUE}/")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${TORCHNPU_INSTALL_LIBDIR})
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
SET(CMAKE_CXX_FLAGS_RELEASE "-O2")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

option(USE_NPU "Use NPU" ON)

# LTO&PGO optimization in compile option
SET(IF_APPEND FALSE)
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  SET(APPEND_FLAGS "-fGNU-compatibility -Wno-non-pod-varargs")
  if(NOT MOLD_LINKER)
    add_link_options(-fuse-ld=lld)
  endif()
  if (DEFINED ENABLE_LTO)
    SET(IF_APPEND TRUE)
    SET(APPEND_FLAGS "${APPEND_FLAGS} -flto=thin")
  endif()
  if (DEFINED PGO_MODE)
    SET(IF_APPEND TRUE)
    if (PGO_MODE EQUAL 1)
      SET(APPEND_FLAGS "${APPEND_FLAGS} -fprofile-generate")
    elseif (PGO_MODE EQUAL 2)
      SET(APPEND_FLAGS "${APPEND_FLAGS} -fprofile-use=${CMAKE_CURRENT_SOURCE_DIR}/default.profdata")
    endif()
  endif()
else()
  if (DEFINED ENABLE_LTO)
    message(FATAL_ERROR "Currently, LTO auto build is not supported in ${CMAKE_CXX_COMPILER_ID}")
  endif()
  if (DEFINED PGO_MODE)
    message(FATAL_ERROR "Currently, PGO auto build is not supported in ${CMAKE_CXX_COMPILER_ID}")
  endif()
endif()
if (IF_APPEND)
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${APPEND_FLAGS}")
endif()

# check and set CMAKE_CXX_STANDARD
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)
if(env_cxx_standard GREATER -1)
  message(
      WARNING "C++ standard version definition detected in environment variable."
      "PyTorch requires -std=c++20. Please remove -std=c++ settings in your environment.")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TORCHNPU_ROOT "${PROJECT_SOURCE_DIR}/torch_npu/csrc")
set(TORCHNPU_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")

set(Torch_DIR ${PYTORCH_INSTALL_DIR}/share/cmake/Torch)
FIND_PACKAGE(Torch REQUIRED)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
#set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Define build type
IF(CMAKE_BUILD_TYPE MATCHES Debug)
  message("Debug build.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_DEBUG")
ELSEIF(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  message("RelWithDebInfo build")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
ELSE()
  message("Release build.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
ENDIF()

if(USE_NPU)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_NPU")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")
# Eigen fails to build with some versions, so convert this to a warning
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-field-initializers")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-array-bounds")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-overflow")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-aliasing")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-qualifiers")
if (CMAKE_COMPILER_IS_GNUCXX AND NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0.0))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-overflow")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=redundant-decls")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=old-style-cast")

# These flags are not available in GCC-4.8.5. Set only when using clang.
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-invalid-partial-specialization")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-typedef-redefinition")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-warning-option")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-inconsistent-missing-override")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-aligned-allocation-unavailable")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++20-extensions")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-constexpr-not-const")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Qunused-arguments")
  if (${COLORIZE_OUTPUT})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
  endif()
endif()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
  if (${COLORIZE_OUTPUT})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
  endif()
endif()
if ((APPLE AND (NOT ("${CLANG_VERSION_STRING}" VERSION_LESS "9.0")))
  OR (CMAKE_COMPILER_IS_GNUCXX
  AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0 AND NOT APPLE)))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -faligned-new")
endif()
if (WERROR)
  check_cxx_compiler_flag("-Werror" COMPILER_SUPPORT_WERROR)
  if (NOT COMPILER_SUPPORT_WERROR)
    set(WERROR FALSE)
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
endif(WERROR)
if (NOT APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-uninitialized")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -O0")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_STATIC_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-math-errno")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-trapping-math")
set(CMAKE_CXX_COVERAGE $ENV{CALCULATE_CXX_COVERAGE})

if (CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_C_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CMAKE_CXX_FLAGS}")
  set(CXXFLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CXXFLAGS}")
elseif (CMAKE_CXX_COVERAGE STREQUAL "1")
  set(CMAKE_C_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fprofile-arcs -ftest-coverage -fPIE -pie ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fprofile-arcs -ftest-coverage -fPIE -pie ${CMAKE_CXX_FLAGS}")
  set(CXXFLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fprofile-arcs -ftest-coverage -fPIE -pie ${CXXFLAGS}")
else()
  set(CMAKE_C_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CMAKE_CXX_FLAGS}")
  set(CXXFLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CXXFLAGS}")
endif()

if (NOT DEFINED GLIBCXX_USE_CXX11_ABI)
  set(GLIBCXX_USE_CXX11_ABI 0)
endif()
message(STATUS "Determined _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")
set(_GLIBCXX_USE_CXX11_ABI ${GLIBCXX_USE_CXX11_ABI})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")
if (${GLIBCXX_USE_CXX11_ABI} EQUAL 0)
  set(CMAKE_CXX_FLAGS "-fabi-version=11 ${CMAKE_CXX_FLAGS}")
else()
  set(CXX_STANDARD_REQUIRED ON)
  if (DEFINED ABI_VERSION)
    set(CMAKE_CXX_FLAGS "-fabi-version=${ABI_VERSION} ${CMAKE_CXX_FLAGS}")
  endif()
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/torch_npu/csrc/aten)
include_directories(${PROJECT_SOURCE_DIR}/torch_npu/csrc/inductor)
include_directories(${PROJECT_SOURCE_DIR}/third_party/hccl/inc)
include_directories(${PROJECT_SOURCE_DIR}/third_party/acl/inc)
include_directories(${PROJECT_SOURCE_DIR}/third_party/Tensorpipe)
include_directories(${PROJECT_SOURCE_DIR}/third_party/nlohmann/include)

# Set installed PyTorch dir
if(DEFINED PYTORCH_INSTALL_DIR)
  include_directories(${PYTORCH_INSTALL_DIR}/include)
  include_directories(${PYTORCH_INSTALL_DIR}/include/kineto)
  include_directories(${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include)
  include_directories(${PYTORCH_INSTALL_DIR}/include/torch/csrc/distributed)
else()
  message(FATAL_ERROR "Cannot find installed PyTorch directory")
endif()

# Set Python include dir
if(DEFINED PYTHON_INCLUDE_DIR)
  include_directories(${PYTHON_INCLUDE_DIR})
else()
  message(FATAL_ERROR "Cannot find installed Python head file directory")
endif()

# sources
set(ATEN_SRCS)
set(CORE_SRCS)
set(FRAMEWORK_SRCS)
set(LOGGING_SRCS)
set(INDUCTOR_SRCS)
set(DIST_SRCS)
set(FLOP_SRCS)
set(NPU_SRCS)
set(PROF_SRCS)
set(IPC_SRCS)
set(UTILS_SRCS)
set(SAN_SRCS)
set(AFD_SRCS)
set(NPU_CPP_LIBS_SRCS)

add_subdirectory(${TORCHNPU_ROOT}/aten)
add_subdirectory(${TORCHNPU_ROOT}/core)
add_subdirectory(${TORCHNPU_ROOT}/framework)
add_subdirectory(${TORCHNPU_ROOT}/flopcount)
add_subdirectory(${TORCHNPU_ROOT}/logging)
add_subdirectory(${TORCHNPU_ROOT}/custom_dtype)
add_subdirectory(${TORCHNPU_ROOT}/inductor)
add_subdirectory(${TORCHNPU_ROOT}/distributed)
add_subdirectory(${TORCHNPU_ROOT}/npu)
add_subdirectory(${TORCHNPU_ROOT}/profiler)
add_subdirectory(${TORCHNPU_ROOT}/ipc)
add_subdirectory(${TORCHNPU_ROOT}/utils)
add_subdirectory(${TORCHNPU_ROOT}/sanitizer)
add_subdirectory(${TORCHNPU_ROOT}/afd)
add_subdirectory(${TORCHNPU_ROOT}/libs)

set(OPS_PLUGIN_SRCS)
# Add subdirectory of op-plugin
set(TORCH_NPU_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
include_directories(${PROJECT_SOURCE_DIR}/third_party/op-plugin)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/op-plugin/op_plugin)
if (NOT DEFINED BUILD_LIBTORCH)
  include("${PROJECT_SOURCE_DIR}/cmake/NPUSymmetricMemory.cmake")
endif()

if (DEFINED BUILD_TENSORPIPE)
  add_definitions(-DUSE_RPC_FRAMEWORK)
  set(BUILD_SHARED_LIBS ON)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0.0")
    message(WARNING "tensorpipe forces CMake compatibility")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
  endif()
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/Tensorpipe)
  # Suppress warning to unblock libnop compilation by clang-17+
  # See https://github.com/pytorch/pytorch/issues/151316
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-Wno-missing-template-arg-list-after-template-kw"
      COMPILER_SUPPORTS_NO_MISSING_TEMPLATE_ARG_LIST)
  if(COMPILER_SUPPORTS_NO_MISSING_TEMPLATE_ARG_LIST)
    target_compile_options(tensorpipe PRIVATE -Wno-missing-template-arg-list-after-template-kw)
  endif()
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0.0")
    unset(CMAKE_POLICY_VERSION_MINIMUM)
  endif()
  set(BUILD_SHARED_LIBS OFF)
endif()


set(CPP_SRCS ${ATEN_SRCS} ${INDUCTOR_SRCS} ${CORE_SRCS} ${OPS_PLUGIN_SRCS} ${DIST_SRCS} ${FLOP_SRCS} ${CUS_DTYPE_SRCS} ${LOGGING_SRCS} ${FRAMEWORK_SRCS} ${NPU_SRCS} ${PROF_SRCS} ${IPC_SRCS} ${UTILS_SRCS} ${SAN_SRCS} ${AFD_SRCS} ${NPU_CPP_LIBS_SRCS})

# --- Python/C++ source split ---
# Files moved out of the core source list into the torch_npu_python target.
# Keep this list explicit: any new binding file defaults to the python target,
# while files required by core TUs must stay in the core list
set(TORCH_NPU_PYTHON_SRCS
    ${TORCHNPU_ROOT}/npu/Module.cpp
    ${TORCHNPU_ROOT}/npu/Stream.cpp
    ${TORCHNPU_ROOT}/npu/Event.cpp
    ${TORCHNPU_ROOT}/npu/Graph.cpp
    ${TORCHNPU_ROOT}/npu/npurt.cpp
    ${TORCHNPU_ROOT}/npu/DataParallelComm.cpp
    ${TORCHNPU_ROOT}/npu/Stress_detect.cpp
    ${TORCHNPU_ROOT}/npu/NpuSleep.cpp
    ${TORCHNPU_ROOT}/npu/lazy_fusion.cpp
    ${TORCHNPU_ROOT}/profiler/init.cpp
    ${TORCHNPU_ROOT}/profiler/python/combined_traceback.cpp
    ${TORCHNPU_ROOT}/profiler/profiler_python.cpp
    ${TORCHNPU_ROOT}/ipc/StorageSharing.cpp
    ${TORCHNPU_ROOT}/utils/TensorType.cpp
    ${TORCHNPU_ROOT}/utils/AutocastMode.cpp
    ${TORCHNPU_ROOT}/afd/Init.cpp
    ${TORCHNPU_ROOT}/afd/ScheduleContext.cpp
    ${TORCHNPU_ROOT}/distributed/Init.cpp
    ${TORCHNPU_ROOT}/distributed/rpc/init.cpp
    ${TORCHNPU_ROOT}/distributed/PythonDebugAdapter.cpp
    ${TORCHNPU_ROOT}/utils/PythonGilAdapter.cpp
    ${TORCHNPU_ROOT}/utils/PythonOomAdapter.cpp
    ${TORCHNPU_ROOT}/custom_dtype/Init.cpp
    ${TORCHNPU_ROOT}/flopcount/Init.cpp
    ${TORCHNPU_ROOT}/logging/Init.cpp
    ${TORCHNPU_ROOT}/core/npu/NPURecoveryBindings.cpp
    ${TORCHNPU_ROOT}/npu/MemPool.cpp
    ${TORCHNPU_ROOT}/aten/python_functions_0.cpp
    ${TORCHNPU_ROOT}/aten/python_functions_1.cpp
    ${TORCHNPU_ROOT}/sanitizer/PyCallbackTrigger.cpp
    ${TORCHNPU_ROOT}/utils/LazyInit.cpp
    ${TORCHNPU_ROOT}/inductor/aoti_runner/pybind.cpp
    ${TORCHNPU_ROOT}/inductor/aoti_package/shape_handling.cpp
    ${TORCHNPU_ROOT}/inductor/dvm/pybind_api.cpp
    ${TORCHNPU_ROOT}/inductor/mlir/cpp_common.cpp
    ${TORCHNPU_ROOT}/inductor/mlir/mlir_binding.cpp
    ${TORCHNPU_ROOT}/inductor/static_launcher/runtime.cpp
    ${TORCHNPU_ROOT}/inductor/static_launcher/fast_launcher.cpp
    ${TORCHNPU_ROOT}/inductor/static_launcher/bindings.cpp
    ${PROJECT_SOURCE_DIR}/torch_npu/_inductor/experimental/python_wrapper_fast_launch/csrc/bindings.cpp)

if (DEFINED BUILD_LIBTORCH)
  list(REMOVE_ITEM CPP_SRCS ${TORCH_NPU_PYTHON_SRCS})
endif()

add_library(torch_npu_objs OBJECT ${CPP_SRCS})
add_dependencies(torch_npu_objs check_public_environments)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fvisibility=hidden" COMPILER_SUPPORTS_HIDDEN_VISIBILITY)
if(${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
  target_compile_options(torch_npu_objs PRIVATE "-fvisibility=hidden")
endif()

if (DEFINED BUILD_TORCHAIR)
  add_subdirectory(${TORCHNPU_THIRD_PARTY_ROOT}/torchair)
endif()

add_subdirectory(${TORCHNPU_THIRD_PARTY_ROOT}/fmt EXCLUDE_FROM_ALL)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/dvm)

# Propagate compile definitions and include dirs from PyTorch to the object library
target_link_libraries(torch_npu_objs PUBLIC torch torch_cpu c10 fmt::fmt-header-only)

set(ATEN_THREADING "OMP" CACHE STRING "ATen parallel backend")
message(STATUS "Using ATen parallel backend: ${ATEN_THREADING}")
if ("${ATEN_THREADING}" STREQUAL "OMP")
  target_compile_definitions(torch_npu_objs PUBLIC "-DAT_PARALLEL_OPENMP=1")
elseif ("${ATEN_THREADING}" STREQUAL "NATIVE")
  target_compile_definitions(torch_npu_objs PUBLIC "-DAT_PARALLEL_NATIVE=1")
elseif ("${ATEN_THREADING}" STREQUAL "TBB")
  target_compile_definitions(torch_npu_objs PUBLIC "-DAT_PARALLEL_NATIVE_TBB=1")
else()
  message(FATAL_ERROR "Unknown ATen parallel backend: ${ATEN_THREADING}")
endif()

target_compile_options(torch_npu_objs PRIVATE "-DC10_BUILD_MAIN_LIB")

add_library(${PLUGIN_NAME} SHARED $<TARGET_OBJECTS:torch_npu_objs>)
target_link_libraries(${PLUGIN_NAME} PUBLIC torch_npu_objs)
target_link_options(${PLUGIN_NAME} PRIVATE "-Wl,-Bsymbolic-functions,--no-as-needed")

if (TARGET torch_npu_symm_mem_kernels)
  target_sources(${PLUGIN_NAME} PRIVATE "${_symm_mem_object}")
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${_symm_mem_runtime_libraries})
  set_property(SOURCE "${_symm_mem_source_dir}/NPUSHMEMSymmetricMemory.cpp"
    APPEND PROPERTY COMPILE_DEFINITIONS USE_NPU_SYMM_MEM)
  add_dependencies(${PLUGIN_NAME} torch_npu_symm_mem_kernels)
endif()

# Shared library: link step, dvm / torchair deps only gate linking, not compilation.
add_dependencies(${PLUGIN_NAME} dvm_build)
if (DEFINED BUILD_TORCHAIR)
  add_dependencies(${PLUGIN_NAME} copy_torchair_pyfiles)
endif()

target_link_libraries(${PLUGIN_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/third_party/dvm/dvm/libdvm.a)

link_directories(${PYTORCH_INSTALL_DIR}/lib)
link_directories(${TORCHNPU_THIRD_PARTY_ROOT}/acl/libs)

target_link_libraries(${PLUGIN_NAME} PUBLIC
  ${TORCHNPU_THIRD_PARTY_ROOT}/acl/libs/libhccl.so
  ${TORCHNPU_THIRD_PARTY_ROOT}/acl/libs/libascendcl.so
  ${TORCHNPU_THIRD_PARTY_ROOT}/acl/libs/libacl_op_compiler.so
  ${TORCHNPU_THIRD_PARTY_ROOT}/acl/libs/libge_runner.so
  ${TORCHNPU_THIRD_PARTY_ROOT}/acl/libs/libgraph.so)

if (NOT DEFINED BUILD_LIBTORCH)
  target_link_libraries(${PLUGIN_NAME} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libtorch_python.so)
endif()

if (DEFINED BUILD_TENSORPIPE)
  target_link_libraries(${PLUGIN_NAME} PUBLIC ${TORCHNPU_INSTALL_LIBDIR}/libtensorpipe.so)
endif()

if (NOT DEFINED BUILD_LIBTORCH)
  include(GNUInstallDirs)
  install(TARGETS ${PLUGIN_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

add_subdirectory(${TORCHNPU_ROOT}/toolkit)
target_link_libraries(${PLUGIN_NAME} PUBLIC npu_profiler)

# fxrt: built as a top-level subdirectory.
if (DEFINED BUILD_FXRT)
  set(ENABLE_ASCEND ON)
  set(ENABLE_CPU ON)
  set(ENABLE_TORCH_FRONT ON)
  add_subdirectory(${PROJECT_SOURCE_DIR}/fxrt/fxrt)
endif()

# ============================================================================
# torch_npu._C extension module + wheel install layout.
#
# Reproduces setup.py's CppExtension("torch_npu._C") and the file copies done
# by get_src_py_and_dst()/add_ops_files(). scikit-build-core maps the CMake
# install tree root onto wheel.install-dir ("torch_npu"), so every DESTINATION
# below is relative to torch_npu/ inside the wheel.
# ============================================================================
if (NOT DEFINED BUILD_LIBTORCH AND SKBUILD)

  # --- torch_npu._C (CPython extension module) --------------------------------
  # setup.py: CppExtension('torch_npu._C', sources=[InitNpuBindings.cpp],
  #   libraries=[torch_npu, c10, torch, torch_cpu, torch_python, hccl],
  #   extra_link_args=['-Wl,-rpath,$ORIGIN/lib', '-Wl,-Bsymbolic-functions'],
  #   define_macros=[_GLIBCXX_USE_CXX11_ABI, GLIBCXX_USE_CXX11_ABI]).
  Python3_add_library(_C MODULE WITH_SOABI
    ${TORCHNPU_ROOT}/InitNpuBindings.cpp)

  set_target_properties(_C PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    PREFIX "")

  # Link the exact six libraries the legacy CppExtension linked
  # (-ltorch_npu -lc10 -ltorch -ltorch_cpu -ltorch_python -lhccl). Full paths
  # are used so torch_npu's PUBLIC interface (ascendcl/ge_runner/tensorpipe/
  # npu_profiler/...) is NOT propagated into _C's DT_NEEDED, matching the legacy
  # _C.so exactly. add_dependencies keeps the build order correct.
  target_link_libraries(_C PRIVATE
    ${TORCHNPU_INSTALL_LIBDIR}/libtorch_npu.so
    ${PYTORCH_INSTALL_DIR}/lib/libtorch.so
    ${PYTORCH_INSTALL_DIR}/lib/libtorch_cpu.so
    ${PYTORCH_INSTALL_DIR}/lib/libc10.so
    ${PYTORCH_INSTALL_DIR}/lib/libtorch_python.so
    ${TORCHNPU_THIRD_PARTY_ROOT}/acl/libs/libhccl.so)
  add_dependencies(_C ${PLUGIN_NAME})

  target_link_options(_C PRIVATE
    "-Wl,-rpath,$ORIGIN/lib"
    "-Wl,-Bsymbolic-functions")

  target_compile_definitions(_C PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}
    GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI})

  target_compile_options(_C PRIVATE "-D__FILENAME__=\"InitNpuBindings.cpp\"")

  install(TARGETS _C LIBRARY DESTINATION .)

  # --- generated files --------------------------------------------------------
  # Produced by the pre-configure codegen (prebuild.py) and gitignored, so they
  # must be installed explicitly (the wheel.packages scan skips gitignored
  # files). Several are runtime-essential: custom_ops/exposed_api are imported
  # by torch_npu._init, and the codegen yaml + testing helper are read by
  # op-plugin's compatibility test and torch_npu's own tests.
  install(FILES
    ${PROJECT_SOURCE_DIR}/torch_npu/version.py
    ${PROJECT_SOURCE_DIR}/torch_npu/_op_plugin_docs.py
    DESTINATION .)
  install(FILES
    ${PROJECT_SOURCE_DIR}/torch_npu/utils/custom_ops.py
    ${PROJECT_SOURCE_DIR}/torch_npu/utils/exposed_api.py
    DESTINATION utils)
  install(FILES
    ${PROJECT_SOURCE_DIR}/torch_npu/testing/_npu_testing_utils.py
    DESTINATION testing)
  install(FILES
    ${PROJECT_SOURCE_DIR}/torch_npu/csrc/aten/npu_native_functions_by_codegen.yaml
    DESTINATION csrc/aten)

  # --- op_plugin python sources -> torch_npu/op_plugin/ -----------------------
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/third_party/op-plugin/op_plugin/python/
    DESTINATION op_plugin
    FILES_MATCHING PATTERN "*.py")

  # --- headers -> torch_npu/include/ ------------------------------------------
  # torch_npu/csrc: all .h (max depth 5 in-tree) + depth-2 .hpp only. The
  # depth-3 symm_mem .hpp files are dropped to match the legacy depth-2 glob.
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/torch_npu/csrc/
    DESTINATION include/torch_npu/csrc
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "distributed/symm_mem/*.hpp" EXCLUDE)

  # acl headers referenced as third_party/acl/inc/... by the csrc include group.
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/third_party/acl/inc/
    DESTINATION include/third_party/acl/inc
    FILES_MATCHING PATTERN "*.h")

  # hccl headers.
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/third_party/hccl/inc/
    DESTINATION include/third_party/hccl/inc
    FILES_MATCHING PATTERN "*.h")

  # op-plugin + dvm headers (depth-1 only in each of these dirs, like the
  # legacy globs, so use explicit GLOBs rather than a recursive copy).
  file(GLOB _op_plugin_include_h
    "${PROJECT_SOURCE_DIR}/third_party/op-plugin/op_plugin/include/*.h")
  file(GLOB _op_plugin_utils_h
    "${PROJECT_SOURCE_DIR}/third_party/op-plugin/op_plugin/utils/*.h")
  file(GLOB _op_plugin_dvm_h
    "${PROJECT_SOURCE_DIR}/third_party/op-plugin/op_plugin/ops/dvm/*.h")
  file(GLOB _dvm_include_h
    "${PROJECT_SOURCE_DIR}/third_party/dvm/dvm/include/*.h")
  install(FILES ${_op_plugin_include_h}
    DESTINATION include/third_party/op-plugin/op_plugin/include)
  install(FILES ${_op_plugin_utils_h}
    DESTINATION include/third_party/op-plugin/op_plugin/utils)
  install(FILES ${_op_plugin_dvm_h}
    DESTINATION include/third_party/op-plugin/op_plugin/ops/dvm)
  install(FILES ${_dvm_include_h}
    DESTINATION include/third_party/dvm/dvm/include)

  # acl headers (full recursive copy): include/<relpath from acl/inc>.
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/third_party/acl/inc/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h")

  # --- companion shared libraries -> torch_npu/lib/ ---------------------------
  install(TARGETS npu_profiler op_plugin_atb
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  if (DEFINED BUILD_TENSORPIPE)
    install(TARGETS tensorpipe LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()

  # --- op-plugin torchnpugen codegen -> torchnpugen/ (wheel root) -------------
  # The in-repo torchnpugen/* is picked up by wheel.packages; this merges in the
  # op-plugin/torchnpugen contribution. Install into ${SKBUILD_PLATLIB_DIR} (the
  # wheel root) so the files land next to the in-repo package at torchnpugen/,
  # not under torch_npu/torchnpugen/ (wheel.install-dir). Drop the op-plugin
  # root __init__.py so the in-repo torchnpugen/__init__.py wins; keep the
  # nested api/api.types/struct __init__.py files so those subpackages stay
  # importable.
  file(GLOB _torchnpugen_root_py
    "${PROJECT_SOURCE_DIR}/third_party/op-plugin/torchnpugen/*.py")
  list(REMOVE_ITEM _torchnpugen_root_py
    "${PROJECT_SOURCE_DIR}/third_party/op-plugin/torchnpugen/__init__.py")
  install(FILES ${_torchnpugen_root_py}
    DESTINATION "${SKBUILD_PLATLIB_DIR}/torchnpugen")
  install(DIRECTORY
      ${PROJECT_SOURCE_DIR}/third_party/op-plugin/torchnpugen/api
      ${PROJECT_SOURCE_DIR}/third_party/op-plugin/torchnpugen/struct
      ${PROJECT_SOURCE_DIR}/third_party/op-plugin/torchnpugen/templates
    DESTINATION "${SKBUILD_PLATLIB_DIR}/torchnpugen"
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.yaml"
    PATTERN "*.json"
    PATTERN "*.cpp"
    PATTERN "*.h")

  # --- torchair -> torch_npu/dynamo/ + torch_npu/_inductor/ --------------------
  # copy_torchair_pyfiles (ALL) stages torchair, npugraph_ex, and ascendc into
  # build/packages/torch_npu at build time; copy them into the install tree at
  # install time (install(DIRECTORY) would capture an empty dir at configure
  # time, so use install(CODE)).
  if (DEFINED BUILD_TORCHAIR)
    install(CODE "
      if(EXISTS \"${CMAKE_BINARY_DIR}/packages/torch_npu/dynamo/torchair\")
        file(COPY \"${CMAKE_BINARY_DIR}/packages/torch_npu/dynamo/torchair/\"
             DESTINATION \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/dynamo/torchair\")
      endif()
      if(EXISTS \"${CMAKE_BINARY_DIR}/packages/torch_npu/dynamo/npugraph_ex\")
        file(COPY \"${CMAKE_BINARY_DIR}/packages/torch_npu/dynamo/npugraph_ex/\"
             DESTINATION \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/dynamo/npugraph_ex\")
      endif()
      if(EXISTS \"${CMAKE_BINARY_DIR}/packages/torch_npu/_inductor/ascendc\")
        file(COPY \"${CMAKE_BINARY_DIR}/packages/torch_npu/_inductor/ascendc/\"
             DESTINATION \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/_inductor/ascendc\")
      endif()")
  endif()

  # --- fxrt -> torch_npu/fxrt/ -------------------------------------------------
  if (DEFINED BUILD_FXRT)
    install(CODE "
      if(EXISTS \"${CMAKE_BINARY_DIR}/packages/torch_npu/fxrt\")
        file(COPY \"${CMAKE_BINARY_DIR}/packages/torch_npu/fxrt/\"
             DESTINATION \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/fxrt\")
      endif()")
  endif()

endif()

# The Python watchdog tests do not require GoogleTest or Kineto headers.
add_library(watchdog_fault_injection SHARED EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/distributed/watchdog/watchdog_fault_injection.cpp)
target_link_libraries(watchdog_fault_injection PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(watchdog_fault_injection PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/gtest)

if (BUILD_GTEST)
  enable_testing()
  # Make internal HCCL utilities linkable from unit tests in test builds only.
  if (COMPILER_SUPPORTS_HIDDEN_VISIBILITY)
    set_property(SOURCE ${TORCHNPU_ROOT}/distributed/HCCLUtils.cpp
      APPEND PROPERTY COMPILE_OPTIONS -fvisibility=default)
  endif()
  SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/gtest)
  if (NOT TARGET gtest AND NOT TARGET gtest_main)
    add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/googletest)
  elseif (NOT TARGET gtest OR NOT TARGET gtest_main)
    message(FATAL_ERROR
      "BUILD_GTEST requires both gtest and gtest_main when either target already exists")
  endif()
  include_directories(${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include)

  set(TORCH_API_TEST_SOURCES)
  set(TORCH_HCCL_TEST_SOURCES)
  add_subdirectory(${PROJECT_SOURCE_DIR}/test/cpp/api)
  add_executable(test_api ${TORCH_API_TEST_SOURCES})
  add_library(fr_fault_injection SHARED EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/test/distributed/flight_recorder/fr_fault_injection.cpp)
  target_link_libraries(fr_fault_injection PRIVATE ${CMAKE_DL_LIBS})
  set_target_properties(fr_fault_injection PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/gtest)
  add_executable(test_hccl_utils ${TORCH_HCCL_TEST_SOURCES})
  set_target_properties(test_hccl_utils PROPERTIES
    BUILD_WITH_INSTALL_RPATH FALSE)

  find_path(KINETO_TEST_INCLUDE_DIR ActivityBuffers.h
    HINTS "${PYTORCH_INSTALL_DIR}/include/kineto"
    NO_DEFAULT_PATH
    DOC "Kineto internal headers matching installed PyTorch (libkineto/src)")
  if(NOT EXISTS "${KINETO_TEST_INCLUDE_DIR}/ActivityBuffers.h")
    message(FATAL_ERROR
      "test_api requires ActivityBuffers.h. Set KINETO_TEST_INCLUDE_DIR to "
      "libkineto/src from the Kineto version used to build installed PyTorch.")
  endif()
  target_include_directories(test_api PRIVATE "${KINETO_TEST_INCLUDE_DIR}")
  target_compile_definitions(test_api PRIVATE KINETO_NAMESPACE=libkineto)

  target_link_libraries(test_api PUBLIC torch_npu)
  target_link_libraries(test_api PUBLIC gtest_main gtest gmock)
  target_link_libraries(test_hccl_utils PRIVATE torch_npu gtest_main gtest)
  add_test(NAME test_hccl_utils COMMAND test_hccl_utils)
endif()

if (DEFINED BUILD_LIBTORCH)
  configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/Torch_npuConfig.cmake.in
    ${PROJECT_SOURCE_DIR}/build/Torch_npuConfig.cmake
    @ONLY)
endif()
