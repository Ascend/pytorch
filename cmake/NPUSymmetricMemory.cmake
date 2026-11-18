include_guard(GLOBAL)

# torch-npu selects its libtorch build by presence, not by boolean value.
if(DEFINED BUILD_LIBTORCH)
  message(FATAL_ERROR "Symmetric memory kernels are not supported with BUILD_LIBTORCH")
endif()

set(NPU_SYMM_MEM_ARCH "$ENV{NPU_SYMM_MEM_ARCH}" CACHE STRING "Single symmetric memory kernel architecture")
set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit root for compiler discovery")
# Explicit compiler choices never silently fall back. Auto-discovery is repeated
# on configure so switching SDKs cannot retain a stale find_program cache entry.
if(DEFINED BISHENG)
  set(_symm_mem_compiler "${BISHENG}")
elseif(NOT "$ENV{BISHENG}" STREQUAL "")
  set(_symm_mem_compiler "$ENV{BISHENG}")
else()
  unset(_symm_mem_bisheng CACHE)
  find_program(_symm_mem_bisheng NAMES bisheng
    HINTS "${ASCEND_HOME_PATH}/bin" "${ASCEND_HOME_PATH}/compiler/ccec_compiler/bin")
  set(_symm_mem_compiler "${_symm_mem_bisheng}")
  unset(_symm_mem_bisheng CACHE)
  if(NOT _symm_mem_compiler)
    message(STATUS "BiSheng compiler not found; skipping optional symmetric memory kernels")
    return()
  endif()
endif()

set(_symm_mem_extension "${TORCH_NPU_SOURCE_DIR}/torch_npu/utils/_npu_extension.py")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_LIST_DIR}/resolve_npu_symm_mem_arch.py"
    "${_symm_mem_extension}" "${NPU_SYMM_MEM_ARCH}"
  RESULT_VARIABLE _symm_mem_arch_result
  OUTPUT_VARIABLE _symm_mem_arch
  ERROR_VARIABLE _symm_mem_arch_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE)
if(NOT _symm_mem_arch_result EQUAL 0)
  message(FATAL_ERROR "Cannot resolve NPU_SYMM_MEM_ARCH: ${_symm_mem_arch_error}")
endif()
if(_symm_mem_arch_error)
  message(WARNING "${_symm_mem_arch_error}")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/resolve_npu_symm_mem_arch.py" "${_symm_mem_extension}")

# Reuse the main build's staged ACL headers and link-only stub library.
# Keep devkit outside acl_src, which generate_code.sh cleans after staging ACL.
set(_symm_mem_devkit "${TORCH_NPU_SOURCE_DIR}/third_party/asc-devkit")
set(_symm_mem_acl_include "${TORCH_NPU_SOURCE_DIR}/third_party/acl/inc")
set(_symm_mem_acl_libdir "${TORCH_NPU_SOURCE_DIR}/third_party/acl/libs")
set(_symm_mem_acl "${_symm_mem_acl_libdir}/libascendcl.so")
set(_symm_mem_tiling_script "${_symm_mem_devkit}/impl/adv_api/cmake/scripts/gen_kernel_tiling_data_def.py")
foreach(_required "${_symm_mem_compiler}" "${_symm_mem_acl_include}/acl/acl.h"
    "${_symm_mem_devkit}/include/kernel_operator.h" "${_symm_mem_tiling_script}" "${_symm_mem_acl}")
  if(NOT EXISTS "${_required}" OR IS_DIRECTORY "${_required}")
    message(FATAL_ERROR "Missing symmetric memory build dependency: ${_required}; check BISHENG, asc-devkit submodule and main ACL header/stub preparation")
  endif()
endforeach()

# The ordinary C++ linker does not add BiSheng's implicit host runtime.
# Resolve it from the selected SDK, or the compiler's installation when no
# SDK root was supplied. Keep all libraries in that installation.
set(_symm_mem_runtime_root "${ASCEND_HOME_PATH}")
if(NOT _symm_mem_runtime_root)
  get_filename_component(_symm_mem_compiler_path "${_symm_mem_compiler}" REALPATH)
  get_filename_component(_symm_mem_compiler_dir "${_symm_mem_compiler_path}" DIRECTORY)
  foreach(_root "${_symm_mem_compiler_dir}/.." "${_symm_mem_compiler_dir}/../../..")
    if(EXISTS "${_root}/lib64/libascendc_runtime.a")
      get_filename_component(_symm_mem_runtime_root "${_root}" ABSOLUTE)
      break()
    endif()
  endforeach()
endif()
if(NOT _symm_mem_runtime_root)
  message(FATAL_ERROR "Cannot locate the Ascend C host runtime; set ASCEND_HOME_PATH to the CANN toolkit matching BISHENG")
endif()
set(_symm_mem_runtime_libraries "${_symm_mem_runtime_root}/lib64/libascendc_runtime.a")
foreach(_library runtime error_manager profapi unified_dlog mmpa ascend_dump c_sec)
  list(APPEND _symm_mem_runtime_libraries "${_symm_mem_runtime_root}/lib64/lib${_library}.so")
endforeach()
foreach(_library IN LISTS _symm_mem_runtime_libraries)
  if(NOT EXISTS "${_library}" OR IS_DIRECTORY "${_library}")
    message(FATAL_ERROR "Missing Ascend C host runtime dependency: ${_library}; check ASCEND_HOME_PATH and the CANN toolkit installation")
  endif()
endforeach()

set(_symm_mem_source_dir "${TORCH_NPU_SOURCE_DIR}/torch_npu/csrc/distributed/symm_mem")
set(_symm_mem_source "${_symm_mem_source_dir}/NPUSHMEMSymmetricMemoryKernels.asc")
set(_symm_mem_object_dir "${CMAKE_CURRENT_BINARY_DIR}/symm_mem/${_symm_mem_arch}")
set(_symm_mem_object "${_symm_mem_object_dir}/NPUSHMEMSymmetricMemoryKernels.o")
set(_symm_mem_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/symm_mem/generated")
set(_symm_mem_tiling_header "${_symm_mem_generated_dir}/kernel_tiling/kernel_tiling.h")
# Match the devkit source-tree layout without invoking its top-level CMake.
# BiSheng prepends SDK paths to -I; quoted includes must search the pinned
# source headers first, while retaining compiler builtin and standard headers.
set(_symm_mem_include_flags "-iquote" "${TORCH_NPU_SOURCE_DIR}" "-iquote" "${_symm_mem_acl_include}")
foreach(_include "${_symm_mem_devkit}" "${_symm_mem_devkit}/include"
    "${_symm_mem_devkit}/include/basic_api" "${_symm_mem_devkit}/include/simt_api"
    "${_symm_mem_devkit}/impl" "${_symm_mem_devkit}/impl/basic_api"
    "${_symm_mem_devkit}/impl/simt_api" "${_symm_mem_generated_dir}")
  list(APPEND _symm_mem_include_flags "-iquote" "${_include}")
endforeach()
file(GLOB_RECURSE _symm_mem_headers CONFIGURE_DEPENDS
  "${_symm_mem_devkit}/include/*.h" "${_symm_mem_devkit}/include/*.hpp"
  "${_symm_mem_devkit}/impl/*.h" "${_symm_mem_devkit}/impl/*.hpp"
  "${_symm_mem_acl_include}/*.h")
file(GLOB_RECURSE _symm_mem_tiling_inputs CONFIGURE_DEPENDS
  "${_symm_mem_devkit}/include/adv_api/*tilingdata.h")
# Make, unlike Ninja, does not track changed command lines. This content-stable
# stamp also handles A -> B -> A architecture/toolchain switches and removals
# from the header list. Header contents are tracked by the file dependencies.
string(SHA256 _symm_mem_header_list_hash "${_symm_mem_headers}")
set(_symm_mem_config "${CMAKE_CURRENT_BINARY_DIR}/symm_mem/build-config.txt")
file(CONFIGURE OUTPUT "${_symm_mem_config}"
  CONTENT "${_symm_mem_arch}\n${_symm_mem_compiler}\n${ASCEND_HOME_PATH}\n${Python3_EXECUTABLE}\n${_symm_mem_include_flags}\n${_symm_mem_acl}\n${_symm_mem_header_list_hash}\n" @ONLY)
add_custom_command(
  OUTPUT "${_symm_mem_tiling_header}"
  COMMAND "${Python3_EXECUTABLE}" -B "${_symm_mem_tiling_script}"
    "${_symm_mem_devkit}/include/adv_api" "${_symm_mem_tiling_header}"
  DEPENDS "${_symm_mem_tiling_script}" ${_symm_mem_tiling_inputs} "${_symm_mem_config}"
  COMMENT "Generating Ascend C kernel tiling headers"
  VERBATIM)

# ASC stays C++17, independent of the framework's C++20/LTO/ABI flags.
# The kernel source checks the required primitive capabilities.
add_custom_command(
  OUTPUT "${_symm_mem_object}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_symm_mem_object_dir}"
  COMMAND "${_symm_mem_compiler}" -x asc -c "--npu-arch=${_symm_mem_arch}" -std=c++17 -fPIC -O2
    ${_symm_mem_include_flags} "${_symm_mem_source}" -o "${_symm_mem_object}"
  DEPENDS "${_symm_mem_source}"
    "${_symm_mem_source_dir}/NPUSHMEMSymmetricMemoryKernels.hpp"
    "${_symm_mem_tiling_header}" ${_symm_mem_headers}
    "${_symm_mem_compiler}" "${_symm_mem_config}" "${CMAKE_CURRENT_LIST_FILE}"
  COMMENT "Compiling symmetric memory kernels for ${_symm_mem_arch}"
  VERBATIM)

set_source_files_properties("${_symm_mem_object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
add_custom_target(torch_npu_symm_mem_kernels DEPENDS "${_symm_mem_object}")
