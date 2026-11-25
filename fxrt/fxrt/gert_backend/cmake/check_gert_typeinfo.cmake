# Fails the build if mrt_fx_converter.so defines gert's allocator types itself.
#
# Why this exists rather than a comment on the link line: fx_converter links libgert.so for
# GE's offline-optimizer passes, and GE also ships the same runtime as object files. If any
# reference resolves to an object instead of the shared library, this image ends up with its
# own copy of gert::Allocators / gert::ExternalAllocator typeinfo and GE's static
# initializers. Both failure modes are silent until run time:
#
#   * SelectL1Allocator (running inside libgert.so) dynamic_casts the allocator it is handed;
#     a foreign typeinfo makes that __dynamic_cast crash;
#   * duplicate static initializers and TLS globals corrupt the heap at interpreter exit,
#     which looks like passing assertions plus a non-zero exit code.
#
# Only a *definition* is a problem. An undefined reference (symbol type U) is exactly what we
# want: it means the symbol resolves from libgert.so at load time.
#
# Invoked as: cmake -DTARGET_SO=<path> -DNM=<nm> -P check_gert_typeinfo.cmake

if(NOT DEFINED TARGET_SO OR NOT EXISTS "${TARGET_SO}")
  message(FATAL_ERROR "check_gert_typeinfo: TARGET_SO not set or missing (got '${TARGET_SO}')")
endif()
if(NOT DEFINED NM OR NM STREQUAL "")
  message(FATAL_ERROR "check_gert_typeinfo: NM not set; pass -DNM=\${CMAKE_NM}")
endif()

execute_process(
  COMMAND "${NM}" -C "${TARGET_SO}"
  OUTPUT_VARIABLE _nm_out
  ERROR_VARIABLE _nm_err
  RESULT_VARIABLE _nm_ret
)
if(NOT _nm_ret EQUAL 0)
  message(FATAL_ERROR "check_gert_typeinfo: '${NM} -C ${TARGET_SO}' failed: ${_nm_err}")
endif()

# Keep only nm lines whose symbol type is a definition (not U, not w/W unless weak-defined,
# which for typeinfo still means a definition).
string(REPLACE "\n" ";" _nm_lines "${_nm_out}")
set(_offenders "")
foreach(_line IN LISTS _nm_lines)
  if(NOT _line MATCHES "typeinfo for gert::(Allocators|ExternalAllocator)")
    continue()
  endif()
  if(_line MATCHES "^[0-9a-fA-F]+ [^U] ")
    string(APPEND _offenders "    ${_line}\n")
  endif()
endforeach()

if(NOT _offenders STREQUAL "")
  message(FATAL_ERROR
    "check_gert_typeinfo: ${TARGET_SO} defines gert allocator typeinfo, so this image "
    "carries a second copy of GE's allocator runtime:\n${_offenders}\n"
    "This crashes GE's SelectL1Allocator (__dynamic_cast on a foreign typeinfo) and corrupts "
    "the heap at interpreter exit. The usual cause is linking a prebuilt GE object archive, "
    "or any reference libgert.so does not export. Link against libgert.so instead.")
endif()

message(STATUS "check_gert_typeinfo: ${TARGET_SO} borrows gert's allocator types from libgert.so")
