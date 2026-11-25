# ACLNN symbols must be borrowed from FXRT's common loader, never linked through
# CANN stubs. Check the artifact so new direct calls cannot silently regress this.
execute_process(
  COMMAND "${NM}" -D --undefined-only "${TARGET_SO}"
  OUTPUT_VARIABLE _imports ERROR_VARIABLE _error RESULT_VARIABLE _result)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "Cannot inspect ACLNN imports: ${_error}")
endif()
if(_imports MATCHES "(^|\n)[^\n]* U (aclnn[A-Za-z0-9_]+|acl(Create|Destroy)[A-Za-z0-9_]+)(@[^\n]*)?(\n|$)")
  message(FATAL_ERROR "Converter has a direct ACLNN import: ${CMAKE_MATCH_0}")
endif()
execute_process(
  COMMAND "${READELF}" -d "${TARGET_SO}"
  OUTPUT_VARIABLE _dynamic ERROR_VARIABLE _error RESULT_VARIABLE _result)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "Cannot inspect ACLNN dependencies: ${_error}")
endif()
if(_dynamic MATCHES "Shared library: \\[lib(opapi|nnopbase)\\.so[^]]*\\]")
  message(FATAL_ERROR "Converter has a direct ACLNN library dependency: ${CMAKE_MATCH_0}")
endif()
message(STATUS "check_aclnn_dependencies: no direct ACLNN imports or opapi/nnopbase dependencies")
