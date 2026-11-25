# These GE implementation details must be resolved inside the converter and must
# not be exported. Checking the dynamic table catches both accidental imports
# (which link stubs can satisfy) and symbol interposition with another runtime.
execute_process(
  COMMAND "${NM}" -D -C "${TARGET_SO}"
  OUTPUT_VARIABLE _symbols
  ERROR_VARIABLE _error
  RESULT_VARIABLE _result
)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "check_ge_compat: cannot inspect ${TARGET_SO}: ${_error}")
endif()

string(REPLACE "\n" ";" _lines "${_symbols}")
foreach(_line IN LISTS _lines)
  if(_line MATCHES " [A-Za-z] gert::(HashBuffer::|TilingCache::|TilingCacheKey::|TilingCacheKeyHash::|TilingCacheManager::|RtKernelLaunchArgsEx::(MakeCopy|InitHostInputInfoByDynamicIo)\\(|GertTensorData::GertTensorData\\(void\\*|kernel::(AddTilingCache|ApplyTilingCache|AlignWorkspaceSizes|RefreshOutputAddr)\\()")
    message(FATAL_ERROR "check_ge_compat: GE compatibility implementation is not local: ${_line}")
  endif()
endforeach()
message(STATUS "check_ge_compat: GE compatibility symbols are local to the converter")
