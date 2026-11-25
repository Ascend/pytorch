# The producer and consumer now share the same in-tree FXRT sources.
get_filename_component(_FXRT_PKG_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(MRT_SRC_ROOT "${_FXRT_PKG_ROOT}/src")

# Probe the tree the way the in-tree DVM module probes torch_npu: look for the exact file
# this build depends on, and say what is missing rather than failing later in a compile.
set(FXC_IR_EXPORT_HEADER "${MRT_SRC_ROOT}/runtime/executor/ir_graph_export.h")
set(FXC_IR_EXPORT_FIELDS "${MRT_SRC_ROOT}/runtime/executor/ir_graph_export_fields.inc")
foreach(_probe "${MRT_SRC_ROOT}/ir/graph.h" "${FXC_IR_EXPORT_HEADER}" "${FXC_IR_EXPORT_FIELDS}")
  if(NOT EXISTS "${_probe}")
    message(FATAL_ERROR
      "FXRT source '${_FXRT_PKG_ROOT}' does not look like a torch_npu tree "
      "carrying fxrt: missing '${_probe}'.\n"
      "If ir_graph_export_fields.inc is the one missing, that torch_npu predates the "
      "versioned IR export contract and cannot be built against.")
  endif()
endforeach()

# Check the schema before compiling the isolated ABI=0 reader. Catch a mismatch here rather than
# letting Convert mis-read a dict at runtime.
set(_FXC_SUPPORTED_IR_SCHEMA_VERSIONS "5" "6")
file(STRINGS "${FXC_IR_EXPORT_HEADER}" _schema_line
  REGEX "constexpr[ \t]+int[ \t]+kIrExportSchemaVersion[ \t]*=")
if("${_schema_line}" STREQUAL "")
  message(FATAL_ERROR
    "Could not find kIrExportSchemaVersion in ${FXC_IR_EXPORT_HEADER}. "
    "That torch_npu predates the versioned IR export contract.")
endif()
string(REGEX MATCH "=[ \t]*([0-9]+)" _schema_match "${_schema_line}")
set(FXC_TORCH_NPU_IR_SCHEMA_VERSION "${CMAKE_MATCH_1}")
if(NOT FXC_TORCH_NPU_IR_SCHEMA_VERSION IN_LIST _FXC_SUPPORTED_IR_SCHEMA_VERSIONS)
  message(FATAL_ERROR
    "IR export schema mismatch: this converter supports versions 5 and 6, "
    "but ${_FXRT_PKG_ROOT} produces version "
    "${FXC_TORCH_NPU_IR_SCHEMA_VERSION}.\n"
    "Port the ir_import reader to the new schema, or build against a matching torch_npu.")
endif()

set(FXC_EXPECTED_IR_SCHEMA_VERSION "${FXC_TORCH_NPU_IR_SCHEMA_VERSION}")

message(STATUS "fx_converter fxrt core: ${_FXRT_PKG_ROOT}")
message(STATUS "fx_converter IR export schema version: ${FXC_TORCH_NPU_IR_SCHEMA_VERSION}")
