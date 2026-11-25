#include "exe_graph/shared_const_store.h"

namespace fx_converter {
namespace {

// allow_hf32 is a 2-char string the aclnn adapter reads back through
// OpExecuteContext::GetAllowHf32(): [0] conv, [1] matmul, '1' enables. Default to
// torch-npu's own defaults (conv on, matmul off) so the field is never the zero
// bytes of a value-initialized struct, which reads as "neither" and silently pins
// every cube op to full fp32.
gert::OpExecuteOptions g_exec_options = {0, 0, {'1', '0', '\0'}, {}};
// SharedRuntime still feeds one Float32 dtype Const into CalcTensorSize.
// Per-op graphs pass their own dtype Const.
ge::DataType g_dtype_float = ge::DT_FLOAT;

}  // namespace

gert::OpExecuteOptions *SharedExecOptions() { return &g_exec_options; }

void SetAllowHf32(bool conv, bool matmul) {
  g_exec_options.allow_hf32[0] = conv ? '1' : '0';
  g_exec_options.allow_hf32[1] = matmul ? '1' : '0';
  g_exec_options.allow_hf32[2] = '\0';
}

ge::DataType *SharedDtypeFloat() { return &g_dtype_float; }

}  // namespace fx_converter
