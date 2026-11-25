#pragma once

#include "exe_graph/runtime/op_execute_context.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/tensor_data.h"
#include "graph/types.h"

namespace fx_converter {

// Process-long Const payloads shared by all dual-stage lowerings.
gert::OpExecuteOptions* SharedExecOptions();

// Follow torch's cube precision switches (torch.npu.{conv,matmul}.allow_hf32). The
// dual-stage aclnn kernels read this out of the shared OpExecuteOptions Const at
// launch, which is how a GE-built graph gets the hf32 mode torch-npu passes to aclnn
// as cubeMathType. Plain bools: this crosses the ABI=0 boundary.
void SetAllowHf32(bool conv, bool matmul);
ge::DataType* SharedDtypeFloat();

} // namespace fx_converter
