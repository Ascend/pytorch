#include <vector>

#include "ops/cpu/symbolic/size.h"
#include "ops/op_register.h"
#include "ops/utils/op_constants.h"
#include "common/logger.h"
#include "ir/common/dtype.h"

namespace fxrt {
namespace ops {
OpsErrorCode Size::InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) {
  // For now we implement the common case: `size(tensor, dim) -> i64`.
  // This is required by dynamic-shape graphs where expand/view shapes are built from `fxrt.size`.
  if (input.size() != kInputSize2) {
    RT_GLOG(ERROR) << "Size::InferShape expects 2 inputs, but got: " << input.size();
    return INVALID_INPUT_NUM;
  }
  if (!input[kIndex0] || !input[kIndex0]->IsTensor()) {
    RT_GLOG(ERROR) << "Size::InferShape expects input[0] to be a tensor";
    return INVALID_PARAM;
  }
  if (!input[kIndex1]) {
    RT_GLOG(ERROR) << "Size::InferShape expects input[1] (dim) not null";
    return INVALID_PARAM;
  }

  if (!input[kIndex1]->IsInt() || output->IsSymbol()) {
    // Handle case where input[kIndex1] could be None and returns a tuple.
    // Falls back to parent class Operator::InferShape (verified).
    Operator::InferShape(input, output);
    return SUCCESS;
  }

  const auto& shape = input[kIndex0]->ToTensor()->Shape();
  int64_t dim = input[kIndex1]->ToInt();

  // Support negative dim indexing: -n <= dim < n
  int64_t ndim = static_cast<int64_t>(shape.size());
  CHECK_IF_FAIL(dim >= -ndim && dim < ndim);
  if (dim < 0) {
    dim += ndim;
  }

  // Output is an i64 Value; its tag is fixed at graph-build time, so we must
  // keep it as Tag::Int (Value(int64_t)).
  *output = ir::Value(static_cast<int64_t>(shape[dim]));
  return SUCCESS;
}

OpsErrorCode Size::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  return SUCCESS;
}

bool Size::NeedLaunch() {
  return false;
}

FXRT_REG_OP(size, Size, CPU);
} // namespace ops
} // namespace fxrt
