#include "ops/ascend/mem/empty_strided.h"

#include <string>
#include <vector>

#include "common/common.h"
#include "common/logger.h"
#include "ir/tensor/format.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
namespace {
constexpr size_t kSizeInputIndex = 0;
constexpr size_t kStrideInputIndex = 1;

std::vector<int64_t> GetIntListInput(const std::vector<const ir::Value*>& input, size_t index, const char* inputName) {
  CHECK_IF_FAIL_MSG(input.size() > index, "empty_strided missing " + std::string(inputName) + " input");
  CHECK_IF_NULL(input[index]);
  if (!input[index]->IsTuple()) {
    RT_GLOG(EXCEPTION) << "empty_strided expects " << inputName << " to be an int tuple, got: " << *input[index];
  }
  return input[index]->ToTuple()->ToIntList();
}

bool IsContiguousStride(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides) {
  if (shape.size() != strides.size()) {
    return false;
  }
  int64_t expectedStride = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    if (shape[i] < 0 || strides[i] < 0) {
      return false;
    }
    if (shape[i] == 0) {
      return true;
    }
    if (shape[i] != 1 && strides[i] != expectedStride) {
      return false;
    }
    expectedStride *= shape[i];
  }
  return true;
}
} // namespace

OpsErrorCode EmptyStrided::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  CHECK_IF_NULL(output);
  CHECK_IF_NULL(workspaceSize);
  auto outputTensor = output->ToTensor();
  CHECK_IF_NULL(outputTensor);

  const auto size = GetIntListInput(input, kSizeInputIndex, "size");
  const auto strides = GetIntListInput(input, kStrideInputIndex, "stride");
  CHECK_IF_FAIL_MSG(outputTensor->Shape() == size, "empty_strided output shape mismatch");
  if (!IsContiguousStride(size, strides)) {
    RT_GLOG(EXCEPTION) << "empty_strided only supports contiguous stride in FXRT now, shape=" << size
                       << ", stride=" << strides;
  }

  outputTensor->SetStrides(strides);
  outputTensor->SetStorageOffset(0);
  outputTensor->SetStorageShape(size);
  outputTensor->SetFormat(ir::FORMAT_ND);
  *workspaceSize = 0;
  return SUCCESS;
}

OpsErrorCode EmptyStrided::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  return SUCCESS;
}

FXRT_REG_OP(empty_strided, EmptyStrided, Ascend);
} // namespace ops
} // namespace fxrt
