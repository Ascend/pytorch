#include "ops/ascend/aclnn/aclnn_layer_norm.h"

#include <vector>

#include "ir/tensor/tensor.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
namespace {
inline std::optional<ir::TensorPtr> GetOptionalTensor(const ir::Value* value) {
  return value->IsTensor() ? std::optional(value->ToTensor()) : std::nullopt;
}

void GetOutputTensors(
    const ir::Value* output,
    ir::TensorPtr* yOut,
    std::optional<ir::TensorPtr>* meanOut,
    std::optional<ir::TensorPtr>* rstdOut) {
  if (output->IsTuple()) {
    const auto& tup = output->ToTuple();
    *yOut = (*tup)[kIndex0]->ToTensor();
    *meanOut = (*tup)[kIndex1]->ToTensor();
    *rstdOut = (*tup)[kIndex2]->ToTensor();
    return;
  }
  *yOut = output->ToTensor();
  *meanOut = std::nullopt;
  *rstdOut = std::nullopt;
}
} // namespace

constexpr size_t kInputIdx = 0;
constexpr size_t kNormalizedShapeIdx = 1;
constexpr size_t kWeightIdx = 2;
constexpr size_t kBiasIdx = 3;
constexpr size_t kEpsilonIdx = 4;

AclnnLayerNorm::AclnnLayerNorm() {
  executor_ = std::make_unique<AclnnExecutor>("aclnnLayerNorm");
}

OpsErrorCode AclnnLayerNorm::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  ir::TensorPtr yOut;
  std::optional<ir::TensorPtr> meanOut;
  std::optional<ir::TensorPtr> rstdOut;
  GetOutputTensors(output, &yOut, &meanOut, &rstdOut);

  const auto normalizedShape = input[kNormalizedShapeIdx]->ToTuple()->ToIntList();
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[kInputIdx]->ToTensor(),
      normalizedShape,
      GetOptionalTensor(input[kWeightIdx]),
      GetOptionalTensor(input[kBiasIdx]),
      input[kEpsilonIdx]->ToDouble(),
      yOut,
      meanOut,
      rstdOut);

  return SUCCESS;
}

OpsErrorCode AclnnLayerNorm::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  ir::TensorPtr yOut;
  std::optional<ir::TensorPtr> meanOut;
  std::optional<ir::TensorPtr> rstdOut;
  GetOutputTensors(output, &yOut, &meanOut, &rstdOut);

  const auto normalizedShape = input[kNormalizedShapeIdx]->ToTuple()->ToIntList();
  executor_->Launch(
      workspace,
      workspaceSize,
      stream,
      input[kInputIdx]->ToTensor(),
      normalizedShape,
      GetOptionalTensor(input[kWeightIdx]),
      GetOptionalTensor(input[kBiasIdx]),
      input[kEpsilonIdx]->ToDouble(),
      yOut,
      meanOut,
      rstdOut);

  return SUCCESS;
}

FXRT_REG_OP(norm, AclnnLayerNorm, Ascend);
} // namespace ops
} // namespace fxrt
