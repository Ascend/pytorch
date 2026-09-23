#include "ops/ascend/aclnn/argsort.h"

#include <vector>

#include "common/logger.h"
#include "ir/value/value.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
namespace {
constexpr size_t kInputTensorIndex = 0;
constexpr size_t kInputStableIndex = 1;
constexpr size_t kInputDimIndex = 2;
constexpr size_t kInputDescendingIndex = 3;
constexpr size_t kInputSize = 4;

constexpr size_t kOutputValuesIndex = 0;
constexpr size_t kOutputIndicesIndex = 1;
constexpr size_t kOutputSize = 2;
} // namespace

void AclnnArgsort::Init(const std::vector<const ir::Value*>& input, const ir::Value* output) {
  (void)input;
  (void)output;
}

OpsErrorCode AclnnArgsort::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  CHECK_IF_FAIL(input.size() == kInputSize);
  CHECK_IF_NULL(output);
  CHECK_IF_NULL(workspaceSize);

  CHECK_IF_FAIL(input[kInputTensorIndex] != nullptr && input[kInputTensorIndex]->IsTensor());
  CHECK_IF_FAIL(input[kInputStableIndex] != nullptr && input[kInputStableIndex]->IsBool());
  CHECK_IF_FAIL(input[kInputDimIndex] != nullptr && input[kInputDimIndex]->IsInt());
  CHECK_IF_FAIL(input[kInputDescendingIndex] != nullptr && input[kInputDescendingIndex]->IsBool());
  CHECK_IF_FAIL(output->IsTuple());

  const auto& outputTuple = output->ToTuple();
  CHECK_IF_FAIL(outputTuple != nullptr && outputTuple->Size() == kOutputSize);
  CHECK_IF_FAIL((*outputTuple)[kOutputValuesIndex] != nullptr && (*outputTuple)[kOutputValuesIndex]->IsTensor());
  CHECK_IF_FAIL((*outputTuple)[kOutputIndicesIndex] != nullptr && (*outputTuple)[kOutputIndicesIndex]->IsTensor());

  const auto& inTensor = input[kInputTensorIndex]->ToTensor();
  bool stable = input[kInputStableIndex]->ToBool();
  int64_t dim = input[kInputDimIndex]->ToInt();
  bool descending = input[kInputDescendingIndex]->ToBool();
  const auto& valuesTensor = (*outputTuple)[kOutputValuesIndex]->ToTensor();
  const auto& indicesTensor = (*outputTuple)[kOutputIndicesIndex]->ToTensor();

  RT_VLOG(VL_OPS) << "AclnnArgsort::CalcWorkspace: dim=" << dim << ", descending=" << descending
                  << ", stable=" << stable;
  executor_->GetWorkspaceSize(
      reinterpret_cast<uint64_t*>(workspaceSize), inTensor, stable, dim, descending, valuesTensor, indicesTensor);

  return SUCCESS;
}

OpsErrorCode AclnnArgsort::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  CHECK_IF_FAIL(input.size() == kInputSize);
  CHECK_IF_NULL(output);
  CHECK_IF_FAIL(output->IsTuple());

  const auto& outputTuple = output->ToTuple();
  CHECK_IF_FAIL(outputTuple != nullptr && outputTuple->Size() == kOutputSize);
  CHECK_IF_FAIL((*outputTuple)[kOutputValuesIndex] != nullptr && (*outputTuple)[kOutputValuesIndex]->IsTensor());
  CHECK_IF_FAIL((*outputTuple)[kOutputIndicesIndex] != nullptr && (*outputTuple)[kOutputIndicesIndex]->IsTensor());

  const auto& inTensor = input[kInputTensorIndex]->ToTensor();
  bool stable = input[kInputStableIndex]->ToBool();
  int64_t dim = input[kInputDimIndex]->ToInt();
  bool descending = input[kInputDescendingIndex]->ToBool();
  const auto& valuesTensor = (*outputTuple)[kOutputValuesIndex]->ToTensor();
  const auto& indicesTensor = (*outputTuple)[kOutputIndicesIndex]->ToTensor();

  executor_->Launch(workspace, workspaceSize, stream, inTensor, stable, dim, descending, valuesTensor, indicesTensor);
  return SUCCESS;
}

FXRT_REG_OP(argsort, AclnnArgsort, Ascend);

} // namespace ops
} // namespace fxrt
