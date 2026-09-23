#include "ops/ascend/aclnn/aclnn_dynamic_quant_v2.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

namespace {
inline std::optional<ir::TensorPtr> GetOptionalTensor(const ir::Value* value) {
  return value->IsTensor() ? std::optional(value->ToTensor()) : std::nullopt;
}
} // namespace

constexpr size_t kInputIdx = 0;
constexpr size_t kSmoothScalesIdx = 1;
constexpr size_t kGroupIndexIdx = 2;
constexpr size_t kDstTypeIdx = 3;

constexpr size_t kYOutIdx = 0;
constexpr size_t kScaleOutIdx = 1;
constexpr size_t kOffsetOutIdx = 2;

constexpr int64_t kInt8OutputType = 2;
constexpr int64_t kInt32OutputType = 3;

OpsErrorCode AclnnDynamicQuantV2::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  auto xTensor = input[kInputIdx]->ToTensor();
  auto smoothScalesTensor = GetOptionalTensor(input[kSmoothScalesIdx]);
  auto groupIndexTensor = GetOptionalTensor(input[kGroupIndexIdx]);

  if (input[kDstTypeIdx]->IsNone()) {
    outputType_ = kInt8OutputType;
  } else {
    int64_t dtypeInput = input[kDstTypeIdx]->ToInt();
    if (dtypeInput == ir::DataType::Type::QInt8 || dtypeInput == ir::DataType::Type::Int8) {
      outputType_ = kInt8OutputType;
    } else if (dtypeInput == ir::DataType::Type::QUInt4x2) {
      outputType_ = kInt32OutputType;
    } else {
      RT_GLOG(ERROR) << "Dtype must be QInt8, Int8 or QUInt4x2, but got: " << dtypeInput;
      return INVALID_PARAM;
    }
  }

  auto& outputTuple = output->ToTuple();
  auto yOutTensor = (*outputTuple)[kYOutIdx]->ToTensor();
  auto scaleOutTensor = (*outputTuple)[kScaleOutIdx]->ToTensor();

  std::optional<ir::TensorPtr> offsetOutTensor;
  if (outputTuple->Size() > kOffsetOutIdx) {
    auto offsetVal = (*outputTuple)[kOffsetOutIdx];
    if (offsetVal->IsTensor()) {
      offsetOutTensor.emplace(offsetVal->ToTensor());
    }
  }

  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      xTensor,
      smoothScalesTensor,
      groupIndexTensor,
      outputType_,
      yOutTensor,
      scaleOutTensor,
      offsetOutTensor);

  return SUCCESS;
}

OpsErrorCode AclnnDynamicQuantV2::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  auto xTensor = input[kInputIdx]->ToTensor();
  auto smoothScalesTensor = GetOptionalTensor(input[kSmoothScalesIdx]);
  auto groupIndexTensor = GetOptionalTensor(input[kGroupIndexIdx]);

  auto& outputTuple = output->ToTuple();
  auto yOutTensor = (*outputTuple)[kYOutIdx]->ToTensor();
  auto scaleOutTensor = (*outputTuple)[kScaleOutIdx]->ToTensor();

  std::optional<ir::TensorPtr> offsetOutTensor;
  if (outputTuple->Size() > kOffsetOutIdx) {
    auto offsetVal = (*outputTuple)[kOffsetOutIdx];
    if (offsetVal->IsTensor()) {
      offsetOutTensor.emplace(offsetVal->ToTensor());
    }
  }

  executor_->Launch(
      workspace,
      workspaceSize,
      stream,
      xTensor,
      smoothScalesTensor,
      groupIndexTensor,
      outputType_,
      yOutTensor,
      scaleOutTensor,
      offsetOutTensor);

  return SUCCESS;
}

FXRT_REG_OP(npu_dynamic_quant, AclnnDynamicQuantV2, Ascend);

} // namespace ops
} // namespace fxrt
