#include "ops/ascend/aclnn/aclnn_ascend_quant_v3.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

namespace {
inline std::optional<ir::TensorPtr> GetOptionalTensor(const ir::Value* value) {
  return value->IsTensor() ? std::optional(value->ToTensor()) : std::nullopt;
}

} // namespace

constexpr size_t kSelfIdx = 0;
constexpr size_t kScalesIdx = 1;
constexpr size_t kZeroPointsIdx = 2;
constexpr size_t kDtypeIdx = 3;
constexpr size_t kAxisIdx = 4;
constexpr size_t kDivModeIdx = 5;

OpsErrorCode AclnnAscendQuantV3::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  auto xTensor = input[kSelfIdx]->ToTensor();
  auto scaleTensor = input[kScalesIdx]->ToTensor();
  auto offsetTensor = GetOptionalTensor(input[kZeroPointsIdx]);

  // Verify divMode, this op only supports divMode=false (using AscendQuantV3)
  // op plugin use: acl_op::npu_quantize
  div_mode_ = input[kDivModeIdx]->ToBool();
  if (div_mode_) {
    RT_GLOG(ERROR) << "divMode must be false for AclnnAscendQuantV3";
    return INVALID_PARAM;
  }

  int64_t axisInput = input[kAxisIdx]->ToInt();
  axis_ = (axisInput < -1) ? static_cast<int32_t>(axisInput) : -1;

  int64_t dtypeInput = input[kDtypeIdx]->ToInt();
  if (dtypeInput == ir::DataType::Type::QInt8) {
    dst_type_ = ir::DataType::Type::Int8;
  } else if (dtypeInput == ir::DataType::Type::QUInt4x2) {
    dst_type_ = ir::DataType::Type::Int32;
  } else {
    RT_GLOG(ERROR) << "Dtype must be QInt8 or QUInt4x2, but got: " << dtypeInput;
    return INVALID_PARAM;
  }

  // Fixed parameters for AscendQuantV3
  bool sqrtMode = false;
  const char* roundMode = "round";

  auto yTensor = output->ToTensor();

  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      xTensor,
      scaleTensor,
      offsetTensor,
      sqrtMode,
      roundMode,
      dst_type_,
      axis_,
      yTensor);

  return SUCCESS;
}

OpsErrorCode AclnnAscendQuantV3::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  auto xTensor = input[kSelfIdx]->ToTensor();
  auto scaleTensor = input[kScalesIdx]->ToTensor();
  auto offsetTensor = GetOptionalTensor(input[kZeroPointsIdx]);

  bool sqrtMode = false;
  const char* roundMode = "round";
  auto yTensor = output->ToTensor();

  executor_->Launch(
      workspace,
      workspaceSize,
      stream,
      xTensor,
      scaleTensor,
      offsetTensor,
      sqrtMode,
      roundMode,
      dst_type_,
      axis_,
      yTensor);

  return SUCCESS;
}

FXRT_REG_OP(npu_quantize, AclnnAscendQuantV3, Ascend);

} // namespace ops
} // namespace fxrt
