#include <vector>

#include "ops/ascend/aclnn/aclnn_add_rms_norm_quant_v2.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
namespace {
inline std::optional<ir::TensorPtr> GetOptionalTensor(const ir::Value* value) {
  return value->IsTensor() ? std::optional(value->ToTensor()) : std::nullopt;
}
} // namespace

// Input parameter index definitions aligned with frontend signature:
// npu_add_rms_norm_quant(x1, x2, gamma, scales1, zero_points1=None, beta=None,
//                        scales2=None, zero_points2=None, axis=-1, epsilon=1e-06, div_mode=True)
constexpr size_t kX1Idx = 0;
constexpr size_t kX2Idx = 1;
constexpr size_t kGammaIdx = 2;
constexpr size_t kScales1Idx = 3;
constexpr size_t kZeroPoints1OptionalIdx = 4;
constexpr size_t kBetaOptionalIdx = 5;
constexpr size_t kScales2OptionalIdx = 6;
constexpr size_t kZeroPoints2OptionalIdx = 7;
constexpr size_t kAxisIdx = 8;
constexpr size_t kEpsilonIdx = 9;
constexpr size_t kDivModeIdx = 10;

// Output parameter index definitions aligned with ACLNN interface: y1Out, y2Out, xOut, rmsNormOut
constexpr size_t kY1OutIdx = 0;
constexpr size_t kY2OutIdx = 1;
constexpr size_t kXOutIdx = 2;
constexpr size_t kRmsNormOutIdx = 3;

OpsErrorCode AclnnAddRmsNormQuantV2::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  // Parameter validation: scales2 must be None
  auto scales2_opt = GetOptionalTensor(input[kScales2OptionalIdx]);
  if (scales2_opt.has_value()) {
    RT_GLOG(ERROR) << "Error: scales2 only support None.";
    return INVALID_PARAM;
  }

  // Parameter validation: zero_points2 must be None
  auto zero_points2_opt = GetOptionalTensor(input[kZeroPoints2OptionalIdx]);
  if (zero_points2_opt.has_value()) {
    RT_GLOG(ERROR) << "Error: zero_points2 only support None.";
    return INVALID_PARAM;
  }

  // Parameter validation: axis must be -1
  auto axis_val = input[kAxisIdx]->ToInt();
  if (axis_val != -1) {
    RT_GLOG(ERROR) << "Error: axis only support -1, but got " << axis_val << ".";
    return INVALID_PARAM;
  }

  // Parameter validation: div_mode must be True
  auto div_mode_val = input[kDivModeIdx]->ToBool();
  if (!div_mode_val) {
    RT_GLOG(ERROR) << "Error: div_mode only support True.";
    return INVALID_PARAM;
  }

  auto& outputTuple = output->ToTuple();

  // Note: The ACLNN interface parameter order is:
  // x1, x2, gamma, scales1,
  // scales2Optional, zeroPoints1Optional, zeroPoints2Optional, betaOptional,
  // axis, epsilon, divMode, workspaceSize, executor
  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[kX1Idx]->ToTensor(),
      input[kX2Idx]->ToTensor(),
      input[kGammaIdx]->ToTensor(),
      input[kScales1Idx]->ToTensor(),
      // Optional parameters must match ACLNN interface order: scales2, zeroPoints1, zeroPoints2, beta
      GetOptionalTensor(input[kScales2OptionalIdx]),
      GetOptionalTensor(input[kZeroPoints1OptionalIdx]),
      GetOptionalTensor(input[kZeroPoints2OptionalIdx]),
      GetOptionalTensor(input[kBetaOptionalIdx]),
      input[kAxisIdx]->ToInt(),
      input[kEpsilonIdx]->ToDouble(),
      input[kDivModeIdx]->ToBool(),
      (*outputTuple)[kY1OutIdx]->ToTensor(),
      (*outputTuple)[kY2OutIdx]->ToTensor(),
      (*outputTuple)[kXOutIdx]->ToTensor(),
      nullptr);

  return SUCCESS;
}

OpsErrorCode AclnnAddRmsNormQuantV2::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  auto& outputTuple = output->ToTuple();

  // Parameter order must match CalcWorkspace and ACLNN aclnnAddRmsNormQuantV2 interface
  executor_->Launch(
      workspace,
      workspaceSize,
      stream,
      input[kX1Idx]->ToTensor(),
      input[kX2Idx]->ToTensor(),
      input[kGammaIdx]->ToTensor(),
      input[kScales1Idx]->ToTensor(),
      // Optional parameters order: scales2, zeroPoints1, zeroPoints2, beta
      GetOptionalTensor(input[kScales2OptionalIdx]),
      GetOptionalTensor(input[kZeroPoints1OptionalIdx]),
      GetOptionalTensor(input[kZeroPoints2OptionalIdx]),
      GetOptionalTensor(input[kBetaOptionalIdx]),
      input[kAxisIdx]->ToInt(),
      input[kEpsilonIdx]->ToDouble(),
      input[kDivModeIdx]->ToBool(),
      (*outputTuple)[kY1OutIdx]->ToTensor(),
      (*outputTuple)[kY2OutIdx]->ToTensor(),
      (*outputTuple)[kXOutIdx]->ToTensor(),
      nullptr);

  return SUCCESS;
}

FXRT_REG_OP(add_rms_norm_quant, AclnnAddRmsNormQuantV2, Ascend);
} // namespace ops
} // namespace fxrt
