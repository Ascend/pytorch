#include <vector>

#include "ops/ascend/aclnn/aclnn_fused_infer_attention_score.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
namespace {
inline std::optional<ir::TensorPtr> GetOptionalTensor(const ir::Value* value) {
  return value->IsTensor() ? std::optional(value->ToTensor()) : std::nullopt;
}

inline std::vector<ir::TensorPtr> GetTensorList(const ir::Value* value) {
  return value->IsTuple() ? value->ToTuple()->ToTensorList() : std::vector<ir::TensorPtr>();
}

inline std::optional<std::vector<int64_t>> GetOptionalIntList(const ir::Value* value) {
  return value->IsTuple() ? std::optional(value->ToTuple()->ToIntList()) : std::nullopt;
}

inline std::pair<std::vector<int64_t>, bool> GetIntListPair(const ir::Value* value) {
  return value->IsTuple() ? std::make_pair(value->ToTuple()->ToIntList(), true)
                          : std::make_pair(std::vector<int64_t>{}, true);
}
} // namespace

constexpr size_t kQueryIdx = 0;
constexpr size_t kKeyIdx = 1;
constexpr size_t kValueIdx = 2;
constexpr size_t kPseShiftIdx = 3;
constexpr size_t kAttenMaskIdx = 4;
constexpr size_t kActualSeqLengthsIdx = 5;
constexpr size_t kActualSeqLengthsKvIdx = 6;
constexpr size_t kDeqScale1Idx = 7;
constexpr size_t kQuantScale1Idx = 8;
constexpr size_t kDeqScale2Idx = 9;
constexpr size_t kQuantScale2Idx = 10;
constexpr size_t kQuantOffset2Idx = 11;
constexpr size_t kAntiquantScaleIdx = 12;
constexpr size_t kAntiquantOffsetIdx = 13;
constexpr size_t kBlockTableIdx = 14;
constexpr size_t kQueryPaddingSizeIdx = 15;
constexpr size_t kKvPaddingSizeIdx = 16;
constexpr size_t kKeyAntiquantScaleIdx = 17;
constexpr size_t kKeyAntiquantOffsetIdx = 18;
constexpr size_t kValueAntiquantScaleIdx = 19;
constexpr size_t kValueAntiquantOffsetIdx = 20;
constexpr size_t kKeySharedPrefixIdx = 21;
constexpr size_t kValueSharedPrefixIdx = 22;
constexpr size_t kActualSharedPrefixLenIdx = 23;
constexpr size_t kQueryRopeIdx = 24;
constexpr size_t kKeyRopeIdx = 25;
constexpr size_t kKeyRopeAntiquantScaleIdx = 26;
constexpr size_t kNumHeadsIdx = 27;
constexpr size_t kScaleValueIdx = 28;
constexpr size_t kPreTokensIdx = 29;
constexpr size_t kNextTokensIdx = 30;
constexpr size_t kInputLayoutIdx = 31;
constexpr size_t kNumKeyValueHeadsIdx = 32;
constexpr size_t kSparseModeIdx = 33;
constexpr size_t kInnerPreciseIdx = 34;
constexpr size_t kBlockSizeIdx = 35;
constexpr size_t kAntiquantModeIdx = 36;
constexpr size_t kSoftmaxLseFlagIdx = 37;
constexpr size_t kKeyAntiquantModeIdx = 38;
constexpr size_t kvalueAntiquantModeIdx = 39;

constexpr size_t kAttentionOutIdx = 0;
constexpr size_t kSoftmaxLseOutIdx = 1;

OpsErrorCode AclnnFusedInferAttentionScore::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  auto& output_tuple = output->ToTuple();

  executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[kQueryIdx]->ToTensor(),
      GetTensorList(input[kKeyIdx]),
      GetTensorList(input[kValueIdx]),
      GetOptionalTensor(input[kPseShiftIdx]),
      GetOptionalTensor(input[kAttenMaskIdx]),
      GetIntListPair(input[kActualSeqLengthsIdx]),
      GetIntListPair(input[kActualSeqLengthsKvIdx]),
      GetOptionalTensor(input[kDeqScale1Idx]),
      GetOptionalTensor(input[kQuantScale1Idx]),
      GetOptionalTensor(input[kDeqScale2Idx]),
      GetOptionalTensor(input[kQuantScale2Idx]),
      GetOptionalTensor(input[kQuantOffset2Idx]),
      GetOptionalTensor(input[kAntiquantScaleIdx]),
      GetOptionalTensor(input[kAntiquantOffsetIdx]),
      GetOptionalTensor(input[kBlockTableIdx]),
      GetOptionalTensor(input[kQueryPaddingSizeIdx]),
      GetOptionalTensor(input[kKvPaddingSizeIdx]),
      GetOptionalTensor(input[kKeyAntiquantScaleIdx]),
      GetOptionalTensor(input[kKeyAntiquantOffsetIdx]),
      GetOptionalTensor(input[kValueAntiquantScaleIdx]),
      GetOptionalTensor(input[kValueAntiquantOffsetIdx]),
      GetOptionalTensor(input[kKeySharedPrefixIdx]),
      GetOptionalTensor(input[kValueSharedPrefixIdx]),
      GetOptionalTensor(input[kActualSharedPrefixLenIdx]),
      GetOptionalTensor(input[kQueryRopeIdx]),
      GetOptionalTensor(input[kKeyRopeIdx]),
      GetOptionalTensor(input[kKeyRopeAntiquantScaleIdx]),
      input[kNumHeadsIdx]->ToInt(),
      input[kScaleValueIdx]->ToDouble(),
      input[kPreTokensIdx]->ToInt(),
      input[kNextTokensIdx]->ToInt(),
      input[kInputLayoutIdx]->ToString(),
      input[kNumKeyValueHeadsIdx]->ToInt(),
      input[kSparseModeIdx]->ToInt(),
      input[kInnerPreciseIdx]->ToInt(),
      input[kBlockSizeIdx]->ToInt(),
      input[kAntiquantModeIdx]->ToInt(),
      input[kSoftmaxLseFlagIdx]->ToBool(),
      input[kKeyAntiquantModeIdx]->ToInt(),
      input[kvalueAntiquantModeIdx]->ToInt(),
      (*output_tuple)[kAttentionOutIdx]->ToTensor(),
      (*output_tuple)[kSoftmaxLseOutIdx]->ToTensor());

  return SUCCESS;
}

OpsErrorCode AclnnFusedInferAttentionScore::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  auto& output_tuple = output->ToTuple();

  executor_->Launch(
      workspace,
      workspaceSize,
      stream,
      input[kQueryIdx]->ToTensor(),
      GetTensorList(input[kKeyIdx]),
      GetTensorList(input[kValueIdx]),
      GetOptionalTensor(input[kPseShiftIdx]),
      GetOptionalTensor(input[kAttenMaskIdx]),
      GetIntListPair(input[kActualSeqLengthsIdx]),
      GetIntListPair(input[kActualSeqLengthsKvIdx]),
      GetOptionalTensor(input[kDeqScale1Idx]),
      GetOptionalTensor(input[kQuantScale1Idx]),
      GetOptionalTensor(input[kDeqScale2Idx]),
      GetOptionalTensor(input[kQuantScale2Idx]),
      GetOptionalTensor(input[kQuantOffset2Idx]),
      GetOptionalTensor(input[kAntiquantScaleIdx]),
      GetOptionalTensor(input[kAntiquantOffsetIdx]),
      GetOptionalTensor(input[kBlockTableIdx]),
      GetOptionalTensor(input[kQueryPaddingSizeIdx]),
      GetOptionalTensor(input[kKvPaddingSizeIdx]),
      GetOptionalTensor(input[kKeyAntiquantScaleIdx]),
      GetOptionalTensor(input[kKeyAntiquantOffsetIdx]),
      GetOptionalTensor(input[kValueAntiquantScaleIdx]),
      GetOptionalTensor(input[kValueAntiquantOffsetIdx]),
      GetOptionalTensor(input[kKeySharedPrefixIdx]),
      GetOptionalTensor(input[kValueSharedPrefixIdx]),
      GetOptionalTensor(input[kActualSharedPrefixLenIdx]),
      GetOptionalTensor(input[kQueryRopeIdx]),
      GetOptionalTensor(input[kKeyRopeIdx]),
      GetOptionalTensor(input[kKeyRopeAntiquantScaleIdx]),
      input[kNumHeadsIdx]->ToInt(),
      input[kScaleValueIdx]->ToDouble(),
      input[kPreTokensIdx]->ToInt(),
      input[kNextTokensIdx]->ToInt(),
      input[kInputLayoutIdx]->ToString(),
      input[kNumKeyValueHeadsIdx]->ToInt(),
      input[kSparseModeIdx]->ToInt(),
      input[kInnerPreciseIdx]->ToInt(),
      input[kBlockSizeIdx]->ToInt(),
      input[kAntiquantModeIdx]->ToInt(),
      input[kSoftmaxLseFlagIdx]->ToBool(),
      input[kKeyAntiquantModeIdx]->ToInt(),
      input[kvalueAntiquantModeIdx]->ToInt(),
      (*output_tuple)[kAttentionOutIdx]->ToTensor(),
      (*output_tuple)[kSoftmaxLseOutIdx]->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(fused_infer_attention_score, AclnnFusedInferAttentionScore, Ascend);
} // namespace ops
} // namespace fxrt
