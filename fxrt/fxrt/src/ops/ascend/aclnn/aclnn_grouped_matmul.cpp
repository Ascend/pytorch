#include <vector>

#include "ops/ascend/aclnn/aclnn_grouped_matmul.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
constexpr size_t kXIndex = 0;
constexpr size_t kWeightIndex = 1;
constexpr size_t kBiasIndex = 2;
constexpr size_t kScaleIndex = 3;
constexpr size_t kOffsetIndex = 4;
constexpr size_t kAntiquantScaleIndex = 5;
constexpr size_t kAntiquantOffsetIndex = 6;
constexpr size_t kPerTokenScaleIndex = 7;
constexpr size_t kGroupListIndex = 8;
constexpr size_t kActivationInputIndex = 9;
constexpr size_t kActivationQuantScaleIndex = 10;
constexpr size_t kActivationQuantOffsetIndex = 11;
constexpr size_t kSplitItemIndex = 12;
constexpr size_t kGroupTypeIndex = 13;
constexpr size_t kGroupListTypeIndex = 14;
constexpr size_t kActTypeIndex = 15;
constexpr size_t kTuningConfigIndex = 16;

void AclnnGroupedMatmul::ClearTensorList() {
  xList_.clear();
  weightList_.clear();
  biasList_.clear();
  scaleList_.clear();
  offsetList_.clear();
  antiquantScaleList_.clear();
  antiquantOffsetList_.clear();
  perTokenScaleList_.clear();
  activationInputList_.clear();
  activationQuantScaleList_.clear();
  activationQuantOffsetList_.clear();
  tuningConfigList_.clear();
  activationFeatureOutList_.clear();
  dynQuantScaleOutList_.clear();
  outputList_.clear();
}

OpsErrorCode AclnnGroupedMatmul::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  ClearTensorList();
  xList_ = input[kXIndex]->ToTuple()->ToTensorList();
  weightList_ = input[kWeightIndex]->ToTuple()->ToTensorList();
  if (input[kBiasIndex]->IsTuple()) {
    biasList_ = input[kBiasIndex]->ToTuple()->ToTensorList();
  }
  if (input[kScaleIndex]->IsTuple()) {
    scaleList_ = input[kScaleIndex]->ToTuple()->ToTensorList();
  }
  if (input[kOffsetIndex]->IsTuple()) {
    offsetList_ = input[kOffsetIndex]->ToTuple()->ToTensorList();
  }
  if (input[kAntiquantScaleIndex]->IsTuple()) {
    antiquantScaleList_ = input[kAntiquantScaleIndex]->ToTuple()->ToTensorList();
  }
  if (input[kAntiquantOffsetIndex]->IsTuple()) {
    antiquantOffsetList_ = input[kAntiquantOffsetIndex]->ToTuple()->ToTensorList();
  }
  if (input[kPerTokenScaleIndex]->IsTuple()) {
    perTokenScaleList_ = input[kPerTokenScaleIndex]->ToTuple()->ToTensorList();
  }
  if (input[kActivationInputIndex]->IsTuple()) {
    activationInputList_ = input[kActivationInputIndex]->ToTuple()->ToTensorList();
  }
  if (input[kActivationQuantScaleIndex]->IsTuple()) {
    activationQuantScaleList_ = input[kActivationQuantScaleIndex]->ToTuple()->ToTensorList();
  }
  if (input[kActivationQuantOffsetIndex]->IsTuple()) {
    activationQuantOffsetList_ = input[kActivationQuantOffsetIndex]->ToTuple()->ToTensorList();
  }
  if (input[kTuningConfigIndex]->IsTuple()) {
    tuningConfigList_ = input[kTuningConfigIndex]->ToTuple()->ToIntList();
  }
  if (output->IsTuple()) {
    outputList_ = output->ToTuple()->ToTensorList();
  }
  groupList_ = input[kGroupListIndex]->IsTensor() ? input[kGroupListIndex]->ToTensor() : nullptr;
  splitItem_ = input[kSplitItemIndex]->ToInt();
  groupType_ = input[kGroupTypeIndex]->ToInt();
  groupListType_ = input[kGroupListTypeIndex]->ToInt();
  activateType_ = input[kActTypeIndex]->ToInt();

  weightFormat_ = weightList_[kIndex0]->Format();
  if (weightFormat_ == ir::MemoryFormat::FORMAT_FRACTAL_NZ) {
    executorNz_->GetWorkspaceSize(
        static_cast<uint64_t*>(workspaceSize),
        xList_,
        weightList_,
        biasList_,
        scaleList_,
        offsetList_,
        antiquantScaleList_,
        antiquantOffsetList_,
        perTokenScaleList_,
        groupList_,
        activationInputList_,
        activationQuantScaleList_,
        activationQuantOffsetList_,
        splitItem_,
        groupType_,
        groupListType_,
        activateType_,
        tuningConfigList_,
        quantPerGroupSize_,
        outputList_,
        activationFeatureOutList_,
        dynQuantScaleOutList_);
  } else {
    executor_->GetWorkspaceSize(
        static_cast<uint64_t*>(workspaceSize),
        xList_,
        weightList_,
        biasList_,
        scaleList_,
        offsetList_,
        antiquantScaleList_,
        antiquantOffsetList_,
        perTokenScaleList_,
        groupList_,
        activationInputList_,
        activationQuantScaleList_,
        activationQuantOffsetList_,
        splitItem_,
        groupType_,
        groupListType_,
        activateType_,
        tuningConfigList_,
        outputList_,
        activationFeatureOutList_,
        dynQuantScaleOutList_);
  }
  return SUCCESS;
}

OpsErrorCode AclnnGroupedMatmul::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  if (weightFormat_ == ir::MemoryFormat::FORMAT_FRACTAL_NZ) {
    executorNz_->Launch(
        workspace,
        workspaceSize,
        stream,
        xList_,
        weightList_,
        biasList_,
        scaleList_,
        offsetList_,
        antiquantScaleList_,
        antiquantOffsetList_,
        perTokenScaleList_,
        groupList_,
        activationInputList_,
        activationQuantScaleList_,
        activationQuantOffsetList_,
        splitItem_,
        groupType_,
        groupListType_,
        activateType_,
        tuningConfigList_,
        quantPerGroupSize_,
        outputList_,
        activationFeatureOutList_,
        dynQuantScaleOutList_);
  } else {
    executor_->Launch(
        workspace,
        workspaceSize,
        stream,
        xList_,
        weightList_,
        biasList_,
        scaleList_,
        offsetList_,
        antiquantScaleList_,
        antiquantOffsetList_,
        perTokenScaleList_,
        groupList_,
        activationInputList_,
        activationQuantScaleList_,
        activationQuantOffsetList_,
        splitItem_,
        groupType_,
        groupListType_,
        activateType_,
        tuningConfigList_,
        outputList_,
        activationFeatureOutList_,
        dynQuantScaleOutList_);
  }

  return SUCCESS;
}

FXRT_REG_OP(grouped_matmul, AclnnGroupedMatmul, Ascend);
} // namespace ops
} // namespace fxrt
