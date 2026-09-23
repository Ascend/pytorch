#include <vector>

#include "ops/ascend/aclnn/aclnn_quant_matmul.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"
#include "ir/tensor/tensor.h"
#include "ir/common/dtype.h"

namespace fxrt {
namespace ops {
constexpr size_t kX1Index = 0;
constexpr size_t kX2Index = 1;
constexpr size_t kScaleIndex = 2;
constexpr size_t kOffsetIndex = 3;
constexpr size_t kPerTokenScaleIndex = 4;
constexpr size_t kBiasIndex = 5;
constexpr size_t kOutputDtypeIndex = 6;
constexpr size_t kGroupSizesIndex = 7;

OpsErrorCode AclnnQuantMatmul::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  x1_ = input[kX1Index]->ToTensor();
  x2_ = input[kX2Index]->ToTensor();
  scale_ = input[kScaleIndex]->ToTensor();
  offset_ = input[kOffsetIndex]->IsTensor() ? input[kOffsetIndex]->ToTensor() : nullptr;
  pertokenScale_ = input[kPerTokenScaleIndex]->IsTensor() ? input[kPerTokenScaleIndex]->ToTensor() : nullptr;
  bias_ = input[kBiasIndex]->IsTensor() ? input[kBiasIndex]->ToTensor() : nullptr;

  bool isW8A8 = x1_->Dtype() == ir::DataType::Int8 && x2_->Dtype() == ir::DataType::Int8;

  if (!isW8A8) {
    RT_GLOG(EXCEPTION) << "Op quant_matmul only supports W8A8 (Int8 x Int8) for now, got x1 dtype: "
                       << x1_->Dtype().ToString() << ", x2 dtype: " << x2_->Dtype().ToString();
  }

  auto outputDtype = output->ToTensor()->Dtype();
  if (scale_->Dtype() == ir::DataType::Float32 && pertokenScale_ == nullptr && outputDtype != ir::DataType::BFloat16 &&
      outputDtype != ir::DataType::Int32) {
    RT_GLOG(EXCEPTION) << "Op quant_matmul does not support Float32 scale, null pertokenScale, and output dtype: "
                       << outputDtype.ToString();
  }

  ir::MemoryFormat weightFormat = x2_->Format();
  isWeightNz_ = (weightFormat == ir::MemoryFormat::FORMAT_FRACTAL_NZ);

  if (isWeightNz_) {
    int64_t groupSize = 0;
    executorNz_->GetWorkspaceSize(
        static_cast<uint64_t*>(workspaceSize),
        x1_,
        x2_,
        pertokenScale_,
        scale_,
        yscale_,
        x1Offset_,
        offset_,
        yOffset_,
        bias_,
        transposeX1_,
        transposeX2_,
        groupSize,
        output->ToTensor());
  } else {
    executor_->GetWorkspaceSize(
        static_cast<uint64_t*>(workspaceSize),
        x1_,
        x2_,
        scale_,
        offset_,
        pertokenScale_,
        bias_,
        transposeX1_,
        transposeX2_,
        output->ToTensor());
  }

  return SUCCESS;
}

OpsErrorCode AclnnQuantMatmul::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  if (isWeightNz_) {
    int64_t groupSize = 0;
    executorNz_->Launch(
        workspace,
        workspaceSize,
        stream,
        x1_,
        x2_,
        pertokenScale_,
        scale_,
        yscale_,
        x1Offset_,
        offset_,
        yOffset_,
        bias_,
        transposeX1_,
        transposeX2_,
        groupSize,
        output->ToTensor());
  } else {
    executor_->Launch(
        workspace,
        workspaceSize,
        stream,
        x1_,
        x2_,
        scale_,
        offset_,
        pertokenScale_,
        bias_,
        transposeX1_,
        transposeX2_,
        output->ToTensor());
  }

  return SUCCESS;
}

FXRT_REG_OP(quant_matmul, AclnnQuantMatmul, Ascend);
} // namespace ops
} // namespace fxrt
