#include <vector>

#include "ops/ascend/aclnn/aclnn_contiguous.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"
#include "hardware/ascend/res_manager/ascend_res_manager.h"

namespace fxrt {
namespace ops {
OpsErrorCode AclnnContiguous::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  const auto& inputTensor = input[kIndex0]->ToTensor();
  srcContiguous_ = inputTensor->IsContiguous();
  if (!srcContiguous_) {
    executor_->GetWorkspaceSize(static_cast<uint64_t*>(workspaceSize), output->ToTensor(), inputTensor);
  }
  return SUCCESS;
}

OpsErrorCode AclnnContiguous::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  const auto& inputTensor = input[kIndex0]->ToTensor();
  const auto& outTensor = output->ToTensor();

  if (!srcContiguous_) {
    executor_->Launch(workspace, workspaceSize, stream, outTensor, inputTensor);
    return SUCCESS;
  }

  // Input tensor is already contiguous, perform direct memory copy
  auto srcSize = inputTensor->Numel() * inputTensor->Dtype().GetSize();
  auto dstSize = outTensor->Numel() * outTensor->Dtype().GetSize();
  if (srcSize > dstSize) {
    RT_GLOG(EXCEPTION) << "Unexpected input and output size mismatch, src size is " << srcSize << ", dst size is "
                       << dstSize;
  }
  auto ret = fxrt::device::ascend::AscendResManager::MemcpyDeviceToDevice(
      outTensor->DataPtr(), dstSize, inputTensor->DataPtr(), dstSize, stream);

  if (!ret) {
    RT_GLOG(ERROR) << "Call aclrtMemcpyAsync in Op Contiguous failed";
    return UNKNOWN_ERROR;
  }
  return SUCCESS;
}

FXRT_REG_OP(contiguous, AclnnContiguous, Ascend);
} // namespace ops
} // namespace fxrt
