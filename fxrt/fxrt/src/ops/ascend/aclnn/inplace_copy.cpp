#include "ops/ascend/aclnn/inplace_copy.h"
#include <vector>
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
OpsErrorCode AclnnInplaceCopy::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  auto dst = input[kIndex0]->ToTensor();
  CHECK_IF_FAIL(input[kIndex1]->IsTensor());
  auto src = input[kIndex1]->ToTensor();
  non_blocking_ = input[kIndex2]->ToBool();
  bool srcNpu = src->GetDevice().type == hardware::DeviceType::NPU;
  bool dstNpu = dst->GetDevice().type == hardware::DeviceType::NPU;

  if (dstNpu && srcNpu) {
    copyMode_ = fxrt::device::CopyType::D2D;
    srcContiguous_ = src->IsContiguous();
    dstContiguous_ = dst->IsContiguous();
    if (!srcContiguous_ || !dstContiguous_) {
      executor_->GetWorkspaceSize(static_cast<uint64_t*>(workspaceSize), dst, src);
    }
  } else if (dstNpu && !srcNpu) {
    copyMode_ = fxrt::device::CopyType::H2D;
  } else if (!dstNpu && srcNpu) {
    copyMode_ = fxrt::device::CopyType::D2H;
  } else if (!dstNpu && !srcNpu) {
    copyMode_ = fxrt::device::CopyType::H2H;
  }

  if (copyMode_ != fxrt::device::CopyType::D2D &&
      (dst->Dtype() != src->Dtype() || dst->Shape() != src->Shape() || !dst->IsContiguous() || !src->IsContiguous())) {
    RT_GLOG(EXCEPTION) << "InplaceCopy H2D/D2H/H2H don't support BroadCast, DtypeCast, discontiguous src/dst yet.";
  }
  return SUCCESS;
}

OpsErrorCode AclnnInplaceCopy::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  auto dst = input[kIndex0]->ToTensor();
  auto src = input[kIndex1]->ToTensor();

  if (copyMode_ == fxrt::device::CopyType::D2D) {
    if (!srcContiguous_ || !dstContiguous_) {
      executor_->Launch(workspace, workspaceSize, stream, dst, src);
      return SUCCESS;
    }
    // For D2D copy, if both src and dst are contiguous, use direct async copy for better performance
    size_t srcSize = src->Numel() * static_cast<size_t>(dst->Dtype().GetSize());
    if (srcSize == 0) {
      return SUCCESS;
    }
    auto ret = res_manager_->AsyncCopy(dst->DataPtr(), src->DataPtr(), srcSize, copyMode_, stream);

    if (!ret) {
      RT_GLOG(ERROR) << "Call aclrtMemcpyAsync in Op InplaceCopy failed";
      return UNKNOWN_ERROR;
    }
    return SUCCESS;
  }

  size_t srcSize = src->Numel() * static_cast<size_t>(dst->Dtype().GetSize());
  if (non_blocking_) {
    res_manager_->AsyncCopy(dst->DataPtr(), src->DataPtr(), srcSize, copyMode_, stream);
  } else {
    stream_mng_->SyncStream(stream);
    res_manager_->SyncCopy(dst->DataPtr(), src->DataPtr(), srcSize, copyMode_);
  }
  return SUCCESS;
}

FXRT_REG_OP(inplace_copy, AclnnInplaceCopy, Ascend);
} // namespace ops
} // namespace fxrt
