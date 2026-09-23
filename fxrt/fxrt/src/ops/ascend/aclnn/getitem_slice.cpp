#include <vector>

#include "ops/ascend/aclnn/getitem_slice.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
OpsErrorCode AclnnGetItemSlice::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  starts_ = input[kIndex1]->ToTuple()->ToIntList();
  ends_ = input[kIndex2]->ToTuple()->ToIntList();
  axes_ = input[kIndex3]->ToTuple()->ToIntList();
  steps_ = input[kIndex4]->ToTuple()->ToIntList();
  auto src = input[0]->ToTensor();
  auto dst = output->ToTensor();
  skipLaunch_ = std::any_of(dst->Shape().begin(), dst->Shape().end(), [](int64_t shape) { return shape == 0; });
  if (skipLaunch_) {
    RT_VLOG(VL_OPS) << "For GetItemSlice, the dst shape is " << dst->Shape() << " which contains 0, skipping launch.";
    return SUCCESS;
  }

  needSqueeze_ = src->Shape().size() != dst->Shape().size();
  if (needSqueeze_) {
    // Adaptation for frontend getitem operation. For example, when x has shape (3, 3, 3, 4) and x[1, ..., 1:4:2] is
    // executed, the actual output shape is (3, 3, 2), which is equivalent to squeezing dimensions with size 1. However,
    // the backend aclnnSliceV2 requires the input and output shapes to have the same number of dimensions, so we need
    // to re-infer the shape and add back the original axes with size 1.
    dst_ = dst->ShallowClone();
    std::vector<int64_t> sliceSizes(axes_.size());
    CHECK_IF_FAIL(starts_.size() == ends_.size());
    CHECK_IF_FAIL(ends_.size() == axes_.size());
    CHECK_IF_FAIL(axes_.size() == steps_.size());
    for (size_t i = 0; i < starts_.size(); i++) {
      CHECK_IF_FAIL(steps_[i] != 0);
      sliceSizes[i] = (ends_[i] - starts_[i] + steps_[i] - 1) / steps_[i];
    }
    std::vector<int64_t> shapes = src->Shape();
    for (size_t i = 0; i < axes_.size(); i++) {
      shapes[axes_[i]] = sliceSizes[i];
    }
    auto dstShape = dst->Shape();
    auto srcSize = std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies<int64_t>());
    auto dstSize = std::accumulate(dstShape.begin(), dstShape.end(), 1, std::multiplies<int64_t>());
    if (srcSize != dstSize) {
      RT_GLOG(EXCEPTION) << "For GetItemSlice, the src shape " << src->Shape() << " cannot be slice to dst shape "
                         << dstShape << ". With starts: " << starts_ << ", ends: " << ends_ << ", axes: " << axes_
                         << ", steps: " << steps_;
    }

    std::vector<int64_t> strides(shapes.size());
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(shapes.size()) - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shapes[i];
    }
    dst_->SetShape(shapes);
    dst_->SetStrides(strides);
    executor_->GetWorkspaceSize(static_cast<uint64_t*>(workspaceSize), src, starts_, ends_, axes_, steps_, dst_);
  } else {
    executor_->GetWorkspaceSize(static_cast<uint64_t*>(workspaceSize), src, starts_, ends_, axes_, steps_, dst);
  }
  return SUCCESS;
}

OpsErrorCode AclnnGetItemSlice::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  if (skipLaunch_) {
    return SUCCESS;
  }
  if (needSqueeze_) {
    executor_->Launch(workspace, workspaceSize, stream, input[0]->ToTensor(), starts_, ends_, axes_, steps_, dst_);
  } else {
    executor_->Launch(
        workspace, workspaceSize, stream, input[0]->ToTensor(), starts_, ends_, axes_, steps_, output->ToTensor());
  }
  return SUCCESS;
}

bool AclnnGetItemSlice::NeedLaunch() {
  return !skipLaunch_;
}

FXRT_REG_OP(getitem_slice, AclnnGetItemSlice, Ascend);
} // namespace ops
} // namespace fxrt
