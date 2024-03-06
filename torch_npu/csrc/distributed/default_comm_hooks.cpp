#include <c10d/default_comm_hooks.hpp>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <c10d/ProcessGroup.hpp>
#include <c10d/comm.hpp>
#include <torch/torch.h>

#include "torch_npu/csrc/core/npu/NPUException.h"

namespace c10d {

c10::intrusive_ptr<c10::ivalue::Future> AllReduceCommHook::runHook(
    GradBucket& bucket) {
  std::vector<at::Tensor> tensors = {bucket.getBufferRef()};
  // Apply the division first to avoid overflow, especially for FP16.
  tensors[0] /= state_->getSize();
  return state_->allreduce(tensors)->getFuture();
}

c10::intrusive_ptr<c10::ivalue::Future> FP16CompressCommHook::runHook(
    GradBucket& bucket) {
  auto compressed_tensor = bucket.getBufferRef().to(torch::kFloat16);
  // Apply the division first to avoid overflow.
  compressed_tensor /= state_->getSize();
  std::vector<at::Tensor> tensors = {compressed_tensor};

  auto allreduce_fut = state_->allreduce(tensors)->getFuture();
  auto decompressed_tensor = bucket.getBufferRef();
  auto decompress = [decompressed_tensor](c10::ivalue::Future& allreduce_fut) {
    auto result = allreduce_fut.value();
    TORCH_INTERNAL_ASSERT(
        result.isTensorList(),
        "ProcessGroup::allreduce should return TensorList", DIST_ERROR(ErrCode::INTERNAL));

    auto reduce_tensor = result.toTensorVector()[0];
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        reduce_tensor.scalar_type() == at::ScalarType::Half,
        "Expected reduced tensor to be fp16 in FP16CompressHook, but got type ",
        reduce_tensor.scalar_type(), DIST_ERROR(ErrCode::TYPE)
    );
    decompressed_tensor.copy_(reduce_tensor);
    return c10::IValue(decompressed_tensor);
  };

  return allreduce_fut->then(decompress, allreduce_fut->elementType());
}

c10::intrusive_ptr<c10::ivalue::Future> _AllReduceBySumCommHook::runHook(GradBucket& bucket) {
  std::vector<at::Tensor> tensors = {bucket.getBufferRef()};
  return state_->allreduce(tensors)->getFuture();
}

} // namespace c10d
