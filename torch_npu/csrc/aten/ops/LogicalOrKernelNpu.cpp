#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

at::Tensor& logical_or_out_npu_nocheck(   
    const at::Tensor& self, 
    const at::Tensor& other,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("LogicalOr")
    .Input(self)
    .Input(other)
    .Output(result)
    .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::logical_or_out(
    const at::Tensor& self, 
    const at::Tensor& other, 
    at::Tensor& result) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(self),
      result.scalar_type(),
      outputSize);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    logical_or_out_npu_nocheck(self, other, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    logical_or_out_npu_nocheck(self, other, result);
  }
  return result;
}

at::Tensor NPUNativeFunctions::logical_or(const at::Tensor& self, const at::Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  logical_or_out_npu_nocheck(self, other, result);
  result = NPUNativeFunctions::npu_dtype_cast(result, at::kBool);
  return result;
}

at::Tensor& NPUNativeFunctions::logical_or_(at::Tensor& self, const at::Tensor& other) {
  OpPreparation::CheckMemory({self, other},{self});
  logical_or_out(self, other, self);
  return self;
}

} // namespace native
} // namespace at_npu
