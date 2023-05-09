#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor &add_relu_out_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Scalar alpha,
    at::Tensor& result){
  at::Tensor addResult = NPUNativeFunctions::add(self, other, alpha);
  OpCommand cmd;
  cmd.Name("Relu")
     .Input(addResult)
     .Output(result)
     .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::_add_relu_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& result) {
  if (!NpuUtils::check_match(&result))
  {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    add_relu_out_nocheck(self, other, alpha, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  }else{
    add_relu_out_nocheck(self, other, alpha, result);
  }
  return result;
}

at::Tensor NPUNativeFunctions::_add_relu(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  return _add_relu_out(self, other, alpha, result);
}

at::Tensor& NPUNativeFunctions::_add_relu_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  c10::SmallVector<at::Tensor, N> inputs = {self, other};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);
  return _add_relu_out(self, other, alpha, self);
}
} // namespace native
} // namespace at_npu