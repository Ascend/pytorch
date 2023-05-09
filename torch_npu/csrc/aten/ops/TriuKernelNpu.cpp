#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& triu_out_npu_nocheck(const at::Tensor& self, int64_t k, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Triu")
      .Input(self)
      .Output(result)
      .Attr("diagonal", k)
      .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::triu_out(const at::Tensor& self, int64_t k, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    triu_out_npu_nocheck(self, k, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    triu_out_npu_nocheck(self, k, result);
  }
  return result;
}

at::Tensor NPUNativeFunctions::triu(const at::Tensor& self, int64_t k) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  triu_out_npu_nocheck(self, k, result);
  return result;
}

at::Tensor& NPUNativeFunctions::triu_(at::Tensor& self, int64_t k) {
  NPUNativeFunctions::triu_out(self, k, self);
  return self;
}

} // namespace native
} // namespace at_npu
