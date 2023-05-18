#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
at::Tensor& histc_out_nocheck(const at::Tensor& self, int64_t bins, at::Scalar min, 
                              at::Scalar max, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Histogram")
      .Input(self)
      .Output(result)
      .Attr("bins", bins)
      .Attr("min", min)
      .Attr("max", max)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::histc_out(const at::Tensor& self, int64_t bins, at::Scalar min, 
                                          at::Scalar max, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    histc_out_nocheck(self, bins, min, max, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    histc_out_nocheck(self, bins, min, max, result);
  }
  return result;
}

at::Tensor NPUNativeFunctions::histc(const at::Tensor& self, int64_t bins, at::Scalar min, 
                                     at::Scalar max) {
  TORCH_CHECK(self.dtype() == at::kInt || self.dtype() == at::kFloat || self.dtype() == at::kHalf,
              "histc input only supported Int32, Float16, Float32, but got", self.dtype());
  bool is_fp = (self.dtype() == at::kInt) ? false : true;
  at::Tensor result = OpPreparation::ApplyTensor({bins}, self.options().dtype(is_fp ? at::kFloat : at::kInt), self);
  histc_out_nocheck(self, bins, min, max, result);
  return result;
}
} // namespace native
} // namespace at_npu
