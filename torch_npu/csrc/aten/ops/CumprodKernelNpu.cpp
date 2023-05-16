#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& cumprod_out_nocheck(at::Tensor& result, const at::Tensor& self, int64_t dim) {
  at::Scalar axis = dim;
  OpCommand cmd;
  cmd.Name("Cumprod")
      .Input(self)
      .Input(axis, at::kLong)
      .Attr("exclusive", (bool)false)
      .Attr("reverse", (bool)false)
      .Output(result)
      .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::cumprod_out(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    at::Tensor& result) {
  at::ScalarType dst_type = self.scalar_type();
  if (dtype.has_value()) {
    dst_type = dtype.value();
  } else if (result.defined()) {
    dst_type = result.scalar_type();
  }

  at::Tensor self_copy = self.scalar_type() == dst_type ? self :
      NPUNativeFunctions::npu_dtype_cast(self, dst_type);
  OpPreparation::CheckOut(
      {self_copy},
      result,
      CalcuOpUtil::GetTensorNpuFormat(result),
      dst_type,
      self_copy.sizes());

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    cumprod_out_nocheck(contiguous_result, self_copy, dim);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    cumprod_out_nocheck(result, self_copy, dim);
  }
  return result;
}

at::Tensor& NPUNativeFunctions::cumprod_out(
    const at::Tensor& self,
    at::Dimname dim,
    c10::optional<at::ScalarType> dtype,
    at::Tensor& result) {
  return NPUNativeFunctions::cumprod_out(self, dimname_to_position(self, dim), dtype, result);
}

at::Tensor& NPUNativeFunctions::cumprod_(
    at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype) {
  TORCH_CHECK(
      !dtype.has_value() || (self.scalar_type() == dtype.value()),
      "provided dtype must match the dtype of self tensor in cumprod. Got ",
      toString(self.scalar_type()),
      " and ",
      toString(dtype.value()),
      ".");
  return NPUNativeFunctions::cumprod_out(self, dim, dtype, self);
}

at::Tensor& NPUNativeFunctions::cumprod_(at::Tensor& self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  return NPUNativeFunctions::cumprod_(self, dimname_to_position(self, dim), dtype);
}

} // namespace native
} // namespace at_npu
