#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::zeros_out(at::IntArrayRef size, at::Tensor& result) {
  result.resize_(size);
  return result.zero_();
}

at::Tensor zeros_without_symint(
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  at::TensorOptions option = option.dtype(dtype_opt)
                                   .layout(layout_opt)
                                   .device(device_opt)
                                   .pinned_memory(pin_memory_opt);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(size, option, ACL_FORMAT_ND);
  return result.zero_();
}

at::Tensor NPUNativeFunctions::zeros_symint(c10::SymIntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  return zeros_without_symint(c10::asIntArrayRefUnchecked(size), dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

at::Tensor NPUNativeFunctions::zeros(
    at::IntArrayRef size,
    c10::optional<at::DimnameList> names,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  return zeros_without_symint(size, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

} // namespace native
} // namespace at_npu