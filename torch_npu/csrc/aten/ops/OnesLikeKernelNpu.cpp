#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {
at::Tensor NPUNativeFunctions::ones_like(const at::Tensor &self,
    c10::optional<c10::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto device = device_opt.has_value() ? device_opt.value() : self.device();
  
  if (!(device.type() == c10::DeviceType::PrivateUse1)) {
    auto result = at::empty_like(self,
                                 dtype_opt,
                                 layout_opt,
                                 device_opt,
                                 pin_memory_opt,
                                 optional_memory_format);

    return result.fill_(1.);
  }

  // construct the output tensor of the NPU
  auto other_options = c10::TensorOptions().dtype(dtype_opt)
                                           .device(device_opt)
                                           .layout(layout_opt)
                                           .pinned_memory(pin_memory_opt);
  auto options = self.options().merge_in(other_options);
  at::Tensor result = OpPreparation::ApplyTensor(self, options);
  // calculate the output result of the NPUc
  return NPUNativeFunctions::one_(result);
}

at::Tensor &NPUNativeFunctions::one_(at::Tensor &self) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor selfContiguous = NpuUtils::format_contiguous(self);
    OpCommand cmd;
    cmd.Name("OnesLike").Input(selfContiguous).Output(selfContiguous).Run();
    NpuUtils::format_fresh_view(self, selfContiguous);
  } else {
    OpCommand cmd;
    cmd.Name("OnesLike").Input(self).Output(self).Run();
  }

  return self;
}

} // namespace native
} // namespace at_npu
