#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/Optional.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

// Take a c10::Device that may not have device_index set (i.e., having it as -1
// representing the current device) and return the corresponding c10::Device
// according to the actual device at the time of this function call.  No-op
// if the device_index is set.
static inline c10::Device ensure_has_index(c10::Device device) {
  if (device.is_cpu() || device.has_index()) {
    return device;
  }
  const c10::impl::DeviceGuardImplInterface* impl =
      c10::impl::getDeviceGuardImpl(device.type());
  return impl->getDevice();
}

static inline at::Tensor to_impl_npu(
    const at::Tensor& self,
    const c10::TensorOptions& options,
    bool non_blocking,
    bool copy) {
  auto memory_format = options.memory_format_opt().value_or(
      c10::MemoryFormat::Contiguous); // Here cpu's default value is Preserve
  if (self.dtype() == options.dtype() && self.layout() == options.layout() &&
      self.device() == options.device() && !copy &&
      (memory_format == c10::MemoryFormat::Preserve ||
       self.suggest_memory_format() == memory_format)) {
    return self;
  }

    bool pin_out = non_blocking && torch_npu::utils::is_npu(self) && options.device().is_cpu() &&
                    (options.layout() == c10::kStrided);

  if (memory_format == c10::MemoryFormat::Preserve) {
    if (self.is_non_overlapping_and_dense()) {
      // Copy all strides
      auto r = at::empty_strided(
          self.sizes(), self.strides(), options.memory_format(c10::nullopt).pinned_memory(pin_out));
      r.copy_(self, non_blocking);
      return r;
    } else {
      memory_format = self.suggest_memory_format();
    }
  }
  // See Note [Explicit nullopt c10::MemoryFormat argument]
  auto r = at::empty(
      self.sizes(), options.memory_format(memory_format).pinned_memory(pin_out), c10::nullopt);
  r.copy_(self, non_blocking);
  return r;
}

at::Tensor NPUNativeFunctions::to(
    const at::Tensor &self,
    c10::optional<at::ScalarType> dtype,
    c10::optional<c10::Layout> layout,
    c10::optional<c10::Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format)
{
    if (device.has_value() && device.value().is_cpu() && optional_memory_format.has_value()) {
        TORCH_CHECK(
            optional_memory_format.value() == c10::MemoryFormat::Preserve ||
            optional_memory_format.value() == c10::MemoryFormat::Contiguous,
            "Only contiguous_format or preserve_format is supported.");
    } else {
        TORCH_CHECK(
            !optional_memory_format.has_value(),
            "NPU not support specify memory_format.", OPS_ERROR(ErrCode::NOT_SUPPORT));
    }

    c10::TensorOptions options_ = c10::TensorOptions().dtype(dtype).layout(layout).device(device);
    TORCH_CHECK(
        !(options_.has_memory_format() && optional_memory_format.has_value()),
        "Cannot set memory_format both in c10::TensorOptions and explicit argument; please delete "
        "the redundant setter.", OPS_ERROR(ErrCode::PARAM));
    auto options =
        options_.merge_in(c10::TensorOptions().memory_format(optional_memory_format));

    TORCH_CHECK(
        options.requires_grad_opt() == c10::nullopt,
        "to(options) expects unset requires_grad flag, but got "
        "options.requires_grad set as ",
        options.requires_grad(), OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(
        !options.has_layout() || self.layout() == options.layout(),
        "to(options) doesn't support converting to a different layout, "
        "but got self.layout being ",
        self.layout(),
        " and options.layout set as ",
        options.layout(), OPS_ERROR(ErrCode::NOT_SUPPORT));

    if (options.has_device()) {
        options = options.device(ensure_has_index(options.device()));
    }
    auto specified_options = self.options().merge_in(options);
    return to_impl_npu(self, specified_options, non_blocking, copy);
}

at::Tensor NPUNativeFunctions::to(
    const at::Tensor& self,
    c10::Device device,
    at::ScalarType dtype,
    bool non_blocking,
    bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  device = ensure_has_index(device);
  return to_impl_npu(
      self,
      self.options().device(device).dtype(dtype).memory_format(
          optional_memory_format),
      non_blocking,
      copy);
}

at::Tensor NPUNativeFunctions::to(
    const at::Tensor& self,
    at::ScalarType dtype,
    bool non_blocking,
    bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  if (self.dtype() == dtype) {
    return self;
  }
  if (at::ScalarType::Double == dtype) {
    static auto warn_once = []() {
        std::cout << "Warning: Device do not support double dtype now, " \
                     "dtype cast repalce with float." << std::endl;
        return true;
    }();
  }
  dtype = (at::ScalarType::Double == dtype) ? at::ScalarType::Float : dtype;
  return custom_ops::npu_dtype_cast(self, dtype);
}

at::Tensor NPUNativeFunctions::to(
    const at::Tensor& self,
    const at::Tensor& other,
    bool non_blocking,
    bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto options = other.options();
  return to_impl_npu(
      self, options.memory_format(optional_memory_format), non_blocking, copy);
}
} // namespace native
} // namespace at_npu
