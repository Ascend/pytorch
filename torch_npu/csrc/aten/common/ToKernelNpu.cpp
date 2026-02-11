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
static inline c10::Device ensure_has_index(c10::Device device)
{
    if (device.is_cpu() || device.has_index()) {
        return device;
    }
    const c10::impl::DeviceGuardImplInterface* impl =
        c10::impl::getDeviceGuardImpl(device.type());
    return impl->getDevice();
}

at::Tensor NPUNativeFunctions::_to_copy(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype,
    c10::optional<c10::Layout> layout,
    c10::optional<c10::Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    c10::optional<c10::MemoryFormat> optional_memory_format)
{
    if (dtype.has_value() && !layout.has_value() && !device.has_value()) {
        if (self.dtype() == dtype) {
            return self;
        }
        if (dtype == at::ScalarType::Double) {
            TORCH_NPU_WARN_ONCE(
                "Device do not support double dtype now, "
                "dtype cast replace with float.");
        }
        dtype = (dtype == at::ScalarType::Double) ? at::ScalarType::Float : dtype;
    }

    c10::TensorOptions options_ = c10::TensorOptions()
        .dtype(dtype)
        .layout(layout)
        .device(device);

    auto options = self.options().merge_in(options_);

    if (layout.has_value()) {
        TORCH_CHECK(
            self.layout() == layout.value(),
            "to(options) doesn't support converting to a different layout, "
            "but got self.layout being ",
            self.layout(),
            " and options.layout set as ",
            layout.value(), OPS_ERROR(ErrCode::NOT_SUPPORT));
    }

    if (device.has_value()) {
        options = options.device(ensure_has_index(device.value()));
    }

    if (optional_memory_format.has_value()) {
        TORCH_CHECK(
            optional_memory_format.value() == c10::MemoryFormat::Preserve ||
            optional_memory_format.value() == c10::MemoryFormat::Contiguous,
            "Only contiguous_format or preserve_format is supported.", OPS_ERROR(ErrCode::NOT_SUPPORT));
        options = options.memory_format(optional_memory_format.value());
    } else {
        if (torch_npu::utils::is_npu(self)) {
            options = options.memory_format(c10::MemoryFormat::Contiguous);
        } else {
            // keep the same as cpu default memory format: Preserve
            options = options.memory_format(c10::MemoryFormat::Preserve);
        }
    }
    TORCH_CHECK(
        options.requires_grad_opt() == c10::nullopt,
        "to(options) expects unset requires_grad flag, but got "
        "options.requires_grad set as ",
        options.requires_grad(), OPS_ERROR(ErrCode::PARAM));

    bool pin_out = non_blocking && torch_npu::utils::is_npu(self) && options.device().is_cpu() &&
                    (options.layout() == c10::kStrided);

    c10::MemoryFormat memory_format = options.memory_format_opt().value_or(c10::MemoryFormat::Contiguous);
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

} // namespace native
} // namespace at_npu
