#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>

#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/common/ResizeNpu.h"

namespace at_npu {
namespace native {

inline const at::Tensor& resize_named_tensor_(
    const at::Tensor& self,
    c10::IntArrayRef size,
    c10::optional<c10::MemoryFormat> format)
{
    TORCH_INTERNAL_ASSERT(self.has_names());
    TORCH_CHECK(
        self.sizes() == size,
        "Cannot resize named tensor with resize_ or resize_as_ (tried to resize "
        "Tensor",
        self.names(),
        " with size ",
        self.sizes(),
        " to ",
        size,
        "). This may be caused by passing a named tensor ",
        "as an `out=` argument; please ensure that the sizes are the same. ",
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(
        !format.has_value(),
        "Unsupported memory format for named tensor resize ",
        format.value(),
        OPS_ERROR(ErrCode::NOT_SUPPORT));
    return self;
}

const at::Tensor& NPUNativeFunctions::resize_(
    const at::Tensor& self,
    c10::IntArrayRef size,
    c10::optional<c10::MemoryFormat> format)
{
    if (self.has_names()) {
        return resize_named_tensor_(self, size, format);
    }
    // because of resize _impl_npu_ only support at base format, so
    // no need to reflush NpuStorageDesc here.
    auto ks = self.key_set();
    bool is_fake_or_meta = ks.has_all(c10::DispatchKeySet(c10::BackendComponent::MetaBit)) ||
                           ks.has_all(c10::DispatchKeySet(c10::DispatchKey::Python)) ||
                           self.is_meta();
    if (!is_fake_or_meta) {
        at::Tensor temp_self = self;
        if (!FormatHelper::IsBaseFormatType(self)) {
            NPUNativeFunctions::npu_format_cast_(temp_self, FormatHelper::GetBaseFormat(self));
        }
    }
    auto* self_ = self.unsafeGetTensorImpl();
    resize_impl_npu_(self_, size, c10::nullopt);
    return self;
}

const at::Tensor& NPUNativeFunctions::resize_as_(
    const at::Tensor& self,
    const at::Tensor& the_template,
    c10::optional<c10::MemoryFormat> format)
{
    TORCH_CHECK(
        !(self.is_sparse() || the_template.is_sparse()),
        "NPU does not support sparse tensors.", OPS_ERROR(ErrCode::NOT_SUPPORT));

    if (format.has_value()) {
        auto memory_format = format.value();
        TORCH_CHECK(
            memory_format == c10::MemoryFormat::Preserve || memory_format == c10::MemoryFormat::Contiguous,
            "NPU only support format is contiguous_format or preserve_format.", OPS_ERROR(ErrCode::NOT_SUPPORT));
    }

    const at::Tensor& result = self.resize_(the_template.sizes());
    at::namedinference::propagate_names(result, the_template);
    return result;
}

} // namespace native
} // namespace at_npu
