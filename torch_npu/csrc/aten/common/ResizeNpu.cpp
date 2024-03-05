#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>

#include "torch_npu/csrc/aten/common/ResizeNpu.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

const at::Tensor& NPUNativeFunctions::resize_(
    const at::Tensor& self,
    c10::IntArrayRef size,
    c10::optional<c10::MemoryFormat> format) {
  // because of resize _impl_npu_ only support at base format, so
  // no need to reflush NpuStorageDesc here.
  at::Tensor temp_self = self;
  if (!FormatHelper::IsBaseFormatType(self)) {
    NPUNativeFunctions::npu_format_cast_(temp_self, FormatHelper::GetBaseFormat(self));
  }
  auto* self_ = self.unsafeGetTensorImpl();
  resize_impl_npu_(self_, size, c10::nullopt);
  return self;
}

const at::Tensor& NPUNativeFunctions::resize_as_(
    const at::Tensor& self,
    const at::Tensor& the_template,
    c10::optional<c10::MemoryFormat> format) {
  TORCH_CHECK(
      !(self.is_sparse() || the_template.is_sparse()),
      "NPU does not support sparse tensors.", OPS_ERROR(ErrCode::NOT_SUPPORT));
  TORCH_CHECK(
      !format.has_value(), "NPU does not support specify memory_format.", OPS_ERROR(ErrCode::NOT_SUPPORT));

  const at::Tensor& result = self.resize_(the_template.sizes());
  at::namedinference::propagate_names(result, the_template);
  return result;
}

} // namespace native
} // namespace at_npu
