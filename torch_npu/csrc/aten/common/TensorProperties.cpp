#include <ATen/ATen.h>

#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::contiguous(const at::Tensor& self, c10::MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }

  TORCH_CHECK(
      memory_format != c10::MemoryFormat::Preserve,
      "preserve memory format is unsupported by the contiguous operator");
  return self.clone();
}

at::Tensor NPUNativeFunctions::format_contiguous(const at::Tensor &self) {
  return NpuUtils::format_contiguous(self);
}

bool NPUNativeFunctions::is_set_to(const at::Tensor& self, const at::Tensor& src) {
  if (self.storage().unsafeGetStorageImpl() == src.storage().unsafeGetStorageImpl() &&
      self.storage_offset() == src.storage_offset() && self.dim() == src.dim() &&
      NPUNativeFunctions::get_storage_size(self) == NPUNativeFunctions::get_storage_size(src) &&
      NPUNativeFunctions::get_npu_format(self) == NPUNativeFunctions::get_npu_format(src)) {
    for (const auto d : c10::irange(self.dim())) {
      if (self.size(d) != src.size(d) || self.stride(d) != src.stride(d)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

} // namespace native
} // namespace at_npu
