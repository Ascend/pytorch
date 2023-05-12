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

}
}