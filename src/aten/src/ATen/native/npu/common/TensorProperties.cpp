#include <ATen/ATen.h>
#include <ATen/native/npu/common/InnerNpuNativeFunction.h>
namespace at {
namespace native {
Tensor contiguous_npu(const Tensor & self) {
  return contiguous_npu(self, MemoryFormat::Contiguous);
}

Tensor contiguous_npu(const Tensor& self, MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }

  TORCH_CHECK(
      memory_format != MemoryFormat::Preserve,
      "preserve memory format is unsupported by the contiguous operator");

  return self.clone();
}

}
}