#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <c10/util/Exception.h>
#include <c10/core/Storage.h>
#include <c10/npu/NPUFunctions.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/THNPUCachingHostAllocator.h"

namespace at_npu {
namespace native {

at::Tensor pin_memory(const at::Tensor& self) {
  if (self.options().backend() != c10::Backend::CPU) {
    AT_ERROR("cannot pin '", self.toString(), "' only dense CPU tensors can be pinned");
  }
  if (self.is_pinned()) {
    return self;
  }

  at::Allocator* allocator = nullptr;
  if (c10::npu::device_count() > 0) {
    allocator = getPinnedMemoryAllocator();
  }
  
  if(allocator == nullptr) {
      return self;
  }
  auto storage = c10::Storage(
      c10::Storage::use_byte_size_t(),
      at::detail::computeStorageNbytes(
          self.sizes(), self.strides(), self.dtype().itemsize()),
      allocator,
      false);
  auto tensor = at::empty({0}, self.options()).set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

TORCH_LIBRARY_IMPL(aten, CPU, m){
  m.impl("pin_memory", TORCH_FN(pin_memory));
}

}
}