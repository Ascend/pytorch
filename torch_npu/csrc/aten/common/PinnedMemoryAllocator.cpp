#include <stdexcept>
#include <ATen/Context.h>
#include <ATen/TensorUtils.h>
#include <c10/core/Storage.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/CPUFunctions.h>
#include <c10/core/TensorImpl.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/CachingHostAllocator.h"

namespace at_npu {
namespace native {

bool NPUNativeFunctions::is_pinned(const at::Tensor& self, c10::optional<at::Device> device)
{
    // Only CPU tensors can be pinned
    if (!self.is_cpu()) {
        return false;
    }

    return CachingHostAllocator_isPinned(self.storage().mutable_data());
}

at::Tensor NPUNativeFunctions::_pin_memory(const at::Tensor& self, c10::optional<at::Device> device)
{
    auto allocator = getPinnedMemoryAllocator();
    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        at::detail::computeStorageNbytes(
            self.sizes(),
            self.strides(),
            self.dtype().itemsize()),
        allocator,
        false);
    auto tensor = at::cpu::empty({0}, self.options()).set_(storage, 0, self.sizes(), self.strides());
    tensor.copy_(self);
    return tensor;
}

} // namespace native
} // namespace at_npu
