#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::_reshape_alias(const at::Tensor& self, at::IntArrayRef sizes, at::IntArrayRef strides) {
  return self.view(sizes);
}

} // namespace native
} // namespace at_npu
