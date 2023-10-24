#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::_copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
  TORCH_CHECK(self.sizes() == dst.sizes(), "_copy_from_and_resize now only support copy with same size!");
  TORCH_CHECK(self.is_cpu() && dst.device().is_privateuseone(),
      "_copy_from_and_resize now only support copy from cpu tensor to npu tensor, but got src tensor device is ",
      self.device(), " and dst device is ", dst.device());
  dst.copy_(self);
  return dst;
}

}
}
