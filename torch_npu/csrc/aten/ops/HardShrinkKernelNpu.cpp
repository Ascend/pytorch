#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& hardshrink_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar lambd) {
  OpCommand cmd;
  cmd.Name("HardShrink")
    .Input(self)
    .Attr("lambd", lambd)
    .Output(result).Run();
    
    return result;
}

at::Tensor NPUNativeFunctions::hardshrink(const at::Tensor& self, const at::Scalar& lambd) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  hardshrink_nocheck(result, self, lambd);

  return result;
}

} // namespace native
} // namespace at_npu