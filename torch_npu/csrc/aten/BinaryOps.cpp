#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"


namespace at_npu {
namespace native {

at::Tensor true_divide_Tensor(const at::Tensor& self, const at::Tensor& divisor)
{
    return self.div(divisor);
}

at::Tensor& true_divide_out_Tensor(const at::Tensor& self, const at::Tensor& divisor, at::Tensor& result)
{
    return at::div_out(result, self, divisor);
}

at::Tensor& true_divide__Tensor(at::Tensor& self, const at::Tensor& divisor)
{
    return self.div_(divisor);
}

TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("true_divide.Tensor", TORCH_FN(true_divide_Tensor));
    m.impl("true_divide.out", TORCH_FN(true_divide_out_Tensor));
    m.impl("true_divide_.Tensor", TORCH_FN(true_divide__Tensor));
}
}
}