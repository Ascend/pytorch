#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor true_divide_Tensor(const at::Tensor& self, const at::Tensor& divisor);

at::Tensor& true_divide_out_Tensor(const at::Tensor& self, const at::Tensor& divisor, at::Tensor& result);

at::Tensor& true_divide__Tensor(at::Tensor& self, const at::Tensor& divisor);

bool _has_compatible_shallow_copy_type(const at::Tensor &self, const at::Tensor &from);

at::Tensor pin_memory(const at::Tensor& self);

}
}
