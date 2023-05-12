#include <ATen/WrapDimUtilsMulti.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu{
namespace native{

at::Tensor NPUNativeFunctions::count_nonzero(
    const at::Tensor &self,
    c10::IntArrayRef dim) {
  return NPUNativeFunctions::sum((self != 0), dim, false, at::ScalarType::Long);
}

at::Tensor NPUNativeFunctions::count_nonzero(
    const at::Tensor &self,
    c10::optional<int64_t> dim) {
  if (dim.has_value()) {
    return NPUNativeFunctions::count_nonzero(self, at::IntArrayRef{dim.value()});
  }
  return NPUNativeFunctions::count_nonzero(self, at::IntArrayRef{});
}

} // namespace native
} // namespace at_npu
