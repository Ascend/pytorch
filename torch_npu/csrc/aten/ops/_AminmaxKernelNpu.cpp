#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::_aminmax(const at::Tensor& self) {
  auto min = NPUNativeFunctions::min(self);
  auto max = NPUNativeFunctions::max(self);

  return std::tie(min, max);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::_aminmax(
    const at::Tensor& self,
    const int64_t dim,
    const bool keepdim) {
  auto min = NPUNativeFunctions::min(self, {dim}, keepdim);
  auto max = NPUNativeFunctions::max(self, {dim}, keepdim);

  return std::tie(std::get<0>(min), std::get<0>(max));
}

std::tuple<at::Tensor &, at::Tensor &> NPUNativeFunctions::aminmax_out(
    const at::Tensor & self,
    c10::optional<int64_t> dim,
    bool keepdim,
    at::Tensor & min,
    at::Tensor & max) {
  if (dim.has_value()) {
    max = NPUNativeFunctions::amax_out(self, dim.value(), keepdim, max);
    min = NPUNativeFunctions::amin_out(self, dim.value(), keepdim, min);
  }
  else {
    at::SmallVector<int64_t, SIZE> dims = CalcuOpUtil::GetDimlistForTensor(self);
    max = NPUNativeFunctions::amax_out(self, dims, keepdim, max);
    min = NPUNativeFunctions::amin_out(self, dims, keepdim, min);
  }
  return std::tie(min,max);
}

} // namespace native
} // namespace at_npu
