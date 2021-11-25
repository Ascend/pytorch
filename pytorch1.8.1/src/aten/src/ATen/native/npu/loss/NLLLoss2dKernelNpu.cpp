// Copyright (c) 2020, Huawei Technologies.All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

namespace {

tuple<SmallVector<int64_t, SIZE>, SmallVector<int64_t, SIZE>> nll_loss2d_npu_output_size(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    int64_t ignore_index) {
  SmallVector<int64_t, SIZE> outputSize;
  SmallVector<int64_t, SIZE> totalWeightSize;

  if (reduction == Reduction::None) {
    outputSize = {self.size(0), self.size(2), self.size(3)};
  }

  return tuple<SmallVector<int64_t, SIZE>, SmallVector<int64_t, SIZE>>(
      outputSize, totalWeightSize);
}
} // namespace

tuple<Tensor&, Tensor&> nll_loss2d_forward_out_npu(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    Tensor& result,
    Tensor& total_weight) {
  Tensor weight = c10::value_or_else(weight_opt, [] {return Tensor();});
  Tensor weight_tensor;
  if (weight.defined()) {
    weight_tensor = NpuUtils::format_contiguous(weight);
  } else {
    weight_tensor = at::ones(self.size(1), self.options());
  }

  if (ignore_index >= 0) {
    Tensor zero = at::zeros(1, self.options());
    void* ignore_ptr = reinterpret_cast<uint8_t*>(weight_tensor.data_ptr()) +
        ignore_index * weight_tensor.itemsize();
    CalcuOpUtil::AclrtMemcpyAsync(
        ignore_ptr,
        weight_tensor.itemsize(),
        reinterpret_cast<void*>(zero.data_ptr()),
        weight_tensor.itemsize(),
        ACL_MEMCPY_DEVICE_TO_DEVICE);
  }

  OpPreparation::CheckMemory({self, target, weight_tensor}, {result, total_weight});

  auto reductionStr = CalcuOpUtil::get_reduction_str(reduction) ;
  OpCommand cmd;
  cmd.Name("NLLLoss")
      .Input(self)
      .Input(target)
      .Input(weight_tensor)
      .Attr("reduction", reductionStr)
      .Output(result)
      .Output(total_weight)
      .Run();

  return tuple<Tensor&, Tensor&>(result, total_weight);
}

tuple<Tensor, Tensor> nll_loss2d_forward_npu(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  auto self_input = self.contiguous();
  self_input = self_input.permute({0, 2, 3, 1});
  self_input = self_input.reshape({-1, self.size(1)});

  auto target_input = target.contiguous();
  target_input = target.reshape({-1});

  // calculate the output size
  auto outputSizes =
      nll_loss2d_npu_output_size(self, target, reduction, ignore_index);

  // construct the output tensor of the NPU
  Tensor result =
      OpPreparation::ApplyTensor(self_input, std::get<0>(outputSizes));
  Tensor total_weight =
      OpPreparation::ApplyTensor(self_input, std::get<1>(outputSizes));

  // calculate the output result of the NPU
  nll_loss2d_forward_out_npu(
      self_input,
      target_input,
      weight_opt,
      reduction,
      ignore_index,
      result,
      total_weight);

  return tuple<Tensor, Tensor>(result, total_weight);
}
TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("nll_loss2d_forward", TORCH_FN(nll_loss2d_forward_npu));
  m.impl("nll_loss2d_forward.output", TORCH_FN(nll_loss2d_forward_out_npu));
}
} // namespace native
} // namespace at