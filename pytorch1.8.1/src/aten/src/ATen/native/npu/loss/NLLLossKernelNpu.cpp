// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
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

tuple<Tensor&, Tensor&> nll_loss_forward_out_npu(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    Tensor& result,
    Tensor& total_weight) {

  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});
  Tensor weight_tensor;
  if (weight.defined()) {
    weight_tensor = NpuUtils::format_contiguous(weight);
  } else {
    weight_tensor = at::ones(self.size(1), self.options());
  }

  if (ignore_index >= 0 && ignore_index < self.size(-1)) {
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

  string reductionStr = CalcuOpUtil::get_reduction_str(reduction);

  Tensor targetCast = target;
  auto scalar_type = target.scalar_type();
  if (scalar_type == at::kLong) {
    targetCast = target.to(at::kInt);
  }  else if (scalar_type == at::kInt) {
    ;
  }
  else {
    AT_ERROR("Expected object of scalar type ", at::kLong, " or ", at::kInt, " but got scalar type ", scalar_type,
        " for argument 'target'  in call to nll_loss_forward");
  }

  OpCommand cmd;
  cmd.Name("NLLLoss")
      .Input(self)
      .Input(targetCast)
      .Input(weight_tensor)
      .Output(result)
      .Output(total_weight)
      .Attr("reduction", reductionStr)
      .Attr("ignore_index", ignore_index)
      .Run();

  return tuple<Tensor&, Tensor&>(result, total_weight);
}

tuple<Tensor, Tensor> nll_loss_forward_npu(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  // calculate the output size
  SmallVector<int64_t, SIZE> outputSize = {};
  SmallVector<int64_t, SIZE> totalWeightSize = {};
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

  if (reduction == Reduction::None) {
    outputSize = {self.size(0)};
  }
  auto outputSizes = tuple<SmallVector<int64_t, SIZE>, SmallVector<int64_t, SIZE>>(
      outputSize, totalWeightSize);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      std::get<0>(outputSizes),
      self.options(),
      CalcuOpUtil::get_tensor_npu_format(self));
  Tensor total_weight = at::empty_with_format(
      std::get<1>(outputSizes),
      self.options(),
      CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  nll_loss_forward_out_npu(
      self, target, weight, reduction, ignore_index, result, total_weight);

  return tuple<Tensor, Tensor>(result, total_weight);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("nll_loss_forward", TORCH_FN(nll_loss_forward_npu));
  m.impl("nll_loss_forward.output", TORCH_FN(nll_loss_forward_out_npu));
}
} // namespace native
} // namespace at