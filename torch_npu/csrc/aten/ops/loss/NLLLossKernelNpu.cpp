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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::nll_loss_forward_out(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    at::Tensor& result,
    at::Tensor& total_weight) {

  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  at::Tensor weight_tensor;
  if (weight.defined()) {
    weight_tensor = NpuUtils::format_contiguous(weight);
  } else {
    weight_tensor = at::ones(self.size(1), self.options());
  }

  if (ignore_index >= 0 && ignore_index < self.size(-1)) {
    at::Tensor zero = at::zeros(1, self.options());
    CalcuOpUtil::AclrtMemcpyAsync(
        {weight_tensor, ignore_index},
        weight_tensor.itemsize(),
        {zero, 0},
        weight_tensor.itemsize(),
        ACL_MEMCPY_DEVICE_TO_DEVICE);
  }

  string reductionStr = CalcuOpUtil::get_reduction_str(reduction);

  at::Tensor targetCast = target;
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

  return tuple<at::Tensor&, at::Tensor&>(result, total_weight);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::nll_loss_forward(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  // calculate the output size
  c10::SmallVector<int64_t, SIZE> outputSize = {};
  c10::SmallVector<int64_t, SIZE> totalWeightSize = {};
  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});

  if (reduction == at::Reduction::None) {
    outputSize = {self.size(0)};
  }
  auto outputSizes = tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>(
      outputSize, totalWeightSize);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      std::get<0>(outputSizes),
      self.options(),
      CalcuOpUtil::get_tensor_npu_format(self));
  at::Tensor total_weight = OpPreparation::ApplyTensorWithFormat(
      std::get<1>(outputSizes),
      self.options(),
      CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  NPUNativeFunctions::nll_loss_forward_out(
      self, target, weight, reduction, ignore_index, result, total_weight);

  return tuple<at::Tensor, at::Tensor>(result, total_weight);
}

} // namespace native
} // namespace at_npu