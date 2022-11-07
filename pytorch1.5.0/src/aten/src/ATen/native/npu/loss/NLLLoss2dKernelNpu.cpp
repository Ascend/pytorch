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
    const Tensor& weight,
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
    Tensor& result,
    Tensor& total_weight,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  Tensor weight_tensor;
  if (weight.defined()) {
    weight_tensor = NpuUtils::format_contiguous(weight);
  } else {
    weight_tensor = at::ones(self.size(1), self.options());
  }

  if (ignore_index >= 0 && ignore_index < self.size(-1)) {
    Tensor zero = at::zeros(1, self.options());
    CalcuOpUtil::AclrtMemcpyAsync(
        {weight_tensor, ignore_index},
        weight_tensor.itemsize(),
        {zero, 0},
        weight_tensor.itemsize(),
        ACL_MEMCPY_DEVICE_TO_DEVICE);
  }

  std::string reductionStr = NpuUtils::get_reduction_str(reduction) ;
  OpCommand cmd;
  cmd.Name("NLLLoss")
      .Input(self)
      .Input(target)
      .Input(weight_tensor)
      .Attr("reduction", reductionStr)
      .Attr("ignore_index", ignore_index)
      .Output(result)
      .Output(total_weight)
      .Run();

  at::npu_reshape_out(result, result, result.sizes(), true);
  return tuple<Tensor&, Tensor&>(result, total_weight);
}

tuple<Tensor, Tensor> nll_loss2d_forward_npu(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  // Check Target Dtype
  auto scalar_type = target.scalar_type();
  TORCH_CHECK(scalar_type == at::kLong || scalar_type == at::kInt, 
      "Expected object of scalar type ", at::kLong, " or ", at::kInt, " but got scalar type ", scalar_type,
      " for argument 'target'  in call to nll_loss2d_forward");
  Tensor targetCast = target.npu_dtype_cast(at::kInt);

  auto self_input = self.contiguous();
  self_input = self_input.permute({0, 2, 3, 1});
  self_input = self_input.reshape({-1, self.size(1)});

  auto target_input = targetCast.contiguous();
  target_input = targetCast.reshape({-1});

  // calculate the output size
  auto outputSizes =
      nll_loss2d_npu_output_size(self, target, weight, reduction, ignore_index);

  // construct the output tensor of the NPU
  Tensor result =
      OpPreparation::ApplyTensor(self_input, std::get<0>(outputSizes));
  Tensor total_weight =
      OpPreparation::ApplyTensor(self_input, std::get<1>(outputSizes));

  // calculate the output result of the NPU
  nll_loss2d_forward_out_npu(
      result,
      total_weight,
      self_input,
      target_input,
      weight,
      reduction,
      ignore_index);

  return tuple<Tensor, Tensor>(result, total_weight);
}

} // namespace native
} // namespace at