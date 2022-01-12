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
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

tuple<Tensor&, Tensor&> nll_loss_forward_npu_nocheck(
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

  std::string reductionStr = NpuUtils::get_reduction_str(reduction);

  Tensor targetCast = target;
  auto scalar_type = target.scalar_type();
  if (scalar_type == at::kLong) {
    targetCast = target.npu_dtype_cast(at::kInt);
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

tuple<Tensor&, Tensor&> nll_loss_forward_out_npu(
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
    weight_tensor = ones_npu(self.size(1), self.options());
  }
  SmallVector<int64_t, SIZE> outputSize = {};
  if (reduction == Reduction::None) {
    outputSize = {self.size(0)};
  }
  OpPipeWithMultiOut<Tensor&, Tensor&> pipe(result, total_weight);
  return pipe.FixOutputSizeAndFormat<0>({self, target, weight_tensor}, self, ACL_FORMAT_ND, outputSize)
            .FixOutputSizeAndFormat<1>({self, target, weight_tensor}, self, ACL_FORMAT_ND, {})
            .Call([&self, &target, &weight, &reduction, &ignore_index](Tensor& result, Tensor& total_weight) {
              nll_loss_forward_npu_nocheck(result, total_weight, self, target, weight, reduction, ignore_index);})
            .ReturnRef<Tensor&, Tensor&>();
}

tuple<Tensor, Tensor> nll_loss_forward_npu(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  // ND case
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_CHECK(
      target.dim() == 1,
      "1D target tensor expected, multi-target not supported");
  TORCH_CHECK(
      self.size(0) == target.size(0),
      "size mismatch (got input: ",
      self.sizes(),
      ", target: ",
      target.sizes(),
      ")");
  SmallVector<int64_t, SIZE> outputSize = {};
  SmallVector<int64_t, SIZE> totalWeightSize = {};

  if (reduction == Reduction::None) {
    outputSize = {self.size(0)};
  }
  auto outputSizes = tuple<SmallVector<int64_t, SIZE>, SmallVector<int64_t, SIZE>>(
      outputSize, totalWeightSize);

  // Special output, output' dim is <= 1 fixedly！
  Tensor result = OpPreparation::ApplyTensorWithFormat(self, outputSize, ACL_FORMAT_ND);
  Tensor total_weight = OpPreparation::ApplyTensorWithFormat(self, totalWeightSize, ACL_FORMAT_ND);

  nll_loss_forward_npu_nocheck(
      result, total_weight, self, target, weight, reduction, ignore_index);

  return tuple<Tensor, Tensor>(result, total_weight);
}

} // namespace native
} // namespace at