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

tuple<at::Tensor&, at::Tensor&> nll_loss_forward_npu_nocheck(
    at::Tensor& result,
    at::Tensor& total_weight,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
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

  string reductionStr = CalcuOpUtil::GetReductionStr(reduction);

  at::Tensor targetCast = target;
  auto scalar_type = target.scalar_type();
  if (scalar_type == at::kLong) {
    targetCast = NPUNativeFunctions::npu_dtype_cast(target, at::kInt);
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

tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::nll_loss_forward_out(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    at::Tensor& result,
    at::Tensor& total_weight) {
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_CHECK(
      target.dim() <= 1,
      "0D or 1D target tensor expected, multi-target not supported");
  auto no_batch_dim = self.dim() == 1 && target.dim() == 0;
  TORCH_CHECK(
      no_batch_dim || self.size(0) == target.size(0),
      "size mismatch (got input: ",
      self.sizes(),
      ", target: ",
      target.sizes(),
      ")");
  at::Tensor self_cp = self.dim() == 1 ? self.unsqueeze(0) : self;
  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  at::Tensor weight_tensor;
  if (weight.defined()) {
    weight_tensor = NpuUtils::format_contiguous(weight);
  } else {
    auto options = self_cp.options();
    weight_tensor = NPUNativeFunctions::ones(
        self_cp.size(1),
        optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(),
        options.device_opt(), options.pinned_memory_opt());
  }
  c10::SmallVector<int64_t, SIZE> output_size = {};
  if (reduction == at::Reduction::None) {
    output_size = {self_cp.size(0)};
  }
  OpPipeWithMultiOut<at::Tensor&, at::Tensor&> pipe(result, total_weight);
  pipe.FixOutputSizeAndFormat<0>({self_cp, target, weight_tensor}, self_cp, ACL_FORMAT_ND, output_size)
      .FixOutputSizeAndFormat<1>({self_cp, target, weight_tensor}, self_cp, ACL_FORMAT_ND, {})
      .Call([&self_cp, &target, &weight, &reduction, &ignore_index](at::Tensor& result, at::Tensor& total_weight) {
        nll_loss_forward_npu_nocheck(result, total_weight, self_cp, target, weight, reduction, ignore_index);})
      .ReturnRef<at::Tensor&, at::Tensor&>();
  if (self.dim() == 1 && reduction == at::Reduction::None) {
    result.squeeze_(0);
  }
  return std::tie(result, total_weight);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::nll_loss_forward(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  // ND case
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_CHECK(
      target.dim() <= 1,
      "0D or 1D target tensor expected, multi-target not supported");
  auto no_batch_dim = self.dim() == 1 && target.dim() == 0;
  TORCH_CHECK(
      no_batch_dim || self.size(0) == target.size(0),
      "size mismatch (got input: ",
      self.sizes(),
      ", target: ",
      target.sizes(),
      ")");
  at::Tensor self_cp = self.dim() == 1 ? self.unsqueeze(0) : self;
  c10::SmallVector<int64_t, SIZE> output_size = {};
  c10::SmallVector<int64_t, SIZE> totalWeightSize = {};
  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});

  if (reduction == at::Reduction::None) {
    output_size = {self_cp.size(0)};
  }

  // Special output, output' dim is <= 1 fixedly！
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(self_cp, output_size, ACL_FORMAT_ND);
  at::Tensor total_weight = OpPreparation::ApplyTensorWithFormat(self_cp, totalWeightSize, ACL_FORMAT_ND);

  nll_loss_forward_npu_nocheck(
      result, total_weight, self_cp, target, weight, reduction, ignore_index);
  if (self.dim() == 1 && reduction == at::Reduction::None) {
    result.squeeze_(0);
  }
  return tuple<at::Tensor, at::Tensor>(result, total_weight);
}

} // namespace native
} // namespace at_npu
