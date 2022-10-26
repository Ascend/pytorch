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

at::Tensor& NPUNativeFunctions::binary_cross_entropy_out(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    at::Tensor& result) {
    const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  at::Tensor weightTensor = weight;
  if (!weight.defined()) {
      weightTensor = at::ones(self.sizes(), self.options());
  }

  string reductionStr = CalcuOpUtil::get_reduction_str(reduction);
  OpCommand cmd;
  cmd.Name("BinaryCrossEntropy")
      .Input(self)
      .Input(target)
      .Input(weightTensor)
      .Output(result)
      .Attr("reduction", reductionStr)
      .Run();
  return result;
}

at::Tensor NPUNativeFunctions::binary_cross_entropy(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction) {
    const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});  

  at::IntArrayRef outputSize;
  if (reduction == at::Reduction::None) {
    outputSize = input_same_output_size(self);
  } else {
    outputSize = at::ArrayRef<int64_t>();
  }

  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  if (self.numel() == 0) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::kFloat).fill_(0);
    result = result / 0;
    return result;
  }

  NPUNativeFunctions::binary_cross_entropy_out(self, target, weight, reduction, result);
  return result;
}
} // namespace native
} // namespace at_npu