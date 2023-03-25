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

at::Tensor NPUNativeFunctions::binary_cross_entropy_with_logits(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& pos_weight_opt,
    int64_t reduction) {
  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  const at::Tensor& pos_weight = c10::value_or_else(pos_weight_opt, [] {return at::Tensor();});
  at::IntArrayRef outputSize;
  int64_t resultformat = CalcuOpUtil::GetTensorNpuFormat(self);

  if (reduction == at::Reduction::None) {
    outputSize = input_same_output_size(self);
  } else {
    outputSize = at::ArrayRef<int64_t>();
    resultformat = ACL_FORMAT_ND;
  }

  at::Tensor result = OpPreparation::ApplyTensorWithFormat(outputSize, self.options(), resultformat);
  at::Tensor weightTensor;
  if (weight.defined()) {
    weightTensor = NpuUtils::format_contiguous(weight);
    weightTensor = (weight.scalar_type() != self.scalar_type()) ? NPUNativeFunctions::npu_dtype_cast(weightTensor,
        self.scalar_type()) : weightTensor;
  } else {
    weightTensor = at::ones(self.sizes(), self.options());
  }

  at::Tensor posWeightTensor;
  if (pos_weight.defined()) {
    posWeightTensor = NpuUtils::format_contiguous(pos_weight);
    posWeightTensor = (posWeightTensor.scalar_type() != self.scalar_type()) ? NPUNativeFunctions::npu_dtype_cast(posWeightTensor,
        self.scalar_type()) : posWeightTensor;
  } else {
    posWeightTensor = at::ones(self.sizes(), self.options());
  }

  string reductionStr = CalcuOpUtil::GetReductionStr(reduction);
  OpCommand cmd;
  cmd.Name("SigmoidCrossEntropyWithLogitsV2")
      .Input(self.to(target.dtype()))
      .Input(target)
      .Input(weightTensor)
      .Input(posWeightTensor)
      .Output(result)
      .Attr("reduction", reductionStr)
      .Run();

  return result;
}
} // namespace native
} // namespace at_npu