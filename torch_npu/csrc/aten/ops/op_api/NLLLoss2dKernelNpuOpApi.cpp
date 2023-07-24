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

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {

namespace {

tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>> nll_loss2d_npu_output_size(
    const at::Tensor& self, const at::Tensor& target, int64_t reduction, int64_t ignore_index) {
  c10::SmallVector<int64_t, SIZE> outputSize;
  c10::SmallVector<int64_t, SIZE> totalWeightSize;

  if (reduction == at::Reduction::None) {
    outputSize = {self.size(0), self.size(2), self.size(3)};
  }

  return tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>(outputSize, totalWeightSize);
}
}  // namespace

tuple<at::Tensor&, at::Tensor&> NPUNativeOpApiFunctions::nll_loss2d_forward_out(
    const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight_opt, int64_t reduction,
    int64_t ignore_index, at::Tensor& result, at::Tensor& total_weight) {
  DO_COMPATIBILITY(aclnnNLLLoss2d, NPUNativeFunctions::nll_loss2d_forward_out(self, target, weight_opt, reduction,
                                                                              ignore_index, result, total_weight));
  at::Tensor weight_tensor = c10::value_or_else(weight_opt, [] { return at::Tensor(); });
  if (!weight_tensor.defined()) {
    weight_tensor = at::ones(self.size(1), self.options());
  }

  OpPreparation::CheckMemory({self, target, weight_tensor}, {result, total_weight});
  EXEC_NPU_CMD(aclnnNLLLoss2d, self, target, weight_tensor, reduction, ignore_index, result, total_weight);
  return tuple<at::Tensor&, at::Tensor&>(result, total_weight);
}

tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::nll_loss2d_forward(const at::Tensor& self,
                                                                          const at::Tensor& target,
                                                                          const c10::optional<at::Tensor>& weight_opt,
                                                                          int64_t reduction, int64_t ignore_index) {
  DO_COMPATIBILITY(aclnnNLLLoss2d,
                   NPUNativeFunctions::nll_loss2d_forward(self, target, weight_opt, reduction, ignore_index));
  // calculate the output size
  auto outputSizes = nll_loss2d_npu_output_size(self, target, reduction, ignore_index);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, std::get<0>(outputSizes));
  at::Tensor total_weight = OpPreparation::ApplyTensorWithoutFormat(self, std::get<1>(outputSizes));

  // calculate the output result of the NPU
  NPUNativeOpApiFunctions::nll_loss2d_forward_out(self, target, weight_opt, reduction, ignore_index, result,
                                                  total_weight);

  return tuple<at::Tensor, at::Tensor>(result, total_weight);
}

}  // namespace native
}  // namespace at_npu
